mod layers;
#[cfg(test)]
mod test;
mod weights;

pub use self::layers::*;
pub use self::weights::*;
use super::Predict;
use crate::cl_util::*;
use crate::geometry::{ImageGeometry, PaddedSquare, Square};
use crate::layers::*;
use crate::math::{relu, softmax};
use crate::util::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::flags::*;
use ocl::{flags, Buffer, Context, Device, EventList, Kernel, OclPrm, Platform, Program, Queue};
use std::fs;
use std::io::prelude::*;

pub struct ClNetwork<T>
where
    T: Coeff,
{
    queue_a: Queue,
    queue_b: Queue,
    input_shape: ImageGeometry,
    input_buf: Buffer<T>,
    conv1_kernel: Kernel,
    conv2_kernel: Kernel,
    conv2_out_buf: Buffer<T>,
    dense3_in_buf: Buffer<T>,
    dense3_kernel: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> ClNetwork<T>
where
    T: Coeff,
{
    pub fn new(weights: Weights<T>) -> ClNetwork<T> {
        let (queue_a, queue_b, program, _context) = init_cl::<T>();

        let layers = Layers::<T>::new(weights);

        let (conv1, conv2, dense3) = (&layers.conv1, &layers.conv2, &layers.dense3);

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf = conv1.create_wgts_buf(&queue_a);
        let conv2_wgts_buf = conv2.create_wgts_buf(&queue_a);
        let dense3_wgts_buf = dense3.create_wgts_buf(&queue_b);

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let mut conv_bufs = create_buffer_chain(&[conv1, conv2], &queue_a);
        let (dense3_in_buf, dense3_out_buf) =
            dense3.create_io_bufs(flags::MEM_READ_WRITE, flags::MEM_WRITE_ONLY, &queue_b);

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = conv1.create_kernel(
            "conv_relu_1",
            &conv_bufs[0],
            &conv_bufs[1],
            &conv1_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue_a,
        );

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = conv2.create_kernel(
            "conv_relu_2",
            &conv_bufs[1],
            &conv_bufs[2],
            &conv2_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue_a,
        );

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let dense3_kernel = dense3.create_kernel(
            "mtx_mul",
            &dense3_in_buf,
            &dense3_out_buf,
            &dense3_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue_b,
        );

        // TODO: see if queue finish here has an impact on anything

        // Move and store the first and last buffer
        let mut buf_drain = conv_bufs.drain(..);
        let input_buf = buf_drain.next().unwrap();
        let conv2_out_buf = buf_drain.next_back().unwrap();

        ClNetwork::<T> {
            queue_a,
            queue_b,
            input_shape: *conv1.input_shape(),
            input_buf,
            conv1_kernel: conv_relu1,
            conv2_kernel: conv_relu2,
            conv2_out_buf,
            dense3_in_buf,
            dense3_kernel,
            dense3_out_buf,
            dense4: layers.dense4,
            dense5: layers.dense5,
        }
    }
    // HACK: the input shape should already be accessible from a more primitive type than this
    pub fn input_shape(&self) -> &ImageGeometry {
        &self.input_shape
    }
}

impl<T> Predict<T> for ClNetwork<T>
where
    T: CoeffFloat,
{
    fn predict(&self, input_data: &[T]) -> Vec<f32> {
        let mut event_list = EventList::new();

        unsafe {
            map_to_buf(&self.input_buf, input_data).unwrap();

            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            self.conv1_kernel.cmd().queue(&self.queue_a).enq().unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv2_kernel.cmd().queue(&self.queue_a).enq().unwrap();

            self.conv2_out_buf
                .copy(&self.dense3_in_buf, None, None)
                .queue(&self.queue_a)
                .enew(&mut event_list)
                .enq()
                .unwrap();

            // Enqueue the 3rd layer (fully-connected)
            self.dense3_kernel
                .cmd()
                .queue(&self.queue_b)
                .ewait(&event_list)
                .enq()
                .unwrap();
        }
        // Wait for all on-device calculations to finish
        self.queue_b.finish().unwrap();

        let dense3_out = &unsafe { read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(dense5_out)
    }
}

fn init_cl<T>() -> (Queue, Queue, Program, Context)
where
    T: Coeff,
{
    // Init OpenCL
    let kernel_files = ["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"];

    let sources = kernel_files
        .iter()
        .map(|&fname| {
            let mut f = fs::File::open(&fname).unwrap();
            let mut contents = String::new();
            f.read_to_string(&mut contents).unwrap();
            contents
        })
        .collect::<Vec<String>>();

    let platform = Platform::default();

    // Prefer GPU a device for the convolution device
    let device_a = select_device(DevicePreference::PreferGpu);
    // Prefer CPU for the dense-3 matrix multiplication
    let device_b = select_device(DevicePreference::RequireCpu);

    let context = Context::builder()
        .platform(platform)
        .devices::<&[Device]>(&[device_a, device_b])
        .build()
        .unwrap();

    let mut program_b = Program::builder();

    // Add default compiler options
    configure_program::<T, &[Device]>(&mut program_b, &[device_a, device_b]);

    // Input the kernel source files
    for src in sources {
        program_b.src(src);
    }

    let program = program_b.build(&context).unwrap();

    // Create the queue for the default device
    let profile_flag = None;
    let queue_a = Queue::new(&context, device_a, profile_flag).unwrap();
    let queue_b = Queue::new(&context, device_b, profile_flag).unwrap();

    (queue_a, queue_b, program, context)
}
