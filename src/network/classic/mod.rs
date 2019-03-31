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
use crate::network::cl_context::ClContext;
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
    cl: ClContext,
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
        let cl = init_cl::<T>();

        let layers = Layers::<T>::new(weights);

        let (conv1, conv2, dense3) = (&layers.conv1, &layers.conv2, &layers.dense3);

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf = conv1.create_wgts_buf(&cl.conv_queue());
        let conv2_wgts_buf = conv2.create_wgts_buf(&cl.conv_queue());
        let dense3_wgts_buf = dense3.create_wgts_buf(&cl.cpu_queue);

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let mut conv_bufs = create_buffer_chain(&[conv1, conv2], &cl.conv_queue());
        let (dense3_in_buf, dense3_out_buf) =
            dense3.create_io_bufs(flags::MEM_READ_WRITE, flags::MEM_WRITE_ONLY, &cl.cpu_queue);

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = conv1.create_kernel(
            "conv_relu_1",
            &conv_bufs[0],
            &conv_bufs[1],
            &conv1_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &cl.program,
            &cl.conv_queue(),
        );

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = conv2.create_kernel(
            "conv_relu_2",
            &conv_bufs[1],
            &conv_bufs[2],
            &conv2_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &cl.program,
            &cl.conv_queue(),
        );

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let dense3_kernel = dense3.create_kernel(
            "mtx_mul",
            &dense3_in_buf,
            &dense3_out_buf,
            &dense3_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &cl.program,
            &cl.cpu_queue,
        );

        // Log info about the created network
        info!(
            "Classic layers 1-2 will be run on {}, layer 3 will be run on {}, layers 4-5 will be run on host (Rust).",
            cl.conv_queue().device().name().unwrap(),
            cl.cpu_queue.device().name().unwrap(),
        );

        // TODO: see if queue finish here has an impact on anything

        // Move and store the first and last buffer
        let mut buf_drain = conv_bufs.drain(..);
        let input_buf = buf_drain.next().unwrap();
        let conv2_out_buf = buf_drain.next_back().unwrap();

        ClNetwork::<T> {
            cl,
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
            self.conv1_kernel
                .cmd()
                .queue(&self.cl.conv_queue())
                .enq()
                .unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv2_kernel
                .cmd()
                .queue(&self.cl.conv_queue())
                .enq()
                .unwrap();

            self.conv2_out_buf
                .copy(&self.dense3_in_buf, None, None)
                .queue(&self.cl.conv_queue())
                .enew(&mut event_list)
                .enq()
                .unwrap();

            // Enqueue the 3rd layer (fully-connected)
            self.dense3_kernel
                .cmd()
                .queue(&self.cl.cpu_queue)
                .ewait(&event_list)
                .enq()
                .unwrap();
        }
        // Wait for all on-device calculations to finish
        self.cl.cpu_queue.finish().unwrap();

        let dense3_out = &unsafe { read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(dense5_out)
    }
}

fn init_cl<T>() -> ClContext
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

    let all_devices = Device::list(platform, None).unwrap();
    if all_devices.is_empty() {
        panic!("running the classic network requires at least one OpenCL device");
    }

    // Prefer a GPU device for the convolution device
    let conv_device = *match all_devices
        .iter()
        .find(|&&dt| device_type_of(dt).contains(DeviceType::empty().gpu()))
    {
        Some(d) => d,
        None => all_devices.first().unwrap(),
    };
    // Prefer a CPU device for the matrix multiplication
    let matmul_device = *match all_devices
        .iter()
        .find(|&&dt| device_type_of(dt).contains(DeviceType::empty().cpu()))
    {
        Some(d) => d,
        None => all_devices.first().unwrap(),
    };

    let single_device = conv_device == matmul_device;
    let selected_devices = if single_device {
        vec![conv_device]
    } else {
        vec![conv_device, matmul_device]
    };

    let context = Context::builder()
        .platform(platform)
        .devices::<&[Device]>(&selected_devices)
        .build()
        .unwrap();

    let mut program_b = Program::builder();
    program_b.devices(&selected_devices);

    // Add default compiler options
    configure_program::<T>(&mut program_b);

    // Input the kernel source files
    for src in sources {
        program_b.src(src);
    }

    let program = program_b.build(&context).unwrap();

    let profile_flag = None;

    // Create the queue for the default device
    let (gpu_queue, cpu_queue) = if single_device {
        (
            None,
            Queue::new(&context, conv_device, profile_flag).unwrap(),
        )
    } else {
        let gpu_queue = Queue::new(&context, conv_device, profile_flag).unwrap();
        let cpu_queue = Queue::new(&context, matmul_device, profile_flag).unwrap();
        (Some(gpu_queue), cpu_queue)
    };

    ClContext {
        gpu_queue,
        cpu_queue,
        program,
        _context: context,
    }
}
