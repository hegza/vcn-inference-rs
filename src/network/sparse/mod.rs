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
    queue: Queue,
    input_shape: ImageGeometry,
    input_buf: Buffer<T>,
    conv1_kernel: Kernel,
    conv2_kernel: Kernel,
    conv2_out_buf: Buffer<T>,
    sparse3: SparseLayer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl ClNetwork<f32> {
    pub fn new(weights: Weights<f32>) -> ClNetwork<f32> {
        let (queue, program, _context) = init_cl::<f32>();

        let layers = Layers::<f32>::new(weights);

        let (conv1, conv2) = (&layers.conv1, &layers.conv2);

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf = conv1.create_wgts_buf(&queue);
        let conv2_wgts_buf = conv2.create_wgts_buf(&queue);

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let mut conv_bufs = create_buffer_chain(&[conv1, conv2], &queue);

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = conv1.create_kernel(
            "conv_relu_1",
            &conv_bufs[0],
            &conv_bufs[1],
            &conv1_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue,
        );

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = conv2.create_kernel(
            "conv_relu_2",
            &conv_bufs[1],
            &conv_bufs[2],
            &conv2_wgts_buf,
            LocalWorkSizePolicy::UseDefault,
            &program,
            &queue,
        );

        // Log info about the created network
        debug!("A sparse network was created with layers:");
        debug!(
            "  {:?} (devices = {:?})",
            conv1,
            conv_relu1.devices().unwrap()
        );
        debug!(
            "  {:?} (devices = {:?})",
            conv2,
            conv_relu2.devices().unwrap()
        );
        debug!("  {:?} (device = host)", layers.sparse3);
        debug!("  {:?} (device = host)", layers.dense4);
        debug!("  {:?} (device = host)", layers.dense5);
        info!(
            "Sparse layers 1-2 will be run on {}, layers 3-5 will be run on host (Rust).",
            device_id_to_name(conv_relu1.devices().unwrap()[0])
        );

        // Move and store the first and last buffer
        let mut buf_drain = conv_bufs.drain(..);
        let input_buf = buf_drain.next().unwrap();
        let conv2_out_buf = buf_drain.next_back().unwrap();

        ClNetwork::<f32> {
            queue,
            input_shape: *conv1.input_shape(),
            input_buf,
            conv1_kernel: conv_relu1,
            conv2_kernel: conv_relu2,
            conv2_out_buf,
            sparse3: layers.sparse3,
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
        unsafe {
            map_to_buf(&self.input_buf, input_data).unwrap();

            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            self.conv1_kernel.cmd().queue(&self.queue).enq().unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv2_kernel.cmd().queue(&self.queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        self.queue.finish().unwrap();

        let sparse3_out = relu(
            self.sparse3
                .compute(unsafe { &read_buf(&self.conv2_out_buf).unwrap() }),
        );

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&sparse3_out));

        // Run the 5th layer (fully-connected)
        softmax(self.dense5.compute(&dense4_out))
    }
}

fn init_cl<T>() -> (Queue, Program, Context)
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
    let device = select_device(DevicePreference::PreferGpu);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    let mut program_b = Program::builder();

    // Add default compiler options
    configure_program::<T, Device>(&mut program_b, device);

    // Input the kernel source files
    for src in sources {
        program_b.src(src);
    }

    let program = program_b.build(&context).unwrap();

    // Create the queue for the default device
    let profile_flag = None;
    let queue = Queue::new(&context, device, profile_flag).unwrap();

    (queue, program, context)
}
