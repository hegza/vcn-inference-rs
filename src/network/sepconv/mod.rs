mod layers;
#[cfg(test)]
mod test;
mod weights;

pub use self::layers::*;
pub use self::weights::*;
use super::{Predict, PRIMARY_DEVICE};
use cl_util;
use geometry::{ImageGeometry, PaddedSquare, Square};
use layers::*;
use math::{relu, softmax};
use ocl;
use ocl::{
    builders::*, enums::*, flags, flags::*, Buffer, Context, Device, EventList, Kernel, OclPrm,
    Platform, Program, Queue, SpatialDims,
};
use std::fs;
use std::io::prelude::*;
use util::*;

pub const SEPCONV_HYPER_PARAMS: SepconvHyperParams = SepconvHyperParams {
    side: 96,
    // TODO: these 5 are an implementation detail and should not be exposed here
    // These should probably be inferred from the OpenCL API info
    vconv1_blockdim_x: 32,
    vconv1_blockdim_y: 8,
    hconv1_blockdim_y: 4,
    vconv2_blockdim_x: 16,
    hconv2_blockdim_y: 4,
    kernel_len: 5,
    num_channels: 3,
    // TODO: also implementation detail; should not be exposed
    conv_kernel_split: 7,
    num_conv_fms: 32,
    fully_connected_const: 100,
    num_output_classes: 4,
};

#[derive(Clone, Debug)]
pub struct SepconvHyperParams {
    pub side: usize,
    pub vconv1_blockdim_x: usize,
    pub vconv1_blockdim_y: usize,
    pub hconv1_blockdim_y: usize,
    pub vconv2_blockdim_x: usize,
    pub hconv2_blockdim_y: usize,
    pub kernel_len: usize,
    pub num_channels: usize,
    pub conv_kernel_split: usize,
    pub num_conv_fms: usize,
    pub fully_connected_const: usize,
    pub num_output_classes: usize,
}

pub struct ClNetwork<T>
where
    T: Coeff,
{
    queue_a: Queue,
    queue_b: Queue,
    pub in_buf: Buffer<T>,
    krn_vconv1: Kernel,
    krn_hconv1: Kernel,
    krn_max_pool1: Kernel,
    krn_vconv2: Kernel,
    krn_hconv2: Kernel,
    krn_max_pool2: Kernel,
    dense3_in_buf: Buffer<T>,
    krn_dense3: Kernel,
    device_a_out_buf: Buffer<T>,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> ClNetwork<T>
where
    T: Coeff,
{
    pub fn new(wgts: Weights<T>) -> ClNetwork<T> {
        let mut p = SEPCONV_HYPER_PARAMS.clone();

        // HACK: Reduce dimensions of overshot layers
        ClNetwork::<T>::fix_params_for_default_gpu(&mut p);

        let layers: Layers<T> = Layers::new(wgts);

        // Init OpenCL
        let flags = ClNetwork::<T>::compile_flags(&p, &layers);
        let (queue_a, queue_b, device_a, _device_b, program, _context) =
            init_cl::<T>(&flags.iter().map(AsRef::as_ref).collect::<Vec<&str>>());

        // Create shorthands
        let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, dense3) = (
            &layers.vconv1,
            &layers.hconv1,
            &layers.mxp1,
            &layers.vconv2,
            &layers.hconv2,
            &layers.mxp2,
            &layers.dense3,
        );

        // Allocate read-only memory on-device for the weights buffers
        let conv_wgts_bufs = create_weights_bufs(&[vconv1, hconv1, vconv2, hconv2], &queue_a);
        let dense3_wgts_buf = dense3.create_wgts_buf(&queue_b);

        // Allocate memory on-device for the I/O buffers
        let mut conv_bufs =
            create_buffer_chain(&[vconv1, hconv1, mxp1, vconv2, hconv2, mxp2], &queue_a);
        let (dense3_in_buf, dense3_out_buf) =
            dense3.create_io_bufs(flags::MEM_READ_WRITE, flags::MEM_WRITE_ONLY, &queue_b);

        let dev_max_wgs = cl_util::max_wgs(Some(&device_a));

        // Create kernels
        let kernels = {
            let mut b = ClKernelChainBuilder::<T>::new(
                &conv_bufs,
                &conv_wgts_bufs,
                &program,
                queue_a.clone(),
            );
            (
                b.build_iow_kernel(
                    vconv1,
                    "col_conv",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(
                        p.vconv1_blockdim_x,
                        p.vconv1_blockdim_y,
                    )),
                ),
                b.build_iow_kernel(
                    hconv1,
                    "row_conv",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side, p.hconv1_blockdim_y)),
                ),
                b.build_io_kernel(
                    mxp1,
                    "max_pool_1",
                    LocalWorkSizePolicy::Infer { dev_max_wgs },
                ),
                b.build_iow_kernel(
                    vconv2,
                    "col_conv_2",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(
                        p.vconv2_blockdim_x,
                        p.vconv1_blockdim_y,
                    )),
                ),
                b.build_iow_kernel(
                    hconv2,
                    "row_conv_2",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y)),
                ),
                b.build_io_kernel(
                    mxp2,
                    "max_pool_2",
                    LocalWorkSizePolicy::Infer { dev_max_wgs },
                ),
            )
        };

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
        let mut device_a_buf_drain = conv_bufs.drain(..);
        let in_buf = device_a_buf_drain.next().unwrap();
        let device_a_out_buf = device_a_buf_drain.next_back().unwrap();

        ClNetwork {
            queue_a,
            queue_b,
            in_buf,
            krn_vconv1: kernels.0,
            krn_hconv1: kernels.1,
            krn_max_pool1: kernels.2,
            krn_vconv2: kernels.3,
            krn_hconv2: kernels.4,
            krn_max_pool2: kernels.5,
            device_a_out_buf,
            dense3_in_buf,
            krn_dense3: dense3_kernel,
            dense3_out_buf,
            dense4: layers.dense4,
            dense5: layers.dense5,
        }
    }
    pub fn fix_params_for_default_gpu(p: &mut SepconvHyperParams) {
        // HACK: Max-work-size is not enough, use halved dimensions for hconv1
        let max_wgs = cl_util::max_wgs(None);
        if p.side * p.hconv1_blockdim_y > max_wgs {
            p.hconv1_blockdim_y /= 2;
            warn!("using halved dimension for horizontal convolution 1");
        }
    }
    pub fn compile_flags(p: &SepconvHyperParams, layers: &Layers<T>) -> Vec<String> {
        let max_wgs = cl_util::max_wgs(Some(&PRIMARY_DEVICE));
        let mxp1_lws = layers.mxp1.lws_hint(max_wgs).to_lens().unwrap()[0];
        let mxp2_lws = layers.mxp2.lws_hint(max_wgs).to_lens().unwrap()[0];

        vec![
            format!("-D WIDTH={}", p.side as i32),
            format!("-D HEIGHT={}", p.side as i32),
            format!("-D MP1_BLOCK_DIM={}", mxp1_lws as i32),
            format!("-D MP2_BLOCK_DIM={}", mxp2_lws as i32),
            format!("-D ROWS_BLOCKDIM_Y={}", p.hconv1_blockdim_y as i32),
            format!("-D ROWS_2_BLOCKDIM_Y={}", p.hconv2_blockdim_y as i32),
            format!("-D INJECT_RELU_AFTER_MXP={}", 1 as i32),
        ]
    }
}

impl<T> Predict<T> for ClNetwork<T>
where
    T: CoeffFloat,
{
    // Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[T]) -> Vec<f32> {
        let q = &self.queue_a;
        let mut event_list = EventList::new();

        unsafe {
            cl_util::map_to_buf(&self.in_buf, input_data).unwrap();

            self.krn_vconv1.cmd().queue(q).enq().unwrap();
            self.krn_hconv1.cmd().queue(q).enq().unwrap();
            self.krn_max_pool1.cmd().queue(q).enq().unwrap();
            self.krn_vconv2.cmd().queue(q).enq().unwrap();
            self.krn_hconv2.cmd().queue(q).enq().unwrap();
            self.krn_max_pool2.cmd().queue(q).enq().unwrap();

            self.device_a_out_buf
                .copy(&self.dense3_in_buf, None, None)
                .queue(q)
                .enew(&mut event_list)
                .enq()
                .unwrap();

            // Enqueue the 3rd layer (fully-connected)
            self.krn_dense3
                .cmd()
                .queue(&self.queue_b)
                .ewait(&event_list)
                .enq()
                .unwrap();
        }

        // Wait for all on-device calculations to finish
        self.queue_b.finish().unwrap();

        // Load the 3rd layer from the GPU and Run ReLU on it
        let dense3_out = unsafe { cl_util::read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(dense5_out)
    }
}

fn init_cl<T>(flags: &[&str]) -> (Queue, Queue, Device, Device, Program, Context)
where
    T: Coeff,
{
    // Init OpenCL
    let kernel_files = [
        "src/cl/sepconv.cl",
        "src/cl/max_pool.cl",
        "src/cl/mtx_mul.cl",
    ];

    let sources = kernel_files
        .iter()
        .map(|&fname| {
            let mut f = fs::File::open(&fname).unwrap();
            let mut contents = String::new();
            f.read_to_string(&mut contents).unwrap();
            contents
        }).collect::<Vec<String>>();

    let platform = Platform::default();

    // Prefer GPU a device for the convolution device
    let device_a = cl_util::select_device(cl_util::DevicePreference::PreferGpu);
    // Prefer CPU for the dense-3 matrix multiplication
    let device_b = cl_util::select_device(cl_util::DevicePreference::RequireCpu);

    let context = Context::builder()
        .platform(platform)
        .devices::<&[Device]>(&[device_a, device_b])
        .build()
        .unwrap();

    let mut program_b = Program::builder();

    // Add default compiler options
    cl_util::configure_program::<T, &[Device]>(&mut program_b, &[device_a, device_b]);
    for &opt in flags {
        program_b.cmplr_opt(opt);
    }

    // Input the kernel source files
    for src in sources {
        program_b.src(src);
    }

    let program = program_b.build(&context).unwrap();

    // Create the queue for the default device
    let profile_flag = None;
    let queue_a = Queue::new(&context, device_a, profile_flag).unwrap();
    let queue_b = Queue::new(&context, device_b, profile_flag).unwrap();

    (queue_a, queue_b, device_a, device_b, program, context)
}
