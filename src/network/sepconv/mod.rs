mod layers;
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
    // TODO: revisit the names here
    side: 96,
    vconv1_blockdim_x: 32,
    vconv1_blockdim_y: 8,
    hconv1_blockdim_y: 4,
    vconv2_blockdim_x: 16,
    hconv2_blockdim_y: 4,
    kernel_len: 5,
    num_channels: 3,
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
    queue: Queue,
    pub in_buf: Buffer<T>,
    krn_vconv1: Kernel,
    krn_hconv1: Kernel,
    krn_max_pool1: Kernel,
    krn_vconv2: Kernel,
    krn_hconv2: Kernel,
    krn_max_pool2: Kernel,
    krn_dense3: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> ClNetwork<T>
where
    T: Coeff + ReadCsv,
{
    pub fn new(wgts: Weights<T>) -> ClNetwork<T> {
        let mut p = SEPCONV_HYPER_PARAMS.clone();

        // HACK: Reduce dimensions of overshot layers
        ClNetwork::<T>::fix_params_for_default_gpu(&mut p);

        let layers: Layers<T> = Layers::new(wgts);

        // Init OpenCL
        let flags = ClNetwork::<T>::compile_flags(&p, &layers);
        let (queue, program, _context) = cl_util::init::<T>(
            &[
                "src/cl/sepconv.cl",
                "src/cl/max_pool.cl",
                "src/cl/mtx_mul.cl",
            ],
            &flags.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            None,
        );

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
        let wgts_bufs = create_weights_bufs(&[vconv1, hconv1, vconv2, hconv2, dense3], &queue);

        // Allocate memory on-device for the I/O buffers
        let mut bufs = create_buffer_chain(
            &[vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, dense3],
            &queue,
        );

        // Create kernels
        let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
        let dev_max_wgs = cl_util::max_wgs(Some(&primary_device));

        let kernels = {
            let mut b = ClKernelChainBuilder::<T>::new(&bufs, &wgts_bufs, &program, queue.clone());
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
                b.build_iow_kernel(dense3, "mtx_mul", LocalWorkSizePolicy::UseDefault),
            )
        };

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

        // Move and store the first and last buffer
        let mut buf_drain = bufs.drain(..);
        let in_buf = buf_drain.next().unwrap();
        let dense3_out_buf = buf_drain.next_back().unwrap();

        ClNetwork {
            queue,
            in_buf,
            krn_vconv1: kernels.0,
            krn_hconv1: kernels.1,
            krn_max_pool1: kernels.2,
            krn_vconv2: kernels.3,
            krn_hconv2: kernels.4,
            krn_max_pool2: kernels.5,
            krn_dense3: kernels.6,
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
        let q = &self.queue;

        /*let buf = */
        unsafe {
            cl_util::map_to_buf(&self.in_buf, input_data).unwrap();

            self.krn_vconv1.cmd().queue(q).enq().unwrap();
            self.krn_hconv1.cmd().queue(q).enq().unwrap();
            self.krn_max_pool1.cmd().queue(q).enq().unwrap();
            self.krn_vconv2.cmd().queue(q).enq().unwrap();
            self.krn_hconv2.cmd().queue(q).enq().unwrap();
            self.krn_max_pool2.cmd().queue(q).enq().unwrap();
            self.krn_dense3.cmd().queue(q).enq().unwrap();
        }

        // Wait for all on-device calculations to finish
        q.finish().unwrap();

        // Load the 3rd layer from the GPU and Run ReLU on it
        let dense3_out = unsafe { cl_util::read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(&dense5_out)
    }
}

impl Predict<i8> for ClNetwork<i8> {
    // Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[i8]) -> Vec<f32> {
        let q = &self.queue;

        /*let buf = */
        unsafe {
            cl_util::map_to_buf(&self.in_buf, input_data).unwrap();

            self.krn_vconv1.cmd().queue(q).enq().unwrap();
            self.krn_hconv1.cmd().queue(q).enq().unwrap();
            self.krn_max_pool1.cmd().queue(q).enq().unwrap();
            self.krn_vconv2.cmd().queue(q).enq().unwrap();
            self.krn_hconv2.cmd().queue(q).enq().unwrap();
            self.krn_max_pool2.cmd().queue(q).enq().unwrap();
            self.krn_dense3.cmd().queue(q).enq().unwrap();
        }

        // Wait for all on-device calculations to finish
        q.finish().unwrap();

        // Load the 3rd layer from the GPU and Run ReLU on it
        let dense3_out = unsafe { cl_util::read_buf(&self.dense3_out_buf).unwrap() };

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(&dense5_out)
    }
}
