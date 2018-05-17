use super::*;
use ocl::{Device, Platform, SpatialDims};
use geometry::*;
use ndarray::{Array, ShapeBuilder, arr2};
use std::ops::{Deref, Index};

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

pub type Layers<T> = (
    VConvLayer<T>,
    HConvLayer<T>,
    MaxpoolLayer,
    VConvLayer<T>,
    HConvLayer<T>,
    MaxpoolLayer,
    DenseLayer<T>,
    DenseLayer<T>,
    DenseLayer<T>,
);

pub struct Weights<T>(
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
);

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

pub const WEIGHTS_F32_DIR: &str = "input/weights/sepconv-96-97";
pub const WEIGHTS_I8_DIR: &str = "input/weights/sepconv-96-97/i8-converted";

pub struct SepconvNetwork<T>
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

impl<T> SepconvNetwork<T>
where
    T: Coeff + ReadCsv,
{
    pub fn create_layers<W>(p: &SepconvHyperParams, wgts: W) -> Layers<T>
    where
        W: SepconvWeights<T>,
    {
        let wgts = wgts.into_tuple();
        // TODO: weights are read as T, independent of what's actually stored
        let in_shape = ImageGeometry::new(p.side, p.num_channels);
        let vconv1 = VConvLayer::new(p.kernel_len, in_shape, p.conv_kernel_split, wgts.0);
        let hconv1 = HConvLayer::new(p.kernel_len, *vconv1.output_shape(), p.num_conv_fms, wgts.1);
        let mxp1 = MaxpoolLayer::new(*hconv1.output_shape(), 2);
        let vconv2 = VConvLayer::new(
            p.kernel_len,
            *mxp1.output_shape(),
            p.conv_kernel_split,
            wgts.2,
        );
        let hconv2 = HConvLayer::new(p.kernel_len, *vconv2.output_shape(), p.num_conv_fms, wgts.3);
        let mxp2 = MaxpoolLayer::new(*hconv2.output_shape(), 2);

        let dense3 = DenseLayer::new(mxp2.num_out(), p.fully_connected_const, wgts.4);
        let dense4 = DenseLayer::new(dense3.num_out(), p.fully_connected_const, wgts.5);
        let dense5 = DenseLayer::new(dense4.num_out(), p.num_output_classes, wgts.6);

        (
            vconv1,
            hconv1,
            mxp1,
            vconv2,
            hconv2,
            mxp2,
            dense3,
            dense4,
            dense5,
        )
    }
    pub fn new(wgts: Weights<T>) -> SepconvNetwork<T> {
        let mut p = SEPCONV_HYPER_PARAMS.clone();

        // HACK: Reduce dimensions of overshot layers
        SepconvNetwork::<T>::fix_params_for_default_gpu(&mut p);

        let layers: Layers<T> = SepconvNetwork::create_layers(&p, wgts);

        // Init OpenCL
        let flags = SepconvNetwork::<T>::compile_flags(&p, &layers);
        let (queue, program, _context) = cl::init::<T>(
            &["sepconv.cl", "max_pool.cl", "mtx_mul.cl"],
            &flags.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            None,
        );

        // Create shorthands (and move)
        let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, dense3, dense4, dense5) = layers;

        // Allocate read-only memory on-device for the weights buffers
        let wgts_bufs = create_weights_bufs(&[&vconv1, &hconv1, &vconv2, &hconv2, &dense3], &queue);

        // Allocate memory on-device for the I/O buffers
        let mut bufs = create_buffer_chain(
            &[&vconv1, &hconv1, &mxp1, &vconv2, &hconv2, &mxp2, &dense3],
            &queue,
        );

        // Create kernels
        let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
        let dev_max_wgs = cl::max_wgs(Some(&primary_device));

        let kernels = {
            let mut b = ClKernelChainBuilder::<T>::new(&bufs, &wgts_bufs, &program, queue.clone());
            (
                b.build_iow_kernel(
                    &vconv1,
                    "col_conv",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(
                        p.vconv1_blockdim_x,
                        p.vconv1_blockdim_y,
                    )),
                ),
                b.build_iow_kernel(
                    &hconv1,
                    "row_conv",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side, p.hconv1_blockdim_y)),
                ),
                b.build_io_kernel(
                    &mxp1,
                    "max_pool_1",
                    LocalWorkSizePolicy::Infer { dev_max_wgs },
                ),
                b.build_iow_kernel(
                    &vconv2,
                    "col_conv_2",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(
                        p.vconv2_blockdim_x,
                        p.vconv1_blockdim_y,
                    )),
                ),
                b.build_iow_kernel(
                    &hconv2,
                    "row_conv_2",
                    LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y)),
                ),
                b.build_io_kernel(
                    &mxp2,
                    "max_pool_2",
                    LocalWorkSizePolicy::Infer { dev_max_wgs },
                ),
                b.build_iow_kernel(&dense3, "mtx_mul", LocalWorkSizePolicy::UseDefault),
            )
        };

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

        // Move and store the first and last buffer
        let mut buf_drain = bufs.drain(..);
        let in_buf = buf_drain.next().unwrap();
        let dense3_out_buf = buf_drain.next_back().unwrap();

        SepconvNetwork {
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
            dense4,
            dense5,
        }
    }
    pub fn fix_params_for_default_gpu(p: &mut SepconvHyperParams) {
        // HACK: Max-work-size is not enough, use halved dimensions for hconv1
        let max_wgs = cl::max_wgs(None);
        if p.side * p.hconv1_blockdim_y > max_wgs {
            p.hconv1_blockdim_y /= 2;
            warn!("using halved dimension for horizontal convolution 1");
        }
    }
    pub fn compile_flags(p: &SepconvHyperParams, layers: &Layers<T>) -> Vec<String> {
        let max_wgs = cl::max_wgs(Some(&PRIMARY_DEVICE));
        let mxp1_lws = layers.mxp1().lws_hint(max_wgs).to_lens().unwrap()[0];
        let mxp2_lws = layers.mxp2().lws_hint(max_wgs).to_lens().unwrap()[0];

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

impl<T> Predict<T> for SepconvNetwork<T>
where
    T: Coeff,
{
    // Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[T]) -> Vec<f32> {
        let q = &self.queue;

        /*let buf = */
        unsafe {
            cl::map_to_buf(&self.in_buf, input_data).unwrap();

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
        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(&self.dense4.compute(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.compute(&dense4_out);

        softmax(&dense5_out)
    }
}

trait SepconvLayers<T>
where
    T: Coeff,
{
    fn mxp1(&self) -> &MaxpoolLayer;
    fn mxp2(&self) -> &MaxpoolLayer;
}

impl<T> SepconvLayers<T> for Layers<T>
where
    T: Coeff,
{
    fn mxp1(&self) -> &MaxpoolLayer {
        &self.2
    }
    fn mxp2(&self) -> &MaxpoolLayer {
        &self.5
    }
}

impl Default for Weights<f32> {
    fn default() -> Weights<f32> {
        Weights(
            // Load the weights for the convolutional layers
            f32::read_csv(&format!("{}/vcr1-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/hcr1-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/vcr2-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/hcr2-f32.csv", WEIGHTS_F32_DIR)),
            // Load the weights for the dense layers
            f32::read_csv(&format!("{}/fc3-f32-nchw.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/fc4-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/fc5-f32.csv", WEIGHTS_F32_DIR)),
        )
    }
}

impl Default for Weights<i8> {
    fn default() -> Weights<i8> {
        Weights(
            // Load the weights for the convolutional layers
            i8::read_csv(&format!("{}/vcr1-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/hcr1-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/vcr2-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/hcr2-i8.csv", WEIGHTS_I8_DIR)),
            // Load the weights for the dense layers
            i8::read_csv(&format!("{}/fc3-i8-nchw.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/fc4-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/fc5-i8.csv", WEIGHTS_I8_DIR)),
        )
    }
}

pub trait SepconvWeights<T>
where
    T: Coeff,
{
    fn into_tuple(self) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>);
}

impl<T> SepconvWeights<T> for Weights<T>
where
    T: Coeff,
{
    fn into_tuple(self) -> (Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>, Vec<T>) {
        (self.0, self.1, self.2, self.3, self.4, self.5, self.6)
    }
}
