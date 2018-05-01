use super::*;
use ocl::{Device, Platform, SpatialDims};
use geometry::*;
use ndarray::{Array, ShapeBuilder, arr2};
use std::ops::Deref;

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
pub const WEIGHTS_DIR: &str = "input/weights/sepconv-96-97";

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
    T: CoeffFloat + ReadCsv,
{
    pub fn create_layers(p: &SepconvHyperParams) -> Layers<T> {
        // Load the weights for the convolutional layers
        let v1_wgts = T::read_csv(&format!("{}/vcr1-f32.csv", WEIGHTS_DIR));
        let h1_wgts = T::read_csv(&format!("{}/hcr1-f32.csv", WEIGHTS_DIR));
        let v2_wgts = T::read_csv(&format!("{}/vcr2-f32.csv", WEIGHTS_DIR));
        let h2_wgts = T::read_csv(&format!("{}/hcr2-f32.csv", WEIGHTS_DIR));
        // Load the weights for the dense layers
        let dense3_wgts = T::read_csv(&format!("{}/fc3-f32-nchw.csv", WEIGHTS_DIR));
        let dense4_wgts = T::read_csv(&format!("{}/fc4-f32.csv", WEIGHTS_DIR));
        let dense5_wgts = T::read_csv(&format!("{}/fc5-f32.csv", WEIGHTS_DIR));

        let in_shape = ImageGeometry::new(p.side, p.num_channels);
        let vconv1 = VConvLayer::new(p.kernel_len, in_shape, p.conv_kernel_split, v1_wgts);
        let hconv1 = HConvLayer::new(
            p.kernel_len,
            *vconv1.output_shape(),
            p.num_conv_fms,
            h1_wgts,
        );
        let mxp1 = MaxpoolLayer::new(*hconv1.output_shape(), 2);
        let vconv2 = VConvLayer::new(
            p.kernel_len,
            *mxp1.output_shape(),
            p.conv_kernel_split,
            v2_wgts,
        );
        let hconv2 = HConvLayer::new(
            p.kernel_len,
            *vconv2.output_shape(),
            p.num_conv_fms,
            h2_wgts,
        );
        let mxp2 = MaxpoolLayer::new(*hconv2.output_shape(), 2);

        let dense3 = DenseLayer::new(mxp2.num_out(), p.fully_connected_const, dense3_wgts);
        let dense4 = DenseLayer::new(dense3.num_out(), p.fully_connected_const, dense4_wgts);
        let dense5 = DenseLayer::new(dense4.num_out(), p.num_output_classes, dense5_wgts);

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
    pub fn new() -> SepconvNetwork<T> {
        let mut p = SEPCONV_HYPER_PARAMS.clone();

        // HACK: Reduce dimensions of overshot layers
        SepconvNetwork::<T>::fix_params_for_default_gpu(&mut p);

        let layers: Layers<T> = SepconvNetwork::create_layers(&p);

        // Init OpenCL
        let (queue, program, _context) = cl::init(
            &["sepconv.cl", "max_pool.cl", "mtx_mul.cl"],
            &SepconvNetwork::<T>::compile_flags(&p, &layers),
        ).expect(COMPILE_ERR_MSG);

        // Create shorthands (and move)
        let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, dense3, dense4, dense5) = layers;

        // Allocate read-only memory on-device for the weights buffers
        let v1_wgts_buf = vconv1.create_wgts_buf(&queue);
        let h1_wgts_buf = hconv1.create_wgts_buf(&queue);
        let v2_wgts_buf = vconv2.create_wgts_buf(&queue);
        let h2_wgts_buf = hconv2.create_wgts_buf(&queue);
        let d3_wgts_buf = dense3.create_wgts_buf(&queue);

        // Allocate memory on-device for the I/O buffers
        let mut bufs = create_buffer_chain(
            &[
                &vconv1.0, &hconv1.0, &mxp1, &vconv2.0, &hconv2.0, &mxp2, &dense3
            ],
            &queue,
        );

        // Create kernels
        let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
        let dev_max_wgs = cl::max_wgs(Some(&primary_device));
        let b = ClKernelBuilder::new(&program, queue.clone());
        let krn_vconv1 = b.build_iow_kernel(
            "col_conv",
            vconv1.gws_hint(),
            SpatialDims::Two(p.vconv1_blockdim_x, p.vconv1_blockdim_y),
            &bufs[0],     // In
            &bufs[1],     // Out
            &v1_wgts_buf, // Weights
        );
        let krn_hconv1 = b.build_iow_kernel(
            "row_conv",
            hconv1.gws_hint(),
            SpatialDims::Two(p.side, p.hconv1_blockdim_y),
            &bufs[1],     // In
            &bufs[2],     // Out
            &h1_wgts_buf, // Weights
        );
        let krn_max_pool1 = b.build_io_kernel(
            "max_pool_1",
            mxp1.gws_hint(),
            mxp1.lws_hint(dev_max_wgs),
            &bufs[2], // In
            &bufs[3], // Out
        );
        let krn_vconv2 = b.build_iow_kernel(
            "col_conv_2",
            vconv2.gws_hint(),
            SpatialDims::Two(p.vconv2_blockdim_x, p.vconv1_blockdim_y),
            &bufs[3],     // In
            &bufs[4],     // Out
            &v2_wgts_buf, // Weights
        );
        let krn_hconv2 = b.build_iow_kernel(
            "row_conv_2",
            hconv2.gws_hint(),
            SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y),
            &bufs[4],     // In
            &bufs[5],     // Out
            &h2_wgts_buf, // Weights
        );
        let krn_max_pool2 = b.build_io_kernel(
            "max_pool_2",
            mxp2.gws_hint(),
            mxp2.lws_hint(dev_max_wgs),
            &bufs[5], // In
            &bufs[6], // Out
        );
        let krn_dense3 = Kernel::builder()
            .name("mtx_mul_f32")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(dense3.gws_hint())
            .arg(&bufs[6])     // In
            .arg(&bufs[7])     // Out
            .arg(&d3_wgts_buf) // Weights
            .build()
            .unwrap();

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

        // Move and store the first and last buffer
        let mut buf_drain = bufs.drain(..);
        let in_buf = buf_drain.next().unwrap();
        let dense3_out_buf = buf_drain.next_back().unwrap();

        SepconvNetwork {
            queue,
            in_buf,
            krn_vconv1,
            krn_hconv1,
            krn_max_pool1,
            krn_vconv2,
            krn_hconv2,
            krn_max_pool2,
            krn_dense3,
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
    pub fn compile_flags(p: &SepconvHyperParams, layers: &Layers<T>) -> Vec<(&'static str, i32)> {
        let max_wgs = cl::max_wgs(Some(&PRIMARY_DEVICE));
        let mxp1_lws = layers.mxp1().lws_hint(max_wgs).to_lens().unwrap()[0];
        let mxp2_lws = layers.mxp2().lws_hint(max_wgs).to_lens().unwrap()[0];

        vec![
            ("WIDTH", p.side as i32),
            ("HEIGHT", p.side as i32),
            ("MP1_BLOCK_DIM", mxp1_lws as i32),
            ("MP2_BLOCK_DIM", mxp2_lws as i32),
            ("ROWS_BLOCKDIM_Y", p.hconv1_blockdim_y as i32),
            ("ROWS_2_BLOCKDIM_Y", p.hconv2_blockdim_y as i32),
            ("INJECT_RELU_AFTER_MXP", 1 as i32),
        ]
    }
}

impl<T> Predict<T> for SepconvNetwork<T>
where
    T: CoeffFloat + WriteLinesIntoFile,
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
        let dense4_out = relu(&self.dense4.mtx_mul(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.mtx_mul(&dense4_out);

        softmax(&dense5_out)
    }
}

const COMPILE_ERR_MSG: &str = "unable to compile program. It's possible that not all hyper parameters were passed in as compiler definitions. See OpenCL error-message for more info.";

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
