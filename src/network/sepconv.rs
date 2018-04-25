use super::*;
use ocl::SpatialDims;
use geometry::*;
use ndarray::{Array, ShapeBuilder, arr2};

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
    mp1_block_dim: 32,
    mp2_block_dim: 16,
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
    pub mp1_block_dim: usize,
    pub mp2_block_dim: usize,
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
    pub fn create_layers(
        p: &SepconvHyperParams,
    ) -> (
        VConvLayer<T>,
        HConvLayer<T>,
        MaxpoolLayer,
        VConvLayer<T>,
        HConvLayer<T>,
        MaxpoolLayer,
        DenseLayer<T>,
        DenseLayer<T>,
        DenseLayer<T>,
    ) {
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

        // Init OpenCL
        let (queue, program, _context) = cl::init(
            &["sepconv.cl", "max_pool.cl", "mtx_mul.cl"],
            &[
                ("WIDTH", p.side as i32),
                ("HEIGHT", p.side as i32),
                ("MP1_BLOCK_DIM", p.mp1_block_dim as i32),
                ("MP2_BLOCK_DIM", p.mp2_block_dim as i32),
                ("ROWS_BLOCKDIM_Y", p.hconv1_blockdim_y as i32),
                ("ROWS_2_BLOCKDIM_Y", p.hconv2_blockdim_y as i32),
                ("INJECT_RELU_AFTER_MXP", 1 as i32),
            ],
        ).expect(COMPILE_ERR_MSG);

        let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, dense3, dense4, dense5) =
            SepconvNetwork::create_layers(&p);

        // Allocate read-only memory on-device for the weights buffers
        let v1_wgts_buf = vconv1.create_wgts_buf(&queue);
        let h1_wgts_buf = hconv1.create_wgts_buf(&queue);
        let v2_wgts_buf = vconv2.create_wgts_buf(&queue);
        let h2_wgts_buf = hconv2.create_wgts_buf(&queue);
        let d3_wgts_buf = dense3.create_wgts_buf(&queue);

        // Allocate memory on-device for the I/O buffers
        let intermediary_flags = flags::MEM_READ_WRITE;
        let in_buf = vconv1
            .create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();
        let conv1_mid_buf = vconv1.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv1_out_buf = hconv1.create_out_buf(intermediary_flags, &queue).unwrap();
        let mxp1_out_buf: Buffer<T> = mxp1.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv2_mid_buf = vconv2.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv2_out_buf = hconv2.create_out_buf(intermediary_flags, &queue).unwrap();
        // HACK: needs to be ALLOC_HOST_PTR to allow for rearranging the weights on the CPU
        let mxp2_out_buf: Buffer<T> =
            mxp2.create_out_buf(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR, &queue)
                .unwrap();
        let dense3_out_buf = dense3
            .create_out_buf(flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();

        // Write buffers to device
        v1_wgts_buf.write(vconv1.weights()).enq().unwrap();
        h1_wgts_buf.write(hconv1.weights()).enq().unwrap();
        v2_wgts_buf.write(vconv2.weights()).enq().unwrap();
        h2_wgts_buf.write(hconv2.weights()).enq().unwrap();
        d3_wgts_buf.write(dense3.weights()).enq().unwrap();

        // Create kernels
        let b = ClKernelBuilder::new(&program, queue.clone());
        let krn_vconv1 = b.build_iow_kernel(
            "col_conv",
            vconv1.gws_hint(),
            SpatialDims::Three(p.vconv1_blockdim_x, p.vconv1_blockdim_y, 1),
            &in_buf,        // In
            &conv1_mid_buf, // Out
            &v1_wgts_buf,   // Weights
        );
        let krn_hconv1 = b.build_iow_kernel(
            "row_conv",
            hconv1.gws_hint(),
            SpatialDims::Three(p.side, p.hconv1_blockdim_y, 1),
            &conv1_mid_buf, // In
            &conv1_out_buf, // Out
            &h1_wgts_buf,   // Weights
        );
        let krn_max_pool1 = b.build_io_kernel(
            "max_pool_1",
            mxp1.gws_hint(),
            SpatialDims::Three(p.mp1_block_dim, p.mp1_block_dim, 1),
            &conv1_out_buf, // In
            &mxp1_out_buf,  // Out
        );
        let krn_vconv2 = b.build_iow_kernel(
            "col_conv_2",
            vconv2.gws_hint(),
            SpatialDims::Three(p.vconv2_blockdim_x, p.vconv1_blockdim_y, 1),
            &mxp1_out_buf,  // In
            &conv2_mid_buf, // Out
            &v2_wgts_buf,   // Weights
        );
        let krn_hconv2 = b.build_iow_kernel(
            "row_conv_2",
            hconv2.gws_hint(),
            SpatialDims::Three(p.side / 2, p.hconv2_blockdim_y, 1),
            &conv2_mid_buf, // In
            &conv2_out_buf, // Out
            &h2_wgts_buf,   // Weights
        );
        let krn_max_pool2 = b.build_io_kernel(
            "max_pool_2",
            mxp2.gws_hint(),
            SpatialDims::Three(p.mp2_block_dim, p.mp2_block_dim, 1),
            &conv2_out_buf, // In
            &mxp2_out_buf,  // Out
        );
        let krn_dense3 = Kernel::builder()
            .name("mtx_mul_f32")
            .program(&program)
            .queue(queue.clone())
            .global_work_size(dense3.gws_hint())
            .arg(&mxp2_out_buf)
            .arg(&dense3_out_buf)
            .arg(&d3_wgts_buf)
            .build()
            .unwrap();

        // Wait until all commands have finished running before returning.
        queue.finish().unwrap();

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
        // HACK: Max-work-size is not enough, use halved dimensions for hconv1 and/or mxp 1
        let max_wgs = cl::max_wgs(None);
        if p.side * p.hconv1_blockdim_y > max_wgs {
            p.hconv1_blockdim_y /= 2;
            warn!("using halved dimension for horizontal convolution 1");
        }
        if p.mp1_block_dim * p.mp1_block_dim > max_wgs {
            p.mp1_block_dim /= 2;
            warn!("using halved dimension for maxpool 1");
        }
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
