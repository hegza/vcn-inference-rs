use super::*;
use ocl::SpatialDims;
use geometry::*;
use ndarray::Array;
use ndarray::ShapeBuilder; // Needed for .strides() method
use ndarray::arr2;

const SEPCONV_HYPER_PARAMS: SepconvHyperParams = SepconvHyperParams {
    side: 96,
    rows_blockdim_y: 4,
    columns_blockdim_x: 32,
    columns_blockdim_y: 8,
    columns2_blockdim_x: 16,
    kernel_len: 5,
    num_channels: 3,
    kernel_split: 7,
    num_conv_fms: 32,
    // Note: this is not the number of feature maps but something based on image side probably (32x3 = 96)
    mp1_block_dim: 32,
    mp2_block_dim: 16,
};

#[derive(Clone, Debug)]
struct SepconvHyperParams {
    pub side: usize,
    pub rows_blockdim_y: usize,
    pub columns_blockdim_x: usize,
    pub columns_blockdim_y: usize,
    pub columns2_blockdim_x: usize,
    pub kernel_len: usize,
    pub num_channels: usize,
    pub kernel_split: usize,
    pub num_conv_fms: usize,
    pub mp1_block_dim: usize,
    pub mp2_block_dim: usize,
}
pub const WEIGHTS_DIR: &str = "input/weights/sepconv-96-97";

pub struct SepconvNetwork<T>
where
    T: Coeff,
{
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
    pub fn new(program: &Program, queue: &Queue) -> SepconvNetwork<T> {
        let p = SEPCONV_HYPER_PARAMS.clone();

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
        let vconv1 = VConvLayer::new(p.kernel_len, in_shape, p.kernel_split, v1_wgts);
        let hconv1 = HConvLayer::new(
            p.kernel_len,
            *vconv1.output_shape(),
            p.num_conv_fms,
            h1_wgts,
        );
        let mxp1 = MaxpoolLayer::new(*hconv1.output_shape(), 2);
        let vconv2 = VConvLayer::new(p.kernel_len, *mxp1.output_shape(), p.kernel_split, v2_wgts);
        let hconv2 = HConvLayer::new(
            p.kernel_len,
            *vconv2.output_shape(),
            p.num_conv_fms,
            h2_wgts,
        );
        let mxp2 = MaxpoolLayer::new(*hconv2.output_shape(), 2);

        let dense3 = DenseLayer::new(
            mxp2.num_out(),
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            dense3_wgts,
        );
        let dense4 = DenseLayer::new(
            dense3.num_out(),
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            dense4_wgts,
        );
        let dense5 = DenseLayer::new(
            dense4.num_out(),
            CLASSIC_HYPER_PARAMS.num_output_classes,
            dense5_wgts,
        );

        // Allocate read-only memory on-device for the weights buffers
        let v1_wgts_buf = vconv1.create_wgts_buf(queue).unwrap();
        let h1_wgts_buf = hconv1.create_wgts_buf(queue).unwrap();
        let v2_wgts_buf = vconv2.create_wgts_buf(queue).unwrap();
        let h2_wgts_buf = hconv2.create_wgts_buf(queue).unwrap();
        let d3_wgts_buf = dense3.create_wgts_buf(queue).unwrap();

        // Allocate memory on-device for the I/O buffers
        let intermediary_flags = flags::MEM_READ_WRITE;
        let in_buf = vconv1
            .create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, queue)
            .unwrap();
        let conv1_mid_buf = vconv1.create_out_buf(intermediary_flags, queue).unwrap();
        let conv1_out_buf = hconv1.create_out_buf(intermediary_flags, queue).unwrap();
        let mxp1_out_buf: Buffer<T> = mxp1.create_out_buf(intermediary_flags, queue).unwrap();
        let conv2_mid_buf = vconv2.create_out_buf(intermediary_flags, queue).unwrap();
        let conv2_out_buf = hconv2.create_out_buf(intermediary_flags, queue).unwrap();
        // HACK: needs to be ALLOC_HOST_PTR to allow for rearranging the weights on the CPU
        let mxp2_out_buf: Buffer<T> =
            mxp2.create_out_buf(flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR, queue)
                .unwrap();
        let dense3_out_buf = cl::create_buffer::<T>(
            dense3.num_out(),
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            queue,
        ).unwrap();

        // Write buffers to device
        v1_wgts_buf.write(vconv1.weights()).enq().unwrap();
        h1_wgts_buf.write(hconv1.weights()).enq().unwrap();
        v2_wgts_buf.write(vconv2.weights()).enq().unwrap();
        h2_wgts_buf.write(hconv2.weights()).enq().unwrap();
        d3_wgts_buf.write(dense3.weights()).enq().unwrap();

        // Create kernels
        let b = ClKernelBuilder::new(program, queue.clone());
        let krn_vconv1 = b.build_iow_kernel(
            "colConv",
            vconv1.gws_hint(),
            SpatialDims::Three(p.columns_blockdim_x, p.columns_blockdim_y, 1),
            &in_buf,        // In
            &conv1_mid_buf, // Out
            &v1_wgts_buf,   // Weights
        );
        let krn_hconv1 = b.build_iow_kernel(
            "rowConv",
            hconv1.gws_hint(),
            SpatialDims::Three(p.side, p.rows_blockdim_y, 1),
            &conv1_mid_buf, // In
            &conv1_out_buf, // Out
            &h1_wgts_buf,   // Weights
        );
        let krn_max_pool1 = b.build_io_kernel(
            "MaxPool1",
            mxp1.gws_hint(),
            // TODO: my desktop GPU cannot handle the full dimension (p.mp1_block_dim*p.mp1_block_dim*1 = 384), find a way to use 256 instead (without segfaults :P)
            SpatialDims::Three(p.mp1_block_dim, p.mp1_block_dim, 1),
            &conv1_out_buf, // In
            &mxp1_out_buf,  // Out
        );
        let krn_vconv2 = b.build_iow_kernel(
            "colConv2",
            vconv2.gws_hint(),
            SpatialDims::Three(p.columns2_blockdim_x, p.columns_blockdim_y, 1),
            &mxp1_out_buf,  // In
            &conv2_mid_buf, // Out
            &v2_wgts_buf,   // Weights
        );
        let krn_hconv2 = b.build_iow_kernel(
            "rowConv2",
            hconv2.gws_hint(),
            SpatialDims::Three(p.side / 2, p.rows_blockdim_y, 1),
            &conv2_mid_buf, // In
            &conv2_out_buf, // Out
            &h2_wgts_buf,   // Weights
        );
        let krn_max_pool2 = b.build_io_kernel(
            "MaxPool2",
            mxp2.gws_hint(),
            SpatialDims::Three(p.mp2_block_dim, p.mp2_block_dim, 1),
            &conv2_out_buf, // In
            &mxp2_out_buf,  // Out
        );
        let krn_dense3 = Kernel::builder()
            .name("mtx_mulf")
            .program(program)
            .queue(queue.clone())
            .global_work_size(dense3.gws_hint())
            .arg(&mxp2_out_buf)
            .arg(&dense3_out_buf)
            .arg(&d3_wgts_buf)
            .build()
            .unwrap();

        SepconvNetwork {
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
}

impl<T> Predict<T> for SepconvNetwork<T>
where
    T: CoeffFloat + WriteLinesIntoFile,
{
    // Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[T], queue: &Queue) -> Vec<f32> {
        /*let buf = */
        unsafe {
            cl::map_to_buf(&self.in_buf, input_data).unwrap();

            self.krn_vconv1.cmd().queue(queue).enq().unwrap();
            self.krn_hconv1.cmd().queue(queue).enq().unwrap();
            self.krn_max_pool1.cmd().queue(queue).enq().unwrap();
            self.krn_vconv2.cmd().queue(queue).enq().unwrap();
            self.krn_hconv2.cmd().queue(queue).enq().unwrap();
            self.krn_max_pool2.cmd().queue(queue).enq().unwrap();
            self.krn_dense3.cmd().queue(queue).enq().unwrap();
        }

        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        // Load the 3rd layer from the GPU and Run ReLU on it
        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(&self.dense4.mtx_mul(&dense3_out));

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.mtx_mul(&dense4_out);

        softmax(&dense5_out)
    }
}
