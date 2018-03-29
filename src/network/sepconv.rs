use super::*;
use ocl::SpatialDims;
use geometry::*;

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
pub const WEIGHTS_DIR: &'static str = "input/weights/sepconv-96-97";

pub struct SepconvNetwork<T>
where
    T: Coeff,
{
    pub in_buf: Buffer<T>,
    krn_v_conv1: Kernel,
    krn_h_conv1: Kernel,
    krn_max_pool1: Kernel,
    krn_v_conv2: Kernel,
    krn_h_conv2: Kernel,
    krn_max_pool2: Kernel,
    krn_dense3: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,

    // TEMP
    conv1_mid_buf: Buffer<T>,
    conv1_out_buf: Buffer<T>,
    conv2_mid_buf: Buffer<T>,
    conv2_out_buf: Buffer<T>,
    mxp1_out_buf: Buffer<T>,
    mxp2_out_buf: Buffer<T>,
    write_debug: bool,
}

impl<T> SepconvNetwork<T>
where
    T: CoeffFloat + ReadCsvFromFile,
{
    pub fn new(program: &Program, queue: &Queue, write_debug: bool) -> SepconvNetwork<T> {
        let p = SEPCONV_HYPER_PARAMS.clone();

        // Load the weights for the convolutional layers
        let v1_wgts = T::read_csv_from_file(&format!("{}/vcr1-f32.csv", WEIGHTS_DIR));
        let h1_wgts = T::read_csv_from_file(&format!("{}/hcr1-f32.csv", WEIGHTS_DIR));
        let v2_wgts = T::read_csv_from_file(&format!("{}/vcr2-f32.csv", WEIGHTS_DIR));
        let h2_wgts = T::read_csv_from_file(&format!("{}/hcr2-f32.csv", WEIGHTS_DIR));
        // Load the weights for the dense layers
        let dense3_wgts = T::read_csv_from_file(&format!("{}/fc3-f32.csv", WEIGHTS_DIR));
        let dense4_wgts = T::read_csv_from_file(&format!("{}/fc4-f32.csv", WEIGHTS_DIR));
        let dense5_wgts = T::read_csv_from_file(&format!("{}/fc5-f32.csv", WEIGHTS_DIR));

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
        let v1_wgts_buf = vconv1.create_wgts_buf(&queue).unwrap();
        let h1_wgts_buf = hconv1.create_wgts_buf(&queue).unwrap();
        let v2_wgts_buf = vconv2.create_wgts_buf(&queue).unwrap();
        let h2_wgts_buf = hconv2.create_wgts_buf(&queue).unwrap();
        let d3_wgts_buf = dense3.create_wgts_buf(&queue).unwrap();

        // Allocate memory on-device for the I/O buffers
        let intermediary_flags = match write_debug {
            false => flags::MEM_READ_WRITE,
            true => flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
        };
        let in_buf = vconv1
            .create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();
        let conv1_mid_buf = vconv1.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv1_out_buf = hconv1.create_out_buf(intermediary_flags, &queue).unwrap();
        let mxp1_out_buf = mxp1.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv2_mid_buf = vconv2.create_out_buf(intermediary_flags, &queue).unwrap();
        let conv2_out_buf = hconv2.create_out_buf(intermediary_flags, &queue).unwrap();
        let mxp2_out_buf = mxp2.create_out_buf(intermediary_flags, &queue).unwrap();
        let dense3_out_buf = cl::create_buffer::<T>(
            dense3.num_out(),
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();

        // Write buffers to device
        v1_wgts_buf.write(vconv1.weights()).enq().unwrap();
        h1_wgts_buf.write(hconv1.weights()).enq().unwrap();
        v2_wgts_buf.write(vconv2.weights()).enq().unwrap();
        h2_wgts_buf.write(hconv2.weights()).enq().unwrap();
        d3_wgts_buf.write(dense3.weights()).enq().unwrap();
        // HACK:
        queue.finish().unwrap();

        // Create kernels
        let krn_v_conv1 = Kernel::new("colConv", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(vconv1.gws_hint())
            .lws(SpatialDims::Three(
                p.columns_blockdim_x,
                p.columns_blockdim_y,
                1,
            ))
            .arg_buf(&in_buf)
            .arg_buf(&conv1_mid_buf)
            .arg_buf(&v1_wgts_buf);
        let krn_h_conv1 = Kernel::new("rowConv", &program).unwrap()
            .queue(queue.clone())
            .gws(hconv1.gws_hint())
            // NOTE: my desktop GPU cannot handle the full dimension (p.side*p.rows_blockdim_y*1 = 384)
            .lws(SpatialDims::Three(p.side, p.rows_blockdim_y, 1))
            .arg_buf(&conv1_mid_buf)
            .arg_buf(&conv1_out_buf)
            .arg_buf(&h1_wgts_buf);
        let krn_max_pool1 = Kernel::new("MaxPool1", &program).unwrap()
            .queue(queue.clone())
            .gws(mxp1.gws_hint())
            // TODO: my desktop GPU cannot handle the full dimension (p.mp1_block_dim*p.mp1_block_dim*1 = 384), find a way to use 256 instead (without segfaults :P)
            .lws(SpatialDims::Three(p.mp1_block_dim, p.mp1_block_dim, 1))
            .arg_buf(&conv1_out_buf)
            .arg_buf(&mxp1_out_buf);
        let krn_v_conv2 = Kernel::new("colConv2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(vconv2.gws_hint())
            .lws(SpatialDims::Three(
                p.columns2_blockdim_x,
                p.columns_blockdim_y,
                1,
            ))
            .arg_buf(&mxp1_out_buf)
            .arg_buf(&conv2_mid_buf)
            .arg_buf(&v2_wgts_buf);
        let krn_h_conv2 = Kernel::new("rowConv2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(hconv2.gws_hint())
            .lws(SpatialDims::Three(p.side / 2, p.rows_blockdim_y, 1))
            .arg_buf(&conv2_mid_buf)
            .arg_buf(&conv2_out_buf)
            .arg_buf(&h2_wgts_buf);
        let krn_max_pool2 = Kernel::new("MaxPool2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(mxp2.gws_hint())
            .lws(SpatialDims::Three(p.mp2_block_dim, p.mp2_block_dim, 1))
            .arg_buf(&conv2_out_buf)
            .arg_buf(&mxp2_out_buf);
        let krn_dense3 = Kernel::new("mtx_mulf", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(dense3.gws_hint())
            .arg_buf(&mxp2_out_buf)
            .arg_buf(&dense3_out_buf)
            .arg_buf(&d3_wgts_buf);

        SepconvNetwork {
            in_buf: in_buf,
            krn_v_conv1,
            krn_h_conv1,
            krn_max_pool1,
            krn_v_conv2,
            krn_h_conv2,
            krn_max_pool2,
            krn_dense3,
            dense3_out_buf,
            dense4,
            dense5,
            conv1_mid_buf,
            conv1_out_buf,
            conv2_mid_buf,
            conv2_out_buf,
            mxp1_out_buf,
            mxp2_out_buf,
            write_debug,
        }
    }
}

impl<T> Predict<T> for SepconvNetwork<T>
where
    T: CoeffFloat + WriteLinesIntoFile,
{
    // Maps the input buffer, and runs the network, returning the result.
    fn predict(&self, input_data: &[T], queue: &Queue) -> Vec<f32> {
        unsafe {
            if self.write_debug {
                T::write_lines_into_file("output/sepconv/in.f", input_data);
            }

            cl::map_to_buf(&self.in_buf, &input_data).unwrap();

            // HACK: finish after every phase to make sure that everything works; remove after implementing accuracy test
            self.krn_v_conv1.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.conv1_mid_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/vcr1-out.f", &b);
            }

            self.krn_h_conv1.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.conv1_out_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/hcr1-out.f", &b);
            }
            self.krn_max_pool1.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.mxp1_out_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/mxp1-out.f", &b);
            }
            self.krn_v_conv2.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.conv2_mid_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/vcr2-out.f", &b);
            }

            self.krn_h_conv2.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.conv2_out_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/hcr2-out.f", &b);
            }

            self.krn_max_pool2.cmd().queue(&queue).enq().unwrap();
            if self.write_debug {
                queue.finish().unwrap();
                let b = cl::read_buf(&self.mxp2_out_buf).unwrap();
                T::write_lines_into_file("output/sepconv/f32/mxp2-out.f", &b);
            }

            // TODO: move the reordering to the kernel or get new ..weights? from Mir
            // Load the buffer from GPU from max-pool output
            // (fms, y, x); strides == (24*24, 24, 1) -> (24, 1, 24*24)
            // TODO: flatten/reorder the GPU buffer xyz -> zxy
            /*
            let flattened = Array::from_shape_vec(
                (32, 24, 24).strides((24, 1, 24 * 24)),
                cl::read_buf(&self.mxp2_out_buf).unwrap(),
            ).unwrap()
                .into_raw_vec();

            // Re-upload the buffer back to the GPU for use in dense 3
            self.mxp2_out_buf.write(&flattened).enq().unwrap();
            */

            self.krn_dense3.cmd().queue(&queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });
        T::write_lines_into_file("output/sepconv/f32/fc3-out.f", &dense3_out);

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(&self.dense4.mtx_mul(&dense3_out));
        T::write_lines_into_file("output/sepconv/f32/fc4-out.f", &dense4_out);

        // Run the 5th layer (fully-connected)
        let dense5_out = self.dense5.mtx_mul(&dense4_out);
        T::write_lines_into_file("output/sepconv/f32/fc5-out.f", &dense5_out);
        let result = softmax(&dense5_out);

        f32::write_lines_into_file("output/sepconv/f32/out.f", &result);
        result
    }
}
