use super::*;
use ocl::SpatialDims;
use geometry::PaddedSquare;

const SEPCONV_HYPER_PARAMS: SepconvHyperParams = SepconvHyperParams {
    width: 96,
    height: 96,
    rows_blockdim_x: 96,
    rows_blockdim_y: 4,
    rows2_blockdim_x: 48,
    rows2_blockdim_y: 4,
    columns_blockdim_x: 32,
    columns_blockdim_y: 8,
    columns2_blockdim_x: 16,
    columns2_blockdim_y: 8,
    kernel_radius: 2,
    c1: 3,
    num_conv1_fms: 7,
    num_conv2_fms: 32,
    mp1_block_dim: 32,
    mp2_block_dim: 16,
};

#[derive(Clone, Debug)]
struct SepconvHyperParams {
    pub width: usize,
    pub height: usize,
    pub rows_blockdim_x: usize,
    pub rows_blockdim_y: usize,
    pub rows2_blockdim_x: usize,
    pub rows2_blockdim_y: usize,
    pub columns_blockdim_x: usize,
    pub columns_blockdim_y: usize,
    pub columns2_blockdim_x: usize,
    pub columns2_blockdim_y: usize,
    pub kernel_radius: usize,
    pub c1: usize,
    pub num_conv1_fms: usize,
    pub num_conv2_fms: usize,
    pub mp1_block_dim: usize,
    pub mp2_block_dim: usize,
}
pub const WEIGHTS_DIR: &'static str = "input/weights/sepconv-96-97";

pub struct SepconvNetwork<T>
where
    T: Coeff,
{
    pub in_buf: Buffer<T>,
    krn_c_conv1: Kernel,
    krn_r_conv1: Kernel,
    krn_max_pool1: Kernel,
    krn_c_conv2: Kernel,
    krn_r_conv2: Kernel,
    krn_max_pool2: Kernel,
    krn_dense3: Kernel,
    dense3_out_buf: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,

    // TEMP
    v1_out_buf: Buffer<T>,
    h1_out_buf: Buffer<T>,
    v2_out_buf: Buffer<T>,
    h2_out_buf: Buffer<T>,
    mxp1_buf: Buffer<T>,
    mxp2_buf: Buffer<T>,
}

impl<T> SepconvNetwork<T>
where
    T: CoeffFloat + ReadCsvFromFile,
{
    pub fn new(program: &Program, queue: &Queue) -> SepconvNetwork<T> {
        let p = SEPCONV_HYPER_PARAMS.clone();
        let kernel_length = 2 * p.kernel_radius + 1;
        let filter_len: usize = 1 * kernel_length;

        // Load the weights for the convolutional layers
        let v1 = T::read_csv_from_file(&format!("{}/vcr1-f32.csv", WEIGHTS_DIR));
        let h1 = T::read_csv_from_file(&format!("{}/hcr1-f32.csv", WEIGHTS_DIR));
        let v2 = T::read_csv_from_file(&format!("{}/vcr2-f32.csv", WEIGHTS_DIR));
        let h2 = T::read_csv_from_file(&format!("{}/hcr2-f32.csv", WEIGHTS_DIR));
        // Load the weights for the dense layers
        let dense3_wgts = T::read_csv_from_file(&format!("{}/fc3-f32.csv", WEIGHTS_DIR));
        let dense4_wgts = T::read_csv_from_file(&format!("{}/fc4-f32.csv", WEIGHTS_DIR));
        let dense5_wgts = T::read_csv_from_file(&format!("{}/fc5-f32.csv", WEIGHTS_DIR));

        // HACK: manually calculated size for convolution output
        //const PATCH3SQ: usize = ((96 / 2 / 2) * (96 / 2 / 2)) / 2;
        let dense3 = DenseLayer::new(
            // HACK: hardcoded from mxp2_buf size
            24 * 24 * 32, /*PATCH3SQ * CLASSIC_HYPER_PARAMS.num_feature_maps*/
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            dense3_wgts,
        );
        let dense4 = DenseLayer::new(
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            dense4_wgts,
        );
        let dense5 = DenseLayer::new(
            CLASSIC_HYPER_PARAMS.fully_connected_const,
            CLASSIC_HYPER_PARAMS.num_output_classes,
            dense5_wgts,
        );

        // Allocate read-only memory on-device for the weights buffers
        let v1_buf = cl::create_buffer::<T>(
            filter_len * p.c1 * p.num_conv1_fms,
            flags::MEM_READ_ONLY,
            &queue,
        ).unwrap();
        let h1_buf = cl::create_buffer::<T>(
            filter_len * p.num_conv1_fms * p.num_conv2_fms,
            flags::MEM_READ_ONLY,
            &queue,
        ).unwrap();
        let v2_buf = cl::create_buffer::<T>(
            filter_len * p.num_conv2_fms * p.num_conv1_fms,
            flags::MEM_READ_ONLY,
            &queue,
        ).unwrap();
        let h2_buf = cl::create_buffer::<T>(
            filter_len * p.num_conv1_fms * p.num_conv2_fms,
            flags::MEM_READ_ONLY,
            &queue,
        ).unwrap();
        let dense3_buf = dense3.create_wgts_buf(&queue).unwrap();

        // Allocate memory on-device for the I/O buffers
        let in_buf = cl::create_buffer(
            p.c1 * p.width * p.height,
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let v1_out_buf = cl::create_buffer::<T>(
            p.num_conv1_fms * p.width * p.height,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let h1_out_buf = cl::create_buffer::<T>(
            p.num_conv2_fms * p.width * p.height,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let v2_out_buf = cl::create_buffer::<T>(
            p.num_conv1_fms * p.width / 2 * p.height / 2,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let h2_out_buf = cl::create_buffer::<T>(
            32 * 48 * 48,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let mxp1_buf = cl::create_buffer::<T>(
            48 * 48 * 32,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let mxp2_buf = cl::create_buffer::<T>(
            24 * 24 * 32,
            flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();
        let dense3_out_buf = cl::create_buffer::<T>(
            dense3.num_out(),
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        ).unwrap();

        // Write buffers to device
        v1_buf.write(&v1).enq().unwrap();
        h1_buf.write(&h1).enq().unwrap();
        v2_buf.write(&v2).enq().unwrap();
        h2_buf.write(&h2).enq().unwrap();
        dense3_buf.write(dense3.weights()).enq().unwrap();
        // HACK:
        queue.finish().unwrap();

        // Create kernels
        let krn_c_conv1 = Kernel::new("colConv", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width, p.height, p.num_conv1_fms))
            .lws(SpatialDims::Three(
                p.columns_blockdim_x,
                p.columns_blockdim_y,
                1,
            ))
            .arg_buf(&in_buf)
            .arg_buf(&v1_out_buf)
            .arg_buf(&v1_buf);
        let krn_r_conv1 = Kernel::new("rowConv", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width, p.height, p.num_conv2_fms))
            // HACK: my desktop GPU cannot handle the full dimension (p.rows_blockdim_x*p.rows_blockdim_y*1 = 384) use 256 instead
            // TODO: benchmark different combinations: 32x4x2 -> 32x8x1
            .lws(SpatialDims::Three(
                p.num_conv2_fms,
                p.rows_blockdim_y,
                2,
            ))
            .arg_buf(&v1_out_buf)
            .arg_buf(&h1_out_buf)
            .arg_buf(&h1_buf);
        let krn_max_pool1 = Kernel::new("MaxPool1", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width, p.height, 32))
            // TODO: my desktop GPU cannot handle the full dimension (p.mp1_block_dim*p.mp1_block_dim*1 = 384), find a way to use 256 instead (without segfaults :P)
            .lws(SpatialDims::Three(p.mp1_block_dim,p.mp1_block_dim,1))
            .arg_buf(&h1_out_buf)
            .arg_buf(&mxp1_buf);
        let krn_c_conv2 = Kernel::new("colConv2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width / 2, p.height / 2, 7))
            .lws(SpatialDims::Three(
                p.columns2_blockdim_x,
                p.columns2_blockdim_y,
                1,
            ))
            .arg_buf(&mxp1_buf)
            .arg_buf(&v2_out_buf)
            .arg_buf(&v2_buf);
        let krn_r_conv2 = Kernel::new("rowConv2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width / 2, p.height / 2, 32))
            .lws(SpatialDims::Three(
                p.rows2_blockdim_x,
                p.rows2_blockdim_y,
                1,
            ))
            .arg_buf(&v2_out_buf)
            .arg_buf(&h2_out_buf)
            .arg_buf(&h2_buf);
        let krn_max_pool2 = Kernel::new("MaxPool2", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(p.width / 2, p.height / 2, 32))
            .lws(SpatialDims::Three(p.mp2_block_dim, p.mp2_block_dim, 1))
            .arg_buf(&h2_out_buf)
            .arg_buf(&mxp2_buf);
        let krn_dense3 = Kernel::new("mtx_mulf", &program)
            .unwrap()
            .queue(queue.clone())
            .gws(dense3.gws_hint())
            .arg_buf(&mxp2_buf)
            .arg_buf(&dense3_out_buf)
            .arg_buf(&dense3_buf);

        SepconvNetwork {
            in_buf: in_buf,
            krn_c_conv1,
            krn_r_conv1,
            krn_max_pool1,
            krn_c_conv2,
            krn_r_conv2,
            krn_max_pool2,
            krn_dense3,
            dense3_out_buf,
            dense4,
            dense5,
            v1_out_buf,
            h1_out_buf,
            v2_out_buf,
            h2_out_buf,
            mxp1_buf,
            mxp2_buf,
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
            T::write_lines_into_file("output/sepconv/in.f", input_data);

            cl::map_to_buf(&self.in_buf, &input_data).unwrap();

            // HACK: finish after every phase to make sure that everything works; remove after implementing accuracy test
            self.krn_c_conv1.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.v1_out_buf).unwrap();
            T::write_lines_into_file("output/sepconv/v1out-f32.f", &b);

            self.krn_r_conv1.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.h1_out_buf).unwrap();
            T::write_lines_into_file("output/sepconv/h1out-f32.f", &b);

            self.krn_max_pool1.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.mxp1_buf).unwrap();
            T::write_lines_into_file("output/sepconv/mxp1-f32.f", &b);

            self.krn_c_conv2.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.v2_out_buf).unwrap();
            T::write_lines_into_file("output/sepconv/v2out-f32.f", &b);

            self.krn_r_conv2.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.h2_out_buf).unwrap();
            T::write_lines_into_file("output/sepconv/h2out-f32.f", &b);

            self.krn_max_pool2.cmd().queue(&queue).enq().unwrap();
            queue.finish().unwrap();
            let b = cl::read_buf(&self.mxp2_buf).unwrap();
            T::write_lines_into_file("output/sepconv/mxp2-f32.f", &b);
            self.krn_dense3.cmd().queue(&queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = relu(&self.dense4.mtx_mul(&dense3_out));

        // Run the 5th layer (fully-connected)
        softmax(&self.dense5.mtx_mul(&dense4_out))
    }
}
