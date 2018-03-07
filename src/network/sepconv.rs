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
    kernel_length: 5,
    c1: 3,
    c2: 7,
    c3: 32,
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
    pub kernel_length: usize,
    pub c1: usize,
    pub c2: usize,
    pub c3: usize,
    pub mp1_block_dim: usize,
    pub mp2_block_dim: usize,
}

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
    dev_mxp_h2: Buffer<T>,
    dense4: DenseLayer<T>,
    dense5: DenseLayer<T>,
}

impl<T> SepconvNetwork<T>
where
    T: CoeffFloat,
{
    pub fn new(program: &Program, queue: &Queue) -> SepconvNetwork<T> {
        // Create meta
        let params = SEPCONV_HYPER_PARAMS.clone();

        // HACK: these weights are trained for the original implementation and should be replaced
        let dense4_wgts = T::read_bin_from_file("input/weights/fc4-f32-le.bin");
        let dense5_wgts = T::read_bin_from_file("input/weights/fc5-f32-le.bin");
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

        // Create device buffers
        let dev_input = cl::create_buffer(
            params.c1 * params.width * params.height,
            flags::MEM_READ_ONLY,
            &queue,
        ).unwrap();
        let dev_out_v1 = cl::create_buffer::<T>(
            params.c2 * params.width * params.height,
            flags::MEM_READ_WRITE,
            &queue,
        ).unwrap();
        let dev_out_h1 = cl::create_buffer::<T>(
            params.c3 * params.width * params.height,
            flags::MEM_READ_WRITE,
            &queue,
        ).unwrap();
        let dev_out_v2 = cl::create_buffer::<T>(
            params.c2 * params.width / 2 * params.height / 2,
            flags::MEM_READ_WRITE,
            &queue,
        ).unwrap();
        let dev_out_h2 = cl::create_buffer::<T>(32 * 48 * 48, flags::MEM_READ_WRITE, &queue).unwrap();
        let dev_v1 = cl::create_buffer::<T>(1 * 5 * 3 * 7, flags::MEM_READ_ONLY, &queue).unwrap();
        let dev_h1 = cl::create_buffer::<T>(1 * 5 * 7 * 32, flags::MEM_READ_ONLY, &queue).unwrap();
        let dev_v2 = cl::create_buffer::<T>(1 * 5 * 32 * 7, flags::MEM_READ_ONLY, &queue).unwrap();
        let dev_h2 = cl::create_buffer::<T>(1 * 5 * 7 * 32, flags::MEM_READ_ONLY, &queue).unwrap();
        let dev_mxp_h1 = cl::create_buffer::<T>(48 * 48 * 32, flags::MEM_READ_WRITE, &queue).unwrap();
        let dev_mxp_h2 = cl::create_buffer::<T>(24 * 24 * 32, flags::MEM_READ_WRITE, &queue).unwrap();

        // TODO: check if this bin-read implementation matches with what's done in the original sep-conv
        let v1 = T::read_bin_from_file("input/weights/sepconv-_-_/v1-f32-le.bin");
        let h1 = T::read_bin_from_file("input/weights/sepconv-_-_/h1-f32-le.bin");
        let v2 = T::read_bin_from_file("input/weights/sepconv-_-_/v2-f32-le.bin");
        let h2 = T::read_bin_from_file("input/weights/sepconv-_-_/h2-f32-le.bin");

        // Write buffers to device
        dev_v1.write(&v1).enq().unwrap();
        dev_h1.write(&h1).enq().unwrap();
        dev_v2.write(&v2).enq().unwrap();
        dev_h2.write(&h2).enq().unwrap();

        // Create kernels
        let krn_c_conv1 = Kernel::new("colConv", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width, params.height, params.c2))
            .lws(SpatialDims::Three(
                params.columns_blockdim_x,
                params.columns_blockdim_y,
                1,
            ))
            .arg_buf(&dev_input)
            .arg_buf(&dev_out_v1)
            .arg_buf(&dev_v1);
        let krn_r_conv1 = Kernel::new("rowConv", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width, params.height, params.c3))
            // HACK: my desktop GPU cannot handle the full dimension (params.rows_blockdim_x*params.rows_blockdim_y*1 = 384) use 256 instead
            // TODO: benchmark different combinations: 32x4x2 -> 32x8x1
            .lws(SpatialDims::Three(
                params.c3,
                params.rows_blockdim_y,
                2,
            ))
            .arg_buf(&dev_out_v1)
            .arg_buf(&dev_out_h1)
            .arg_buf(&dev_h1);
        let krn_max_pool1 = Kernel::new("MaxPool1", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width, params.height, 32))
            // HACK: my desktop GPU cannot handle the full dimension (params.mp1_block_dim*params.mp1_block_dim*1 = 384) use 256 instead
            // TODO: benchmark different combinations: 32x8x1 -> 32x4x2
            .lws(SpatialDims::Three(
                params.mp1_block_dim,
                params.mp1_block_dim/4,
                1,
            ))
            .arg_buf(&dev_out_h1)
            .arg_buf(&dev_mxp_h1);
        let krn_c_conv2 = Kernel::new("colConv2", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width / 2, params.height / 2, 7))
            .lws(SpatialDims::Three(
                params.columns2_blockdim_x,
                params.columns2_blockdim_y,
                1,
            ))
            .arg_buf(&dev_mxp_h1)
            .arg_buf(&dev_out_v2)
            .arg_buf(&dev_v2);
        let krn_r_conv2 = Kernel::new("rowConv2", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width / 2, params.height / 2, 32))
            .lws(SpatialDims::Three(
                params.rows2_blockdim_x,
                params.rows2_blockdim_y,
                1,
            ))
            .arg_buf(&dev_out_v2)
            .arg_buf(&dev_out_h2)
            .arg_buf(&dev_h2);
        let krn_max_pool2 = Kernel::new("MaxPool2", &program).unwrap()
            .queue(queue.clone())
            .gws(SpatialDims::Three(params.width / 2, params.height / 2, 32))
            .lws(SpatialDims::Three(
                params.mp2_block_dim,
                params.mp2_block_dim,
                1,
            ))
            .arg_buf(&dev_out_h2)
            .arg_buf(&dev_mxp_h2);

        SepconvNetwork {
            in_buf: dev_input,
            krn_c_conv1,
            krn_r_conv1,
            krn_max_pool1,
            krn_c_conv2,
            krn_r_conv2,
            krn_max_pool2,
            dev_mxp_h2,
            dense4,
            dense5,
        }
    }
    // Maps the input buffer, and runs the network, returning the result.
    pub fn predict(&self, input_data: &[T], queue: &Queue) -> Vec<T> {
        unsafe {
            cl::map_to_buf(&self.in_buf, &input_data).unwrap();

            self.krn_c_conv1.cmd().queue(&queue).enq().unwrap();
            self.krn_r_conv1.cmd().queue(&queue).enq().unwrap();
            self.krn_max_pool1.cmd().queue(&queue).enq().unwrap();
            self.krn_c_conv2.cmd().queue(&queue).enq().unwrap();
            self.krn_r_conv2.cmd().queue(&queue).enq().unwrap();
            self.krn_max_pool2.cmd().queue(&queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        let mxp_out = relu(&unsafe { cl::read_buf(&self.dev_mxp_h2).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = mtxmul_relu(&mxp_out, &self.dense4);

        // Run the 5th layer (fully-connected)
        mtxmul_softmax(&dense4_out, &self.dense5)
    }
}
