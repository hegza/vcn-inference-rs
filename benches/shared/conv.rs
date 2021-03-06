use criterion::{black_box, Bencher};
use ocl::{Device, SpatialDims};
use rusty_cnn::cl_util as cl;
use rusty_cnn::*;

lazy_static! {
    static ref SEPCONV_PARAMS: sepconv::SepconvHyperParams = {
        let mut p = sepconv::SEPCONV_HYPER_PARAMS.clone();
        sepconv::ClNetwork::<f32>::fix_params_for_default_gpu(&mut p);
        p
    };
    static ref CLASSIC_LAYERS: classic::Layers<f32> =
        { classic::Layers::<f32>::new(classic::Weights::default()) };
}

pub fn bench_sepconv1() -> impl FnMut(&mut Bencher) {
    let layers = sepconv::Layers::<f32>::new(sepconv::Weights::default());

    let input_data = black_box(f32::read_bin_from_file(TEST_IMAGE_BIN_PATH));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &sepconv::ClNetwork::<f32>::compile_flags(&SEPCONV_PARAMS, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv1, hconv1, mxp1) = (&layers.vconv1, &layers.hconv1, &layers.mxp1);

    let wgts_bufs = create_weights_bufs(&[vconv1, hconv1], &queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(&[vconv1, hconv1, mxp1], &queue);

    // Build OpenCL-kernels
    let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
    let dev_max_wgs = cl::max_wgs(Some(&primary_device));
    let mut b = ClKernelChainBuilder::<f32>::new(&bufs, &wgts_bufs, &program, queue.clone());
    let krn_vconv1 = b.build_iow_kernel(
        vconv1,
        "col_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.vconv1_blockdim_x,
            SEPCONV_PARAMS.vconv1_blockdim_y,
        )),
    );
    let krn_hconv1 = b.build_iow_kernel(
        hconv1,
        "row_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.side,
            SEPCONV_PARAMS.hconv1_blockdim_y,
        )),
    );
    let krn_max_pool1 = b.build_io_kernel(
        mxp1,
        "max_pool_1",
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    move |b| {
        b.iter(|| {
            unsafe {
                krn_vconv1.cmd().queue(&queue).enq().unwrap();
                krn_hconv1.cmd().queue(&queue).enq().unwrap();
                krn_max_pool1.cmd().queue(&queue).enq().unwrap();
            }
            queue.finish().unwrap()
        })
    }
}

pub fn bench_sepconv2() -> impl FnMut(&mut Bencher) {
    let layers = sepconv::Layers::<f32>::new(sepconv::Weights::default());
    let input_data = black_box(f32::read_csv(&format!(
        "{}/mxp1_out-cwh.csv",
        VCN_SEPCONV_F32_BASELINE_DIR
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &sepconv::ClNetwork::<f32>::compile_flags(&SEPCONV_PARAMS, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv2, hconv2, mxp2) = (&layers.vconv2, &layers.hconv2, &layers.mxp2);

    let wgts_bufs = create_weights_bufs(&[vconv2, hconv2], &queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(&[vconv2, hconv2, mxp2], &queue);

    // Build OpenCL-kernels
    let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
    let dev_max_wgs = cl::max_wgs(Some(&primary_device));
    let mut b = ClKernelChainBuilder::new(&bufs, &wgts_bufs, &program, queue.clone());
    let krn_vconv2 = b.build_iow_kernel(
        vconv2,
        "col_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.vconv2_blockdim_x,
            SEPCONV_PARAMS.vconv1_blockdim_y,
        )),
    );
    let krn_hconv2 = b.build_iow_kernel(
        hconv2,
        "row_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.side / 2,
            SEPCONV_PARAMS.hconv2_blockdim_y,
        )),
    );
    let krn_max_pool2 = b.build_io_kernel(
        mxp2,
        "max_pool_2",
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    move |b| {
        b.iter(|| {
            unsafe {
                krn_vconv2.cmd().queue(&queue).enq().unwrap();
                krn_hconv2.cmd().queue(&queue).enq().unwrap();
                krn_max_pool2.cmd().queue(&queue).enq().unwrap();
            }
            queue.finish().unwrap()
        })
    }
}

pub fn bench_sepconv1and2() -> impl FnMut(&mut Bencher) {
    let layers = sepconv::Layers::<f32>::new(sepconv::Weights::default());

    let input_data = black_box(f32::read_bin_from_file(&format!(
        "{}/../in.bin",
        VCN_SEPCONV_F32_BASELINE_DIR
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &sepconv::ClNetwork::<f32>::compile_flags(&SEPCONV_PARAMS, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2) = (
        &layers.vconv1,
        &layers.hconv1,
        &layers.mxp1,
        &layers.vconv2,
        &layers.hconv2,
        &layers.mxp2,
    );

    let wgts_bufs = create_weights_bufs(&[vconv1, hconv1, vconv2, hconv2], &queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(&[vconv1, hconv1, mxp1, vconv2, hconv2, mxp2], &queue);

    // Build OpenCL-kernels
    let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
    let dev_max_wgs = cl::max_wgs(Some(&primary_device));
    let mut b = ClKernelChainBuilder::new(&bufs, &wgts_bufs, &program, queue.clone());
    let krn_vconv1 = b.build_iow_kernel(
        vconv1,
        "col_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.vconv1_blockdim_x,
            SEPCONV_PARAMS.vconv1_blockdim_y,
        )),
    );
    let krn_hconv1 = b.build_iow_kernel(
        hconv1,
        "row_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.side,
            SEPCONV_PARAMS.hconv1_blockdim_y,
        )),
    );
    let krn_max_pool1 = b.build_io_kernel(
        mxp1,
        "max_pool_1",
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );
    let krn_vconv2 = b.build_iow_kernel(
        vconv2,
        "col_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.vconv2_blockdim_x,
            SEPCONV_PARAMS.vconv1_blockdim_y,
        )),
    );
    let krn_hconv2 = b.build_iow_kernel(
        hconv2,
        "row_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            SEPCONV_PARAMS.side / 2,
            SEPCONV_PARAMS.hconv2_blockdim_y,
        )),
    );
    let krn_max_pool2 = b.build_io_kernel(
        mxp2,
        "max_pool_2",
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    move |b| {
        b.iter(|| {
            unsafe {
                krn_vconv1.cmd().queue(&queue).enq().unwrap();
                krn_hconv1.cmd().queue(&queue).enq().unwrap();
                krn_max_pool1.cmd().queue(&queue).enq().unwrap();
                krn_vconv2.cmd().queue(&queue).enq().unwrap();
                krn_hconv2.cmd().queue(&queue).enq().unwrap();
                krn_max_pool2.cmd().queue(&queue).enq().unwrap();
            }
            queue.finish().unwrap()
        })
    }
}

pub fn bench_conv1() -> impl FnMut(&mut Bencher) {
    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = &CLASSIC_LAYERS.conv1;

    let cl_layer = conv1.impl_standalone(
        &["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"],
        "conv_relu_1",
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    move |b| b.iter(|| cl_layer.dry_run())
}

pub fn bench_conv2() -> impl FnMut(&mut Bencher) {
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = &CLASSIC_LAYERS.conv2;

    let cl_layer = conv2.impl_standalone(
        &["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"],
        "conv_relu_2",
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    move |b| b.iter(|| cl_layer.dry_run())
}

pub fn bench_conv1and2() -> impl FnMut(&mut Bencher) {
    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = &CLASSIC_LAYERS.conv1;
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = &CLASSIC_LAYERS.conv2;

    let input_data = black_box(read_image_with_padding_from_bin_in_channels(
        TEST_IMAGE_BIN_PATH,
        conv1.input_shape(),
    ));

    let (queue, program, _context) =
        cl::init::<f32>(&["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"], &[], None);

    let wgts_bufs = create_weights_bufs(&[conv1, conv2], &queue);
    let bufs = create_buffer_chain(&[conv1, conv2], &queue);

    let mut b = ClKernelChainBuilder::new(&bufs, &wgts_bufs, &program, queue.clone());

    // Create the kernels for the first two layers (Convolution + ReLU)
    let conv_relu1 = b.build_iow_kernel(conv1, "conv_relu_1", LocalWorkSizePolicy::UseDefault);
    let conv_relu2 = b.build_iow_kernel(conv2, "conv_relu_2", LocalWorkSizePolicy::UseDefault);

    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    move |b| {
        b.iter(|| {
            unsafe {
                // Enqueue the kernel for the 1st layer (Convolution + ReLU)
                conv_relu1.cmd().queue(&queue).enq().unwrap();
                // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
                conv_relu2.cmd().queue(&queue).enq().unwrap();
            }
            // Wait for all on-device calculations to finish
            queue.finish().unwrap();
        })
    }
}
