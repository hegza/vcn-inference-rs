#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate ocl;
extern crate rusty_cnn;

mod common;

use cl_util as cl;
use common::*;
use criterion::Criterion;
use ocl::{Device, SpatialDims};
use rusty_cnn::*;

// Sample size of 100 puts the max-min of the benches at around 10 us at worst.
const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    // Initialize sepconv network
    let mut p = SEPCONV_HYPER_PARAMS.clone();

    // HACK: Reduce dimensions of overshot layers
    SepconvNetwork::<f32>::fix_params_for_default_gpu(&mut p);

    let layers = SepconvNetwork::<f32>::create_layers(&p, sepconv::Weights::default());

    bench_sepconv1(&layers, &p, c);
    bench_sepconv2(&layers, &p, c);
    bench_sepconv1and2(&layers, &p, c);

    // Initialize classic network
    let net = ClassicNetwork::create_layers(&CLASSIC_HYPER_PARAMS);

    // Create shorthands (and move)
    let (conv1, conv2, ..) = net;

    bench_conv1(&conv1, c);
    bench_conv2(&conv2, c);
    bench_conv1and2(&conv1, &conv2, c);
    //bench_dense3_cl_gpu(&dense3, c);
    // TODO: cl_[c|g]pu_vec16
}

fn bench_sepconv1(layers: &sepconv::Layers<f32>, p: &SepconvHyperParams, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/in.bin",
        SEPCONV_BASELINE
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv1, hconv1, mxp1) = (&layers.0, &layers.1, &layers.2);

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
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.vconv1_blockdim_x, p.vconv1_blockdim_y)),
    );
    let krn_hconv1 = b.build_iow_kernel(
        hconv1,
        "row_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side, p.hconv1_blockdim_y)),
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

    c.bench_function("layer 1 - cl gpu sepconv v+h+mxp", move |b| {
        b.iter(|| {
            unsafe {
                krn_vconv1.cmd().queue(&queue).enq().unwrap();
                krn_hconv1.cmd().queue(&queue).enq().unwrap();
                krn_max_pool1.cmd().queue(&queue).enq().unwrap();
            }
            queue.finish().unwrap()
        })
    });
}

fn bench_sepconv2(layers: &sepconv::Layers<f32>, p: &SepconvHyperParams, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/mxp1-out.bin",
        SEPCONV_BASELINE
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv2, hconv2, mxp2) = (&layers.3, &layers.4, &layers.5);

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
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.vconv2_blockdim_x, p.vconv1_blockdim_y)),
    );
    let krn_hconv2 = b.build_iow_kernel(
        hconv2,
        "row_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y)),
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

    c.bench_function("layer 2 - cl gpu sepconv v+h+mxp", move |b| {
        b.iter(|| {
            unsafe {
                krn_vconv2.cmd().queue(&queue).enq().unwrap();
                krn_hconv2.cmd().queue(&queue).enq().unwrap();
                krn_max_pool2.cmd().queue(&queue).enq().unwrap();
            }
            queue.finish().unwrap()
        })
    });
}

fn bench_sepconv1and2(layers: &sepconv::Layers<f32>, p: &SepconvHyperParams, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/in.bin",
        SEPCONV_BASELINE
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init::<f32>(
        &["src/cl/sepconv.cl", "src/cl/max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers)
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2) = (
        &layers.0, &layers.1, &layers.2, &layers.3, &layers.4, &layers.5,
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
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.vconv1_blockdim_x, p.vconv1_blockdim_y)),
    );
    let krn_hconv1 = b.build_iow_kernel(
        hconv1,
        "row_conv",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side, p.hconv1_blockdim_y)),
    );
    let krn_max_pool1 = b.build_io_kernel(
        mxp1,
        "max_pool_1",
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );
    let krn_vconv2 = b.build_iow_kernel(
        vconv2,
        "col_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.vconv2_blockdim_x, p.vconv1_blockdim_y)),
    );
    let krn_hconv2 = b.build_iow_kernel(
        hconv2,
        "row_conv_2",
        LocalWorkSizePolicy::Specify(SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y)),
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

    c.bench_function("layers 1 + 2 - cl gpu sepconv v+h+mxp", move |b| {
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
    });
}

fn bench_conv1(conv1: &ConvLayer<f32>, c: &mut Criterion) {
    let cl_layer = conv1.impl_standalone(
        &["src/cl/conv_relu.cl", "src/cl/mtx_mul.cl"],
        "conv_relu_1",
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    c.bench_function("layer 1 - cl gpu conv", move |b| {
        b.iter(|| cl_layer.dry_run())
    });
}

fn bench_conv2(conv2: &ConvLayer<f32>, c: &mut Criterion) {
    let cl_layer = conv2.impl_standalone(
        &["src/cl/conv_relu.cl", "src/cl/mtx_mul.cl"],
        "conv_relu_2",
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    c.bench_function("layer 2 - cl gpu conv", move |b| {
        b.iter(|| cl_layer.dry_run())
    });
}

fn bench_conv1and2(conv1: &ConvLayer<f32>, conv2: &ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        *conv1.input_shape(),
    ));

    let (queue, program, _context) =
        cl::init::<f32>(&["src/cl/conv_relu.cl", "src/cl/mtx_mul.cl"], &[], None);

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

    c.bench_function("layers 1 + 2 - cl gpu conv", move |b| {
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
    });
}

/*
 * Disabled: Running mtx_mul on GPU seems to be around more than 10x slower than what it is on a CPU.
 * layer 3 - cl gpu mtxmul time:   [2.1539 ms 2.1547 ms 2.1555 ms]
/*
fn bench_dense3_cl_gpu(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let cl_layer = dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    c.bench_function("layer 3 - cl gpu mtxmul", move |b| {
        b.iter(|| cl_layer.dry_run())
    });
}
*/
*/

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
