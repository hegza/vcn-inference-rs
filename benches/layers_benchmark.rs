#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate ocl;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use ocl::{Device, SpatialDims};
use cl_util as cl;

const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.08;
const BASELINE_DIR: &'static str = "input/baseline";

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    let net = ClassicNetwork::create_layers(&CLASSIC_HYPER_PARAMS);
    // Create shorthands (and move)
    let (conv1, conv2, dense3, dense4, dense5) = net;

    bench_conv1(conv1.clone(), c);
    bench_conv2(conv2.clone(), c);
    bench_conv1and2(conv1, conv2, c);
    bench_dense3_cl_gpu(dense3.clone(), c);
    bench_dense3_cl_cpu(dense3.clone(), c);
    bench_dense3_host_ndarray(dense3.clone(), c);
    bench_dense3_host(dense3, c);
    bench_dense4(dense4, c);
    bench_dense5(dense5, c);
    // TODO: make it easier to implement parts of the whole network
    bench_sepconv1(c);
    bench_sepconv2(c);
    bench_sepconv1and2(c);
}

fn bench_sepconv1(c: &mut Criterion) {
    let mut p = SEPCONV_HYPER_PARAMS.clone();

    // HACK: Reduce dimensions of overshot layers
    SepconvNetwork::<f32>::fix_params_for_default_gpu(&mut p);

    let layers = SepconvNetwork::<f32>::create_layers(&p);
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/sepconv-f32-xcorr/in.bin",
        BASELINE_DIR
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init(
        &["sepconv.cl", "max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers),
    ).expect("cannot init OpenCL");

    let (vconv1, hconv1, mxp1, ..) = layers;

    let v1_wgts_buf = vconv1.create_wgts_buf(&queue);
    let h1_wgts_buf = hconv1.create_wgts_buf(&queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(&[&vconv1.0, &hconv1.0, &mxp1], &queue);

    // Build OpenCL-kernels
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

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    c.bench_function("layer 1 - cl sep-conv v+h+mxp", move |b| {
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

fn bench_sepconv2(c: &mut Criterion) {
    let mut p = SEPCONV_HYPER_PARAMS.clone();

    // HACK: Reduce dimensions of overshot layers
    SepconvNetwork::<f32>::fix_params_for_default_gpu(&mut p);

    let layers = SepconvNetwork::<f32>::create_layers(&p);
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/sepconv-f32-xcorr/mxp1-out.bin",
        BASELINE_DIR
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init(
        &["sepconv.cl", "max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers),
    ).expect("cannot init OpenCL");

    let (_, _, _, vconv2, hconv2, mxp2, ..) = layers;

    let v2_wgts_buf = vconv2.create_wgts_buf(&queue);
    let h2_wgts_buf = hconv2.create_wgts_buf(&queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(&[&vconv2.0, &hconv2.0, &mxp2], &queue);

    // Build OpenCL-kernels
    let primary_device = Device::from(*program.devices().unwrap().first().unwrap());
    let dev_max_wgs = cl::max_wgs(Some(&primary_device));
    let b = ClKernelBuilder::new(&program, queue.clone());
    let krn_vconv2 = b.build_iow_kernel(
        "col_conv_2",
        vconv2.gws_hint(),
        SpatialDims::Two(p.vconv2_blockdim_x, p.vconv1_blockdim_y),
        &bufs[0],     // In
        &bufs[1],     // Out
        &v2_wgts_buf, // Weights
    );
    let krn_hconv2 = b.build_iow_kernel(
        "row_conv_2",
        hconv2.gws_hint(),
        SpatialDims::Two(p.side / 2, p.hconv2_blockdim_y),
        &bufs[1],     // In
        &bufs[2],     // Out
        &h2_wgts_buf, // Weights
    );
    let krn_max_pool2 = b.build_io_kernel(
        "max_pool_2",
        mxp2.gws_hint(),
        mxp2.lws_hint(dev_max_wgs),
        &bufs[2], // In
        &bufs[3], // Out
    );

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    c.bench_function("layer 2 - cl sep-conv v+h+mxp", move |b| {
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

fn bench_sepconv1and2(c: &mut Criterion) {
    let mut p = SEPCONV_HYPER_PARAMS.clone();

    // HACK: Reduce dimensions of overshot layers
    SepconvNetwork::<f32>::fix_params_for_default_gpu(&mut p);

    let layers = SepconvNetwork::<f32>::create_layers(&p);
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/sepconv-f32-xcorr/in.bin",
        BASELINE_DIR
    )));

    // Init OpenCL
    let (queue, program, _context) = cl::init(
        &["sepconv.cl", "max_pool.cl"],
        &SepconvNetwork::<f32>::compile_flags(&p, &layers),
    ).expect("cannot init OpenCL");

    let (vconv1, hconv1, mxp1, vconv2, hconv2, mxp2, ..) = layers;

    let v1_wgts_buf = vconv1.create_wgts_buf(&queue);
    let h1_wgts_buf = hconv1.create_wgts_buf(&queue);
    let v2_wgts_buf = vconv2.create_wgts_buf(&queue);
    let h2_wgts_buf = hconv2.create_wgts_buf(&queue);

    // Allocate memory on-device for the I/O buffers
    let bufs = create_buffer_chain(
        &[&vconv1.0, &hconv1.0, &mxp1, &vconv2.0, &hconv2.0, &mxp2],
        &queue,
    );

    // Build OpenCL-kernels
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

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    c.bench_function("layers 1 + 2 - cl sep-conv v+h+mxp", move |b| {
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

fn bench_conv1(conv1: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/orig-f32-all-layers/in.bin", BASELINE_DIR),
        *conv1.input_shape(),
    ));
    let (kernel, _, queue) = create_standalone_kernel(&conv1, "conv_relu_1", &input_data).unwrap();
    c.bench_function("layer 1 - cl conv", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_conv2(conv2: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm1.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&conv2, "conv_relu_2", &input_data).unwrap();
    c.bench_function("layer 2 - cl conv", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_conv1and2(conv1: ConvLayer<f32>, conv2: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/orig-f32-all-layers/in.bin", BASELINE_DIR),
        *conv1.input_shape(),
    ));

    // Initialize OpenCL
    let (queue, program, _context) = cl::init(&["conv_relu.cl", "mtx_mul.cl"], &[]).unwrap();

    // Allocate read-only memory for the weights of the 1st three layers
    let conv1_wgts_buf = conv1.create_wgts_buf(&queue);
    let conv2_wgts_buf = conv2.create_wgts_buf(&queue);

    // Allocate read-only memory for the input geometry on device with host-accessible pointer for
    // writing input from file
    let bufs = create_buffer_chain(&[&conv1, &conv2], &queue);

    // Create the kernel for the 1st layer (Convolution + ReLU)
    let conv_relu1 = ocl::Kernel::builder().program(&program).name("conv_relu_1")
            .queue(queue.clone())
            .global_work_size(conv1.gws_hint())
            // Input
            .arg(&bufs[0])
            // Output
            .arg(&bufs[1])
            .arg(&conv1_wgts_buf).build().unwrap();

    // Create the kernel for the 2nd layer (Convolution + ReLU)
    let conv_relu2 = ocl::Kernel::builder().program(&program).name("conv_relu_2")
            .queue(queue.clone())
            .global_work_size(conv2.gws_hint())
            // Input
            .arg(&bufs[1])
            // Output
            .arg(&bufs[2])
            .arg(&conv2_wgts_buf).build().unwrap();

    unsafe {
        cl::map_to_buf(&bufs[0], &input_data).unwrap();
    }

    // Wait for setup to finish
    queue.finish().unwrap();

    c.bench_function("layers 1 + 2 - cl conv", move |b| {
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

fn bench_dense3_cl_gpu(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&dense3, "mtx_mul_f32", &input_data).unwrap();
    c.bench_function("layer 3 - cl gpu mtxmul", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_dense3_cl_cpu(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) =
        create_standalone_kernel_cpu(&dense3, "mtx_mul_f32", &input_data).unwrap();
    c.bench_function("layer 3 - cl cpu mtxmul", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_dense3_host(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 3 - host mtxmul", move |b| {
        b.iter(|| dense3.mtx_mul(&input_data))
    });
}

fn bench_dense3_host_ndarray(dense3: DenseLayer<f32>, cr: &mut Criterion) {
    use ndarray::*;
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    let m = dense3.num_out();
    let n = dense3.num_in();
    let a = Array2::<f32>::from_shape_vec((m, n), dense3.weights().clone()).unwrap();
    let k = 1;
    let b = Array2::<f32>::from_shape_vec((n, k), input_data).unwrap();
    cr.bench_function("layer 3 - host ndarray mtxmul", move |be| {
        be.iter(|| a.dot(&b))
    });
}

fn bench_dense4(dense4: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fc3.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 4 - host mtxmul", move |b| {
        b.iter(|| relu(&dense4.mtx_mul(&input_data)))
    });
}

fn bench_dense5(dense5: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fc4.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 5 - host mtxmul", move |b| {
        b.iter(|| softmax(&dense5.mtx_mul(&input_data)))
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
