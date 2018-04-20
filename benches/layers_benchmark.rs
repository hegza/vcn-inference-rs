#[macro_use]
extern crate criterion;
extern crate ocl;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use ocl::{flags, Buffer, Device, Platform, SpatialDims, enums::*};
use cl_util as cl;

const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.08;
const BASELINE_DIR: &'static str = "input/baseline";

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    let net = ClassicNetwork::create_layers(&CLASSIC_HYPER_PARAMS);

    bench_layer1(net.conv1, c);
    bench_layer2(net.conv2, c);
    bench_layer3_cl(net.dense3.clone(), c);
    bench_layer3_host(net.dense3, c);
    bench_layer4(net.dense4, c);
    bench_layer5(net.dense5, c);
    bench_sepconv1(c);
}

fn bench_sepconv1(c: &mut Criterion) {
    let p = &SEPCONV_HYPER_PARAMS;
    // TODO: find common code between SepconvNetwork implementation and this, and refactor
    let (vconv1, hconv1, mxp1, ..) = SepconvNetwork::<f32>::create_layers(&p);
    let input_data = criterion::black_box(f32::read_bin_from_file(&format!(
        "{}/sepconv-f32-xcorr/in.bin",
        BASELINE_DIR
    )));

    // Init OpenCL
    let platform = Platform::default();
    let device = Device::first(platform).unwrap();
    let (hconv1_blockdim_y, mxp1_block_dim) =
        match device.info(DeviceInfo::MaxWorkGroupSize).unwrap() {
            DeviceInfoResult::MaxWorkGroupSize(max_wgs) => {
                // Max-work-size is enough for all the calculations here
                if max_wgs > 256 {
                    (p.hconv_blockdim_y, p.mp1_block_dim)
                }
                // HACK: Max-work-size is not enough, use halved dimensions for hconv1 and mxp 1
                else {
                    (p.hconv_blockdim_y / 2, p.mp1_block_dim / 2)
                }
            }
            e => panic!("ocl library returned invalid enum {:?}", e),
        };
    let (queue, program, _context) = cl::init(
        &["sepconv.cl", "max_pool.cl"],
        &[
            ("WIDTH", p.side as i32),
            ("HEIGHT", p.side as i32),
            ("MP1_BLOCK_DIM", mxp1_block_dim as i32),
            ("MP2_BLOCK_DIM", p.mp2_block_dim as i32),
            ("ROWS_BLOCKDIM_Y", hconv1_blockdim_y as i32),
            ("INJECT_RELU_AFTER_MXP", 1 as i32),
        ],
        platform,
    ).expect("cannot init OpenCL");

    let v1_wgts_buf = vconv1.create_wgts_buf(&queue).unwrap();
    let h1_wgts_buf = hconv1.create_wgts_buf(&queue).unwrap();

    // Allocate memory on-device for the I/O buffers
    let intermediary_flags = flags::MEM_READ_WRITE;
    let in_buf = vconv1
        .create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
        .unwrap();
    let conv1_mid_buf = vconv1.create_out_buf(intermediary_flags, &queue).unwrap();
    let conv1_out_buf = hconv1.create_out_buf(intermediary_flags, &queue).unwrap();
    let mxp1_out_buf: Buffer<f32> =
        mxp1.create_out_buf(flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR, &queue)
            .unwrap();

    // Build OpenCL-kernels
    let b = ClKernelBuilder::new(&program, queue.clone());
    let krn_vconv1 = b.build_iow_kernel(
        "col_conv",
        vconv1.gws_hint(),
        SpatialDims::Three(p.columns_blockdim_x, p.columns_blockdim_y, 1),
        &in_buf,        // In
        &conv1_mid_buf, // Out
        &v1_wgts_buf,   // Weights
    );
    let krn_hconv1 = b.build_iow_kernel(
        "row_conv",
        hconv1.gws_hint(),
        SpatialDims::Three(p.side, hconv1_blockdim_y, 1),
        &conv1_mid_buf, // In
        &conv1_out_buf, // Out
        &h1_wgts_buf,   // Weights
    );
    let krn_max_pool1 = b.build_io_kernel(
        "max_pool_1",
        mxp1.gws_hint(),
        SpatialDims::Three(mxp1_block_dim, mxp1_block_dim, 1),
        &conv1_out_buf, // In
        &mxp1_out_buf,  // Out
    );

    // Map input data to input buffer
    unsafe {
        cl::map_to_buf(&in_buf, &input_data).unwrap();
    }

    c.bench_function("layer 1 - cl sep-conv h+v+mxp", move |b| {
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

fn bench_layer1(conv1: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/orig-f32-all-layers/in.bin", BASELINE_DIR),
        *conv1.input_shape(),
    ));
    let (kernel, _, queue) = create_standalone_kernel(&conv1, "conv_relu_1", &input_data).unwrap();
    c.bench_function("layer 1 - cl conv", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer2(conv2: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm1.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&conv2, "conv_relu_2", &input_data).unwrap();
    c.bench_function("layer 2 - cl conv", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer3_cl(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&dense3, "mtx_mul_f32", &input_data).unwrap();
    c.bench_function("layer 3 - cl mtxmul", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer3_host(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fm2.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 3 - host mtxmul", move |b| {
        b.iter(|| dense3.mtx_mul(&input_data))
    });
}

fn bench_layer4(dense4: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/orig-f32-all-layers/fc3.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 4 - host mtxmul", move |b| {
        b.iter(|| relu(&dense4.mtx_mul(&input_data)))
    });
}

fn bench_layer5(dense5: DenseLayer<f32>, c: &mut Criterion) {
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
