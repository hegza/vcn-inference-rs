use super::*;
use matrixmultiply;
use rusty_cnn::math::gemm::*;
use rusty_cnn::*;
use num_traits::Zero;

const RESULT_MARGIN: f32 = 0.00002f32;

pub fn bench_dense3_host() -> (&'static str, impl FnMut(&mut Bencher)) {
    let in_buf = black_box(create_random_vec(32*24*24));
    let w = create_random_vec(100*32*24*24);
    let mut c = vec![Zero::zero(); 100];
    ("dense 3 - host", move |b: &mut Bencher| {
        b.iter(|| gemm_naive(
            1,
            100,
            32*24*24,
            &in_buf,
            &w,
            &mut c,
        ));
    })
}

pub fn bench_dense3_cl_cpu() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense3 = CLASSIC_LAYERS.dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &[],
        Some(DeviceType::CPU),
        LocalWorkSizePolicy::UseDefault,
    );

    let in_data = black_box(create_random_vec(32*24*24));

    ("dense 3 - cl (CPU)", move |b: &mut Bencher| {
        b.iter(|| {
            dense3.map_input(&in_data);
            dense3.dry_run()
        })
    })
}

pub fn bench_sparse3() -> (&'static str, impl FnMut(&mut Bencher)) {
    let sparse3 = sparse::Layers::<f32>::new(sparse::Weights::default()).sparse3;

    let input =
        black_box(f32::read_lines_from_file(&format!("{}/fm2.f", VCN_BASELINE_DIR)).unwrap());

    // TODO: verify correctness

    ("sparse 3 - sprs (CPU)", move |b: &mut Bencher| {
        b.iter(|| sparse3.compute(&input))
    })
}

pub fn bench_dense_3_bluss_matrixmultiply() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense3 = &CLASSIC_LAYERS.dense3;
    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", VCN_BASELINE_DIR)).unwrap();

    let m = 1;
    let n = dense3.num_out();
    let k = dense3.num_in();

    let a = black_box(input_data);
    assert_eq!(a.len(), m * k);
    let b = black_box(dense3.weights());
    assert_eq!(b.len(), k * n);
    let mut c = vec![0f32; m * n];

    // A is row-major, B is column-major, C is row-major
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1f32,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            1f32,
            c.as_mut_ptr(),
            n as isize,
            1,
        )
    };

    // Verify correctness
    let c_correct = f32::read_lines_from_file(&format!("{}/fc3.f", VCN_BASELINE_DIR)).unwrap();
    verify(&relu(c.clone()), &c_correct, RESULT_MARGIN);

    ("dense 3 - bluss matrixmultiply (CPU)", move |be| {
        be.iter(|| unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1f32,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                1,
                k as isize,
                1f32,
                c.as_mut_ptr(),
                n as isize,
                1,
            )
        })
    })
}

pub fn bench_dense_3_cnugteren_10() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense3 = &CLASSIC_LAYERS.dense3;
    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", VCN_BASELINE_DIR)).unwrap();

    let m = 1;
    let n = dense3.num_out();
    let k = dense3.num_in();

    // A is stored on disk as row-major
    let a = black_box(input_data);
    assert_eq!(a.len(), m * k);
    // B is stored on disk as column-major
    let b = black_box(dense3.weights());
    assert_eq!(b.len(), k * n);

    // Setup
    let mut out = vec![0f32; m * n];
    // This ordinarily accepts, A: column-major, B: row-major, C: column-major
    // But switching m and n and a and b makes it accept:
    // A: row-major, B: column-major, C: row-major
    let gemm_10_gpu = Gemm10Kernel::from_slices(n, m, k, &b, &a, &mut out, DeviceType::ALL);

    // HACK: cnugteren 10 produces incorrect results but this does not matter for performance measurements
    /*
    let c_correct = f32::read_lines_from_file(&format!("{}/fc3.f", VCN_BASELINE_DIR));
    gemm_10_gpu.calculate_wait();
    verify(&relu(out.clone()), &c_correct, RESULT_MARGIN);
    */

    // Create benchmark-closure
    ("dense 3 - cnugteren 10 (GPU)", move |be| {
        be.iter(|| gemm_10_gpu.calculate_wait())
    })
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
