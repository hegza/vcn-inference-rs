use super::*;

pub fn bench_dense3_cl_cpu(id: &str, c: &mut Criterion) {
    let dense3 = PARAMS.create_dense::<f32>(3, Weights::default().2);
    let cl_layer = dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &[],
        Some(DeviceType::CPU),
        LocalWorkSizePolicy::UseDefault,
    );

    c.bench_function(id, move |b| b.iter(|| cl_layer.dry_run()));
}

pub fn bench_dense3_host_ndarray(id: &str, cr: &mut Criterion) {
    let dense3 = PARAMS.create_dense::<f32>(3, Weights::default().2);

    use ndarray::*;
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fm2.f",
        CLASSIC_BASELINE
    )));
    let m = dense3.num_out();
    let n = dense3.num_in();
    let a = Array2::<f32>::from_shape_vec((m, n), dense3.weights().clone()).unwrap();
    let k = 1;
    let b = Array2::<f32>::from_shape_vec((n, k), input_data).unwrap();
    cr.bench_function(id, move |be| be.iter(|| a.dot(&b)));
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
