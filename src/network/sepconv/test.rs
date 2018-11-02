use super::*;
use cl_util as cl;
use ndarray::Array;
use network::Predict;
use rand;
use rand::{Rng, ThreadRng};
use tests::*;

pub const SEPCONV_BASELINE_F32: &'static str = "input/baseline/sepconv-f32-xcorr/case b";

lazy_static! {
    static ref LAYERS: Layers<f32> = { Layers::<f32>::new(Weights::default()) };
    // HACK: Reduce dimensions of overshot layers
    static ref FIXED_SEPCONV_HYPER_PARAMS: SepconvHyperParams = {
        let mut p = SEPCONV_HYPER_PARAMS.clone();
        ClNetwork::<f32>::fix_params_for_default_gpu(&mut p);
        p
    };
    static ref ADDT_CMPLR_OPTS: Vec<String> =
        ClNetwork::<f32>::compile_flags(&FIXED_SEPCONV_HYPER_PARAMS, &LAYERS);
}

#[test]
fn v1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.vconv1;

    // Load image without padding and in (channels, height, width)-order
    let input_data = load_jpeg_chw(&format!("{}/in.jpg", SEPCONV_BASELINE_F32));

    // Output is produced in (channels, height, width)-order
    let output = run_single_layer(
        "col_conv",
        layer,
        &input_data,
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            FIXED_SEPCONV_HYPER_PARAMS.vconv1_blockdim_x,
            FIXED_SEPCONV_HYPER_PARAMS.vconv1_blockdim_y,
        )),
    );

    // Load model outputs in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (7, 96, 96),
        f32::read_csv(&format!("{}/vcr1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

#[test]
fn h1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.hconv1;

    // Load input in (channels, height, width)-order
    let input_data = Array::from_shape_vec(
        (7, 96, 96),
        f32::read_csv(&format!("{}/vcr1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    let (queue, program, _context) = cl_util::init::<f32>(
        &["src/cl/sepconv.cl"],
        &ADDT_CMPLR_OPTS
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
    );

    // Create buffers
    let wgts_buf = layer.create_wgts_buf(&queue);
    let (in_buf, out_buf) = layer.create_io_bufs(
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    );

    let kernel = layer.create_kernel(
        "row_conv",
        &in_buf,
        &out_buf,
        &wgts_buf,
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            FIXED_SEPCONV_HYPER_PARAMS.side,
            FIXED_SEPCONV_HYPER_PARAMS.hconv1_blockdim_y,
        )),
        &program,
        &queue,
    );

    // Enqueue kernel and wait for it to end, return the result
    let output = unsafe {
        cl_util::map_to_buf(&in_buf, &input_data).unwrap();
        kernel.cmd().queue(&queue).enq().unwrap();
        cl_util::read_buf(&out_buf).unwrap()
    };
    queue.finish().unwrap();

    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 96, 96),
        f32::read_csv(&format!("{}/hcr1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

#[test]
fn mxp1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.mxp1;

    // Load input in (channels, height, width)-order (this is what happens in the network)
    let input_data = Array::from_shape_vec(
        (32, 96, 96),
        f32::read_csv(&format!("{}/hcr1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    let device_a = cl_util::select_device(cl_util::DevicePreference::PreferGpu);
    let dev_max_wgs = cl_util::max_wgs(Some(&device_a));

    let output = run_single_layer_unweighted(
        "max_pool_1",
        layer,
        &input_data,
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 48, 48),
        f32::read_csv(&format!("{}/mxp1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

#[test]
fn v2_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.vconv2;

    // Load input in (channels, height, width)-order (this is what happens in the network)
    let input_data = Array::from_shape_vec(
        (32, 48, 48),
        f32::read_csv(&format!("{}/mxp1_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    let output = run_single_layer(
        "col_conv_2",
        layer,
        &input_data,
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            FIXED_SEPCONV_HYPER_PARAMS.vconv2_blockdim_x,
            FIXED_SEPCONV_HYPER_PARAMS.vconv1_blockdim_y,
        )),
    );

    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (7, 48, 48),
        f32::read_csv(&format!("{}/vcr2_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

#[test]
fn h2_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.hconv2;

    // Load input in (channels, height, width)-order (this is what happens in the network)
    let input_data = Array::from_shape_vec(
        (7, 48, 48),
        f32::read_csv(&format!("{}/vcr2_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    let output = run_single_layer(
        "row_conv_2",
        layer,
        &input_data,
        LocalWorkSizePolicy::Specify(SpatialDims::Two(
            FIXED_SEPCONV_HYPER_PARAMS.side / 2,
            FIXED_SEPCONV_HYPER_PARAMS.hconv2_blockdim_y,
        )),
    );
    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 48, 48),
        f32::read_csv(&format!("{}/hcr2_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

#[test]
fn mxp2_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.mxp2;

    // Load input in (channels, height, width)-order (this is what happens in the network)
    let input_data = Array::from_shape_vec(
        (32, 48, 48),
        f32::read_csv(&format!("{}/hcr2_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    let device_a = cl_util::select_device(cl_util::DevicePreference::PreferGpu);
    let dev_max_wgs = cl_util::max_wgs(Some(&device_a));

    let output = run_single_layer_unweighted(
        "max_pool_2",
        layer,
        &input_data,
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );
    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 24, 24),
        f32::read_csv(&format!("{}/mxp2_out-cwh.csv", SEPCONV_BASELINE_F32)),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l3_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense3;
    // Load input in (channels, height, width)-order
    let input_data = {
        let raw = f32::read_csv(&format!("{}/mxp2_out-cwh.csv", SEPCONV_BASELINE_F32));
        let chw = Array::from_shape_vec((32, 24, 24), raw)
            .unwrap()
            .permuted_axes((0, 2, 1))
            .into_iter()
            .cloned()
            .collect::<Vec<f32>>();
        chw
    };

    // Output is in (fc-const)-order
    let output = relu(layer.compute(&input_data));

    // Load model output in (fc-const)-order
    let correct = f32::read_csv(&format!("{}/fc3-out.csv", SEPCONV_BASELINE_F32));

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

#[test]
fn l4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense4;
    let input_data = f32::read_csv(&format!("{}/fc3-out.csv", SEPCONV_BASELINE_F32));

    let output = relu(layer.compute(&input_data));
    let correct = f32::read_csv(&format!("{}/fc4-out.csv", SEPCONV_BASELINE_F32));

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

#[test]
fn l5_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense5;
    let input_data = f32::read_csv(&format!("{}/fc4-out.csv", SEPCONV_BASELINE_F32));

    let output = &layer.compute(&input_data);
    let correct = f32::read_csv(&format!("{}/fc5-out.csv", SEPCONV_BASELINE_F32));

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

/*
fn run_sepconv_i8() -> Vec<f32> {
    use std::i8;
    let mut rng = rand::thread_rng();

    // HACK: Random-generate weights for now
    let wgts = Weights(
        // H/V convs
        (0..5 * 1 * 3 * 7)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..1 * 5 * 7 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..5 * 1 * 32 * 7)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..1 * 5 * 7 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        // Dense LAYERS
        (0..100 * 24 * 24 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..100 * 100)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..100 * 4)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
    );
    let net = ClNetwork::<i8>::new(wgts);
    // TODO: load real input data
    //let input_data = i8::read_csv("input/baseline/sepconv-f32-xcorr/case a/in.csv");
    let input_data: Vec<i8> = (0..96 * 96 * 3)
        .map(|_| rng.gen_range(i8::MIN, i8::MAX))
        .collect();
    net.predict(&input_data)
}
*/

#[test]
fn sepconv_f32_predicts() {
    let output = run_sepconv_f32();
    let correct = softmax(f32::read_csv(&format!(
        "{}/fc5-out.csv",
        SEPCONV_BASELINE_F32
    )));

    verify(&output, &correct, RESULT_MARGIN);
}

fn run_sepconv_f32() -> Vec<f32> {
    let net = ClNetwork::<f32>::new(Weights::default());
    let input_data = load_jpeg_chw(&format!("{}/in.jpg", SEPCONV_BASELINE_F32));
    net.predict(&input_data)
}

// HACK: these run_single_LAYERS are not the best thing to use for prototyping the sepconv
fn run_single_layer<L, T>(
    kernel_func: &str,
    layer: &L,
    input: &[T],
    lws_policy: LocalWorkSizePolicy,
) -> Vec<T>
where
    L: ClWeightedLayer<T>,
    T: Coeff,
{
    let cl_layer = layer.impl_standalone(
        &[
            "src/cl/sepconv.cl",
            "src/cl/max_pool.cl",
            "src/cl/mtx_mul.cl",
        ],
        kernel_func,
        &ADDT_CMPLR_OPTS
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
        lws_policy,
    );

    // Enqueue kernel and wait for it to end, return the result
    cl_layer.run_with_input(&input)
}

fn run_single_layer_unweighted<L, T>(
    kernel_func: &str,
    layer: &L,
    input: &[T],
    lws_policy: LocalWorkSizePolicy,
) -> Vec<T>
where
    L: ClUnweightedLayer<T>,
    T: Coeff,
{
    let cl_layer = layer.impl_standalone(
        &[
            "src/cl/sepconv.cl",
            "src/cl/max_pool.cl",
            "src/cl/mtx_mul.cl",
        ],
        kernel_func,
        &ADDT_CMPLR_OPTS
            .iter()
            .map(AsRef::as_ref)
            .collect::<Vec<&str>>(),
        None,
        lws_policy,
    );

    // Enqueue kernel and wait for it to end, return the result
    cl_layer.run_with_input(&input)
}
