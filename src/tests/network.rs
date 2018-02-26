use super::*;

#[test]
fn test_classic() {
    let output = run_classic().unwrap();
    let correct = f32::read_from_file(&format!("{}/out5.f", BASELINE_DIR));
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_classic() -> ocl::Result<Vec<f32>> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init("original_kernels.cl")?;

    let net = ClassicNetwork::<f32>::new(&program, &queue).unwrap();
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        net.input_shape().clone(),
    );
    net.predict(&input_data, &queue)
}

#[test]
fn test_sepconv() {
    let output = run_sepconv();
    assert!(output.is_ok())
    // TODO: verify output correctness
    //let correct = f32::read_from_file(&format!("{}/out5.f", BASELINE_DIR));
    //assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_sepconv() -> ocl::Result<Vec<f32>> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init("sep_conv_kernels.cl")?;

    let net = SepconvNetwork::<f32>::new(&program, &queue).unwrap();
    let input_data = f32::read_bin_from_file(&format!("{}/in.bin", BASELINE_DIR));

    net.predict(&input_data, &queue)
}
