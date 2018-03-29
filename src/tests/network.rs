use super::*;

#[test]
fn test_classic() {
    let output = run_classic();
    let correct = f32::read_lines_from_file(&format!("{}/out5.f", BASELINE_DIR));
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_sepconv() {
    let output = run_sepconv();
    assert_eq!(output.len(), 4);
    // TODO: verify output correctness
    //let correct = f32::read_lines_from_file(&format!("{}/out5.f", BASELINE_DIR));
    //assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_classic() -> Vec<f32> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init(&["conv_relu.cl", "mtx_mulf.cl"]).unwrap();

    let net = ClassicNetwork::<f32>::new(&program, &queue);
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        net.input_shape().clone(),
    );
    net.predict(&input_data, &queue)
}

fn run_sepconv() -> Vec<f32> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init(&["sepconv.cl", "mtx_mulf.cl"]).unwrap();

    let net = SepconvNetwork::<f32>::new(&program, &queue, true);
    let input_data = f32::read_bin_from_file("input/baseline/sepconv-f32-xcorr/in-le.bin");
    net.predict(&input_data, &queue)
}
