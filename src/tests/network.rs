use super::*;

#[test]
fn test_network() {
    let output = run_network().unwrap();
    let correct = f32::read_from_file(&format!("{}/out5.f", BASELINE_DIR));
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_network() -> ocl::Result<Vec<f32>> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init("original_kernels.cl")?;

    let net = Network::<f32>::new(&program, &queue).unwrap();
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        *net.conv1.input_shape(),
    );

    net.predict(&input_data, &queue)
}
