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
    let correct = f32::read_lines_from_file("input/baseline/sepconv-f32-xcorr/f32/out.f");

    assert!(
        is_within_margin(&output, &correct, COARSE_RESULT_MARGIN),
        "output is not within margin of correct: {:?} != {:?}",
        output,
        correct
    );
}

fn run_classic() -> Vec<f32> {
    let net = ClassicNetwork::<f32>::new();
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        net.input_shape().clone(),
    );
    net.predict(&input_data)
}

fn run_sepconv() -> Vec<f32> {
    let net = SepconvNetwork::<f32>::new();
    let input_data = f32::read_bin_from_file("input/baseline/sepconv-f32-xcorr/in.bin");
    net.predict(&input_data)
}
