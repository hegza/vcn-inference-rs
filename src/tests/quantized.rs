use super::*;
use crate::VCN_SEPCONV_F32_BASELINE_DIR as QUANTIZED_BASELINE;

fn run_quantized(_input: &[u8]) -> Vec<f32> {
    Vec::new()
}

// `cargo test --feature quantized` to run this test
#[cfg_attr(not(feature = "quantized"), ignore)]
#[test]
fn quantized_predicts() {
    let input = math::quantize_vec_u8(
        &f32::read_lines_from_file(&format!("{}/f32/in.f", QUANTIZED_BASELINE)).unwrap(),
    );
    let output = run_quantized(&input);
    assert_eq!(output.len(), 4);

    let correct = f32::read_lines_from_file(&format!("{}/f32/out.f", QUANTIZED_BASELINE)).unwrap();
    verify(&output, &correct, COARSE_RESULT_MARGIN);
}
