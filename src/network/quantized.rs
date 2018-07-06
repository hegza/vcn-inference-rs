use super::*;

#[cfg(test)]
mod test {
    use super::*;
    use super::super::super::tests::*;

    const QUANTIZED_BASELINE: &'static str = SEPCONV_BASELINE;

    fn run_quantized(_input: &[u8]) -> Vec<f32> {
        Vec::new()
    }

    // `cargo test --feature test_quantize` to run this test
    #[cfg_attr(not(feature = "test_quantize"), ignore)]
    #[test]
    fn quantized_predicts() {
        let input = math::quantize_vec_u8(&f32::read_lines_from_file(&format!(
            "{}/f32/in.f",
            QUANTIZED_BASELINE
        )));
        let output = run_quantized(&input);
        assert_eq!(output.len(), 4);

        let correct = f32::read_lines_from_file(&format!("{}/f32/out.f", QUANTIZED_BASELINE));
        verify(&output, &correct, COARSE_RESULT_MARGIN);
    }
}
