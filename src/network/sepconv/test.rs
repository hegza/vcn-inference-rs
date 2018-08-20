#[cfg(test)]
mod test;

fn run_sepconv_f32() -> Vec<f32> {
    let net = sepconv::ClNetwork::<f32>::new(sepconv::Weights::default());
    let input_data = f32::read_bin_from_file(&format!("{}/in.bin", SEPCONV_BASELINE));
    net.predict(&input_data)
}

fn run_sepconv_i8() -> Vec<f32> {
    use std::i8;
    let mut rng = rand::thread_rng();

    // HACK: Random-generate weights for now
    let wgts = sepconv::Weights(
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
        // Dense layers
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
    let net = sepconv::ClNetwork::<i8>::new(wgts);
    // TODO: load real input data
    //let input_data = i8::read_bin_from_file("input/baseline/sepconv-f32-xcorr/in.bin");
    let input_data: Vec<i8> = (0..96 * 96 * 3)
        .map(|_| rng.gen_range(i8::MIN, i8::MAX))
        .collect();
    net.predict(&input_data)
}

#[test]
fn sepconv_f32_predicts() {
    let output = run_sepconv_f32();
    let correct = f32::read_lines_from_file(&format!("{}/f32/out.f", SEPCONV_BASELINE));

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

#[test]
fn sepconv_i8_runs() {
    let output = run_sepconv_i8();
    assert_eq!(output.len(), 4);

    // TODO: verify correctness
    //let correct = i8::read_lines_from_file("X");
    //verify(&output, &correct, COARSE_RESULT_MARGIN);
}
