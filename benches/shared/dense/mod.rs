pub mod dense3;

pub use self::dense3::*;

use super::*;
use rusty_cnn::*;

lazy_static! {
    static ref PARAMS: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS.clone());
}

pub fn bench_dense4() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense4 = PARAMS.create_dense::<f32>(4, Weights::default().3);
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fc3.f",
        CLASSIC_BASELINE
    )));

    ("dense 4 - host mtxmul", move |b| {
        b.iter(|| relu(dense4.compute(&input_data)))
    })
}

pub fn bench_dense5() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense5 = PARAMS.create_dense::<f32>(5, Weights::default().4);
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fc4.f",
        CLASSIC_BASELINE
    )));

    ("dense 5 - host mtxmul", move |b| {
        b.iter(|| softmax(&dense5.compute(&input_data)))
    })
}
