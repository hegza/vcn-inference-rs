pub mod dense3;

pub use self::dense3::*;

use super::*;
use rusty_cnn::*;

lazy_static! {
    static ref CLASSIC_LAYERS: classic::Layers<f32> =
        { classic::Layers::<f32>::new(classic::Weights::default()) };
}

pub fn bench_dense4() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense4 = &CLASSIC_LAYERS.dense4;
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fc3.f",
        CLASSIC_BASELINE
    )));

    ("dense 4 - host mtxmul", move |b| {
        b.iter(|| relu(dense4.compute(&input_data)))
    })
}

pub fn bench_dense5() -> (&'static str, impl FnMut(&mut Bencher)) {
    let dense5 = &CLASSIC_LAYERS.dense5;
    let input_data =
        black_box(f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE)).unwrap());

    ("dense 5 - host mtxmul", move |b| {
        b.iter(|| softmax(dense5.compute(&input_data)))
    })
}
