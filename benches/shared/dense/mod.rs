pub mod dense3;

pub use self::dense3::*;

use super::*;
use criterion::{black_box, Criterion};
use ocl::flags::*;
use rusty_cnn::*;

lazy_static! {
    static ref PARAMS: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS.clone());
}

pub fn bench_dense4(id: &str, c: &mut Criterion) {
    let dense4 = PARAMS.create_dense::<f32>(4, Weights::default().3);
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fc3.f",
        CLASSIC_BASELINE
    )));
    c.bench_function(id, move |b| b.iter(|| relu(dense4.compute(&input_data))));
}

pub fn bench_dense5(id: &str, c: &mut Criterion) {
    let dense5 = PARAMS.create_dense::<f32>(5, Weights::default().4);
    let input_data = black_box(f32::read_lines_from_file(&format!(
        "{}/fc4.f",
        CLASSIC_BASELINE
    )));
    c.bench_function(id, move |b| {
        b.iter(|| softmax(&dense5.compute(&input_data)))
    });
}
