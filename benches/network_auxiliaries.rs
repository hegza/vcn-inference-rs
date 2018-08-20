//! The purpose of this benchmark is to measure the impact and meaning of auxiliary functions in the
//! calculations. Examples include network compilation and moving of inputs between memories.

#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;
extern crate matrixmultiply;
extern crate ndarray;
extern crate num_traits;
extern crate ocl;
extern crate rand;
extern crate rusty_cnn;

mod shared;

use criterion::{Bencher, Benchmark, Criterion};
use rusty_cnn::{classic, sepconv};

const SAMPLE_SIZE: usize = 3;
const NOISE_THRESHOLD: f64 = 0.1;

pub fn bench_classic_f32_init() -> (&'static str, impl FnMut(&mut Bencher)) {
    ("classic-f32 init", move |b: &mut Bencher| {
        b.iter(|| classic::ClNetwork::<f32>::new(classic::Weights::default()))
    })
}

pub fn bench_sepconv_f32_init() -> (&'static str, impl FnMut(&mut Bencher)) {
    ("sepconv-f32 init", move |b: &mut Bencher| {
        b.iter(|| sepconv::ClNetwork::<f32>::new(sepconv::Weights::default()))
    })
}

fn bench_auxiliaries(c: &mut Criterion) {
    let (classic_init_id, classic_init) = bench_classic_f32_init();
    let (sepconv_init_id, sepconv_init) = bench_sepconv_f32_init();

    let bench =
        Benchmark::new(classic_init_id, classic_init).with_function(sepconv_init_id, sepconv_init);

    c.bench("network init", bench);
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_auxiliaries
}
criterion_main!(benches);
