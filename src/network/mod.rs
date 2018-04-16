use super::*;

pub mod classic;
pub mod sepconv;

pub use classic::*;
pub use sepconv::*;
pub use geometry::ImageGeometry;

/// A trait for networks that are able to create a prediction distribution
pub trait Predict<T> {
    fn predict(&self, input_data: &[T]) -> Vec<f32>;
}
