use super::*;

pub mod cl_context;
pub mod classic;
pub mod sepconv;
pub mod sparse;

/// A trait for networks that are able to create a prediction distribution
pub trait Predict<T> {
    fn predict(&self, input_data: &[T]) -> Vec<f32>;
}

lazy_static! {
    // This device is used as a GPU / accelerator for image-type calculations
    static ref PRIMARY_DEVICE: ocl::Device = {
        ocl::Device::first(ocl::Platform::default()).unwrap()
    };
}
