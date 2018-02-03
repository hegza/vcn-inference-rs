mod conv;
mod dense;

pub use self::conv::*;
pub use self::dense::*;

/// A layer of a convolutive neural network.
pub trait Layer<T> {
    fn weights(&self) -> &Vec<T>;
    fn num_weights(&self) -> usize {
        self.weights().len()
    }
    /// Gets the number of elements in the output shape
    fn num_out(&self) -> usize;
    /// Gets the number of elements in the input shape
    fn num_in(&self) -> usize;
}

/// The data contained in any given CNN-layer.
pub struct LayerData<T> {
    weights: Vec<T>,
}
