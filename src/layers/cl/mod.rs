pub use self::dense::*;
pub use self::conv::*;
use super::*;
use ocl::Result as OclResult;
use ocl::*;
use cl_util as cl;

pub trait ClLayer<T>: Layer
where
    T: Coeff,
{
    fn create_in_buf(&self, flags: flags::MemFlags, queue: &Queue) -> OclResult<Buffer<T>> {
        trace!(
            "ClLayer::create_in_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags
        );
        cl::create_buffer::<T>(self.num_in(), flags, queue)
    }
    fn create_out_buf(&self, flags: flags::MemFlags, queue: &Queue) -> OclResult<Buffer<T>> {
        trace!(
            "ClLayer::create_out_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags
        );
        cl::create_buffer::<T>(self.num_out(), flags, queue)
    }
    fn create_io_bufs(
        &self,
        in_flags: flags::MemFlags,
        out_flags: flags::MemFlags,
        queue: &Queue,
    ) -> OclResult<(Buffer<T>, Buffer<T>)> {
        Ok((
            self.create_in_buf(in_flags, queue)?,
            self.create_out_buf(out_flags, queue)?,
        ))
    }
}

pub trait ClWeightedLayer<T>: WeightedLayer<T> + ClLayer<T>
where
    T: Coeff,
{
    /// Create a read-only buffer on-device for weights
    fn create_wgts_buf(&self, queue: &Queue) -> OclResult<Buffer<T>> {
        trace!(
            "ClLayer::create_wgts_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags::MEM_READ_ONLY
        );
        cl::create_buffer::<T>(self.num_weights(), flags::MEM_READ_ONLY, queue)
    }
}

impl<T> ClWeightedLayer<T> for DenseLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for DenseLayer<T>
where
    T: Coeff,
{
}

impl<T> ClWeightedLayer<T> for ConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for ConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClWeightedLayer<T> for SepconvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for SepconvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for MaxpoolLayer
where
    T: Coeff,
{
}
