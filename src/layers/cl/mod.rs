pub mod builders;
pub mod weighted_layer;
pub mod unweighted_layer;

pub use self::dense::*;
pub use self::conv::*;
pub use self::builders::*;
pub use self::weighted_layer::*;
pub use self::unweighted_layer::*;
use super::*;
use ocl::Result as OclResult;
use ocl::*;
use cl_util as cl;
use ocl::builders::KernelBuilder;

pub enum LocalWorkSizePolicy {
    Infer { dev_max_wgs: usize },
    Specify(SpatialDims),
    UseDefault,
}

pub trait ClLayer<T>: Layer
where
    T: Coeff,
{
    fn create_in_buf(&self, flags: flags::MemFlags, queue: &Queue) -> Buffer<T> {
        trace!(
            "ClLayer::create_in_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags
        );
        cl::create_buffer::<T>(self.num_in(), flags, queue).unwrap()
    }
    fn create_out_buf(&self, flags: flags::MemFlags, queue: &Queue) -> Buffer<T> {
        trace!(
            "ClLayer::create_out_buf with {} elements. Flags: {:?}.",
            self.num_out(),
            flags
        );
        cl::create_buffer::<T>(self.num_out(), flags, queue).unwrap()
    }
    fn create_io_bufs(
        &self,
        in_flags: flags::MemFlags,
        out_flags: flags::MemFlags,
        queue: &Queue,
    ) -> (Buffer<T>, Buffer<T>) {
        (
            self.create_in_buf(in_flags, queue),
            self.create_out_buf(out_flags, queue),
        )
    }
}

// Creates a chain of buffers of which the first one is read-only + alloc host-ptr, the ones in between are read-write, and the last one is write-only + alloc host-ptr
pub fn create_buffer_chain<T>(layers: &[&ClLayer<T>], queue: &Queue) -> Vec<Buffer<T>>
where
    T: Coeff,
{
    let mut bufs = vec![];

    // Input buffer
    let in_buf = layers[0].create_in_buf(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR, queue);
    bufs.push(in_buf);

    // Buffers in-between layers 1..N-1
    for idx in 1..layers.len() {
        let buf = layers[idx].create_in_buf(flags::MEM_READ_WRITE, queue);
        bufs.push(buf);
    }

    // Output buffer
    let out_buf = layers
        .iter()
        .last()
        .unwrap()
        .create_out_buf(flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR, queue);
    bufs.push(out_buf);

    bufs
}

pub fn create_weights_bufs<T>(layers: &[&ClWeightedLayer<T>], queue: &Queue) -> Vec<Buffer<T>>
where
    T: Coeff,
{
    layers
        .iter()
        .map(|&layer| layer.create_wgts_buf(queue))
        .collect()
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

impl<T> ClWeightedLayer<T> for HConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for HConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClWeightedLayer<T> for VConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for VConvLayer<T>
where
    T: Coeff,
{
}

impl<T> ClLayer<T> for MaxpoolLayer
where
    T: Coeff,
{
}

impl<T> ClUnweightedLayer<T> for MaxpoolLayer
where
    T: Coeff,
{
}

// A standalone layer
pub struct LayerImpl<T>
where
    T: Coeff,
{
    // TODO: reduce visibility, attempt to remove unused variables
    pub in_buf: Buffer<T>,
    pub out_buf: Buffer<T>,
    pub kernel: Kernel,
    pub queue: Queue,
    pub program: Program,
    pub context: Context,
}

impl<T> LayerImpl<T>
where
    T: Coeff,
{
    pub fn map_input(&self, input_data: &[T]) {
        unsafe {
            cl_util::map_to_buf(&self.in_buf, input_data).unwrap();
        }
        self.queue.finish().unwrap();
    }
}
