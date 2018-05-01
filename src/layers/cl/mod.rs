pub use self::dense::*;
pub use self::conv::*;
use super::*;
use ocl::Result as OclResult;
use ocl::*;
use cl_util as cl;
use ocl::builders::KernelBuilder;

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

pub trait ClWeightedLayer<T>: WeightedLayer<T> + ClLayer<T>
where
    T: Coeff,
{
    /// Create a read-only buffer on-device for weights and write the weights
    fn create_wgts_buf(&self, queue: &Queue) -> Buffer<T> {
        trace!(
            "ClLayer::create_wgts_buf with {} elements. Flags: {:?}.",
            self.num_in(),
            flags::MEM_READ_ONLY
        );
        let buf = cl::create_buffer::<T>(self.num_weights(), flags::MEM_READ_ONLY, queue).unwrap();
        buf.write(self.weights()).enq().unwrap();
        buf
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

// TODO: this could be split into a multi-tier builder
pub struct ClKernelBuilder<'p> {
    program: &'p Program,
    queue: Queue,
}

impl<'p> ClKernelBuilder<'p> {
    pub fn new(program: &Program, queue: Queue) -> ClKernelBuilder {
        ClKernelBuilder { program, queue }
    }
    pub fn build_io_kernel<T>(
        &self,
        kernel_name: &str,
        global_work_size: SpatialDims,
        local_work_size: SpatialDims,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
    ) -> Kernel
    where
        T: OclPrm,
    {
        self.kernel_builder(kernel_name, global_work_size, local_work_size)
            .arg(in_buf)
            .arg(out_buf)
            .build()
            .unwrap()
    }
    pub fn build_iow_kernel<T>(
        &self,
        kernel_name: &str,
        global_work_size: SpatialDims,
        local_work_size: SpatialDims,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
        wgts_buf: &Buffer<T>,
    ) -> Kernel
    where
        T: OclPrm,
    {
        self.kernel_builder(kernel_name, global_work_size, local_work_size)
            .arg(in_buf)
            .arg(out_buf)
            .arg(wgts_buf)
            .build()
            .unwrap()
    }
    fn kernel_builder(
        &self,
        kernel_name: &str,
        global_work_size: SpatialDims,
        local_work_size: SpatialDims,
    ) -> KernelBuilder {
        debug!(
            "Create kernel {}, gws: {:?} = {}, lws: {:?} = {}.",
            kernel_name,
            global_work_size,
            global_work_size.to_len(),
            local_work_size,
            local_work_size.to_len()
        );

        // Last minute check that there are no work-group overshoots, do not panic to get more
        // diagnostic info from OpenCL.
        let max_wgs = cl::max_wgs(None);
        if max_wgs < local_work_size.to_len() {
            error!(
                "local work size is larger than maximum work-group-size: {} > {}",
                local_work_size.to_len(),
                max_wgs
            );
        }

        let mut builder = KernelBuilder::new();
        builder
            .program(self.program)
            .name(kernel_name)
            .queue(self.queue.clone())
            .global_work_size(global_work_size)
            .local_work_size(local_work_size);
        builder
    }
}
