pub use self::dense::*;
pub use self::conv::*;
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

pub trait ClUnweightedLayer<T>: ClLayer<T>
where
    T: Coeff,
{
    fn create_kernel(
        &self,
        kernel_func: &str,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
        lws: LocalWorkSizePolicy,
        program: &Program,
        queue: &Queue,
    ) -> Kernel {
        use self::LocalWorkSizePolicy::*;
        let lws = match lws {
            Specify(dims) => Some(dims),
            Infer { dev_max_wgs } => Some(self.lws_hint(dev_max_wgs)),
            UseDefault => None,
        };
        let gws = self.gws_hint();

        debug!(
            "Create unweighted kernel {}, gws: {:?} = {}.",
            kernel_func,
            gws,
            gws.to_len()
        );

        let mut builder = KernelBuilder::new();
        builder
            .program(program)
            .name(kernel_func)
            .queue(queue.clone())
            .global_work_size(gws)
            .arg(in_buf)
            .arg(out_buf);
        if let Some(lws) = lws {
            builder.local_work_size(lws);
            debug!("\tLocal-work-size set as: {:?} = {}.", lws, lws.to_len());
        }
        builder.build().unwrap()
    }
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
    fn create_kernel(
        &self,
        kernel_func: &str,
        in_buf: &Buffer<T>,
        out_buf: &Buffer<T>,
        wgts_buf: &Buffer<T>,
        lws: LocalWorkSizePolicy,
        program: &Program,
        queue: &Queue,
    ) -> Kernel {
        use self::LocalWorkSizePolicy::*;
        let lws = match lws {
            Specify(dims) => Some(dims),
            Infer { dev_max_wgs } => Some(self.lws_hint(dev_max_wgs)),
            UseDefault => None,
        };
        let gws = self.gws_hint();

        debug!(
            "Create weighted kernel {}, gws: {:?} = {}.",
            kernel_func,
            gws,
            gws.to_len()
        );

        let mut builder = KernelBuilder::new();
        builder
            .program(program)
            .name(kernel_func)
            .queue(queue.clone())
            .global_work_size(gws)
            .arg(in_buf)
            .arg(out_buf)
            .arg(wgts_buf);
        if let Some(lws) = lws {
            builder.local_work_size(lws);
            debug!("\tLocal-work-size set as: {:?} = {}.", lws, lws.to_len());
        }
        builder.build().unwrap()
    }

    // TODO: generalized version over N-layers
    fn impl_standalone(
        &self,
        kernel_srcs: &[&str],
        kernel_name: &str,
        addt_cmplr_opts: Option<&[&str]>,
        device_type: Option<DeviceType>,
        lws_policy: LocalWorkSizePolicy,
    ) -> LayerImpl<T> {
        // Select device
        let platform = Platform::default();
        let device = match device_type {
            Some(dt) => Device::list(platform, Some(dt))
                .unwrap()
                .first()
                .unwrap()
                .clone(),
            None => Device::first(platform).unwrap(),
        };
        let context = Context::builder()
            .platform(platform)
            .devices(device)
            .build()
            .unwrap();
        let queue = Queue::new(&context, device, None).unwrap();

        let mut program_b = Program::builder();
        // Add default compiler options
        cl_util::configure_program::<T>(&mut program_b, &device);

        // Additional compiler options
        if let Some(opts) = addt_cmplr_opts {
            for &opt in opts {
                program_b.cmplr_opt(opt);
            }
        }

        // Input the kernel source files
        for &src in kernel_srcs {
            program_b.src_file(&format!("src/cl/{}", src));
        }

        let program = program_b.build(&context).unwrap();

        // Create buffers
        let wgts_buf = self.create_wgts_buf(&queue);
        let (in_buf, out_buf) = self.create_io_bufs(
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        );

        let kernel = self.create_kernel(
            kernel_name,
            &in_buf,
            &out_buf,
            &wgts_buf,
            lws_policy,
            &program,
            &queue,
        );

        queue.finish().unwrap();

        LayerImpl {
            in_buf,
            out_buf,
            kernel,
            queue,
            program,
            context,
        }
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

pub struct ClKernelChainBuilder<'p, 'b, T>
where
    T: Coeff,
{
    program: &'p Program,
    queue: Queue,
    layer_idx: usize,
    wgts_idx: usize,
    io_bufs: &'b [Buffer<T>],
    wgts_bufs: &'b [Buffer<T>],
}

impl<'p, 'b, T> ClKernelChainBuilder<'p, 'b, T>
where
    T: Coeff,
{
    pub fn new(
        io_bufs: &'b [Buffer<T>],
        wgts_bufs: &'b [Buffer<T>],
        program: &'p Program,
        queue: Queue,
    ) -> ClKernelChainBuilder<'p, 'b, T> {
        ClKernelChainBuilder::<'p, 'b, T> {
            program,
            queue,
            layer_idx: 0,
            wgts_idx: 0,
            io_bufs: io_bufs.clone(),
            wgts_bufs: wgts_bufs.clone(),
        }
    }
    pub fn build_io_kernel(
        &mut self,
        layer: &ClUnweightedLayer<T>,
        kernel_func: &str,
        lws_policy: LocalWorkSizePolicy,
    ) -> Kernel {
        let kernel = layer.create_kernel(
            kernel_func,
            &self.io_bufs[self.layer_idx],     // In
            &self.io_bufs[self.layer_idx + 1], // Out
            lws_policy,
            &self.program,
            &self.queue,
        );
        self.layer_idx += 1;
        kernel
    }
    pub fn build_iow_kernel(
        &mut self,
        layer: &ClWeightedLayer<T>,
        kernel_func: &str,
        lws_policy: LocalWorkSizePolicy,
    ) -> Kernel {
        let kernel = layer.create_kernel(
            kernel_func,
            &self.io_bufs[self.layer_idx],     // In
            &self.io_bufs[self.layer_idx + 1], // Out
            &self.wgts_bufs[self.wgts_idx],    // Weights
            lws_policy,
            &self.program,
            &self.queue,
        );
        self.layer_idx += 1;
        self.wgts_idx += 1;
        kernel
    }
}

// A standalone layer
pub struct LayerImpl<T>
where
    T: Coeff,
{
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
