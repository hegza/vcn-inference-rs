mod conv;
mod dense;
mod sepconv;
mod maxpool;
mod cl;
mod host;

pub use self::conv::*;
pub use self::dense::*;
pub use self::sepconv::*;
pub use self::maxpool::*;
pub use self::cl::*;
pub use self::host::*;
use geometry::*;
use std::ops::Deref;
use ocl::*;
use util::*;
use num_traits::{Float, NumAssign};
use math::GenericOps;
use cl_util;

pub trait Coeff: NumAssign + GenericOps + OclPrm + cl_util::ClVecTypeName {}
pub trait CoeffFloat: Float + Coeff {}

/// Describes a layer of a convolutive neural network.
pub trait Layer {
    /// Gets the number of elements in the input shape
    fn num_in(&self) -> usize;
    /// Gets the number of elements in the output shape
    fn num_out(&self) -> usize;
    /// The probable optimal global work-group-size-shape of the matching kernel
    fn gws_hint(&self) -> SpatialDims;
    // The probable optimal local work-group-size-shape of the matching kernel
    fn lws_hint(&self, device_max_wgs: usize) -> SpatialDims;
    fn name(&self) -> &'static str;
}

pub trait WeightedLayer<T>: Layer {
    fn weights(&self) -> &Vec<T>;
    fn num_weights(&self) -> usize {
        self.weights().len()
    }
}

impl Coeff for f32 {}
impl CoeffFloat for f32 {}
impl Coeff for i8 {}

use flags::DeviceType;

// A standalone layer
pub struct OclLayer<T>
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

impl<T> OclLayer<T>
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

// TODO: generalized version over N-layers
// Allow generating any kind of an OpenCL layer implementation (except non-weighted ones atm).
pub fn impl_ocl_layer<L, T>(
    layer: &L,
    kernel_srcs: &[&str],
    kernel_name: &str,
    addt_cmplr_opts: Option<&[&str]>,
    device_type: Option<DeviceType>,
    lws_policy: LocalWorkSizePolicy,
) -> OclLayer<T>
where
    T: Coeff,
    // TODO: This constraint is unfortunate, maybe need to refactor some time
    L: ClWeightedLayer<T>,
{
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
    let wgts_buf = layer.create_wgts_buf(&queue);
    let (in_buf, out_buf) = layer.create_io_bufs(
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    );

    let kernel = layer.create_kernel(
        kernel_name,
        &in_buf,
        &out_buf,
        &wgts_buf,
        lws_policy,
        &program,
        &queue,
    );

    queue.finish().unwrap();

    OclLayer {
        in_buf,
        out_buf,
        kernel,
        queue,
        program,
        context,
    }
}
