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

lazy_static! {
    // This device is used as a GPU / accelerator for image-type calculations
    static ref PRIMARY_DEVICE: ocl::Device = {
        ocl::Device::first(ocl::Platform::default()).unwrap()
    };
}

use ocl::{Context, Device, Platform};
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
            cl::map_to_buf(&self.in_buf, input_data).unwrap();
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
    cl::configure_program::<T>(&mut program_b, &device);

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
