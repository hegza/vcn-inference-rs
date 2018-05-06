#![allow(dead_code)]
mod cl_types;
mod info;

use util::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};
pub use self::cl_types::*;
pub use self::info::*;

const PROFILING: bool = false;
const KERNEL_PATH: &str = "src/cl";

/// Define which platform and device(s) to use. Create a context, queue, and program.
pub fn init<T>(
    kernel_files: &[&str],
    addt_cmplr_defs: &[(&str, i32)],
) -> ocl::Result<(Queue, Program, Context)>
where
    T: ClVecTypeName,
{
    let platform = ocl::Platform::default();
    let devices = ocl::Device::list_all(&platform).unwrap();
    let device_names: Vec<String> = devices
        .iter()
        .map(|&device| device.name().unwrap())
        .collect();
    debug!("Available OpenCL devices: {:?}.", device_names);

    let device = Device::first(platform)?;
    describe_device(&device)?;

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let mut program = Program::builder();
    program
        .devices(device)
        .cmplr_opt("-I./src/cl")
        .cmplr_opt("-cl-std=CL1.2")
        .cmplr_opt(format!("-D CL_PRIM={}", T::cl_type_name()))
        .cmplr_opt(format!("-D CL_PRIM2={}", T::cl_vec2_type_name()))
        .cmplr_opt(format!("-D CL_PRIM4={}", T::cl_vec2_type_name()))
        .cmplr_opt(format!("-D CL_PRIM8={}", T::cl_vec2_type_name()))
        .cmplr_opt(format!("-D CL_PRIM16={}", T::cl_vec2_type_name()));
    // Input the user-defined compiler definitions
    addt_cmplr_defs.iter().for_each(|&(name, val)| {
        program.cmplr_def(name, val);
    });
    // Input the kernel source files
    kernel_files.iter().for_each(|&src| {
        program.src_file(&format!("{}/{}", KERNEL_PATH, src));
    });
    let program = program.build(&context)?;

    // Create the queue for the default device
    let profile_flag = match PROFILING {
        true => Some(flags::CommandQueueProperties::PROFILING_ENABLE),
        false => None,
    };
    let queue = Queue::new(&context, device, profile_flag)?;

    Ok((queue, program, context))
}

pub fn create_buffer<T>(
    length: usize,
    flags: flags::MemFlags,
    queue: &Queue,
) -> ocl::Result<Buffer<T>>
where
    T: OclPrm,
{
    Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags)
        .len(length)
        .build()
}

pub unsafe fn read_buf<T: OclPrm>(buf: &Buffer<T>) -> ocl::Result<Vec<T>> {
    let mut mem_map = buf.map().flags(flags::MAP_READ).len(buf.len()).enq()?;
    let result = mem_map.to_vec();
    mem_map.unmap().enq()?;
    Ok(result)
}

pub unsafe fn map_to_buf<T: OclPrm>(buf: &Buffer<T>, data: &[T]) -> ocl::Result<()> {
    // Create a host-accessible input buffer for writing the image into device memory
    let mut mem_map = buf.map().flags(flags::MAP_WRITE).len(buf.len()).enq()?;

    // Read the input image into the input_buf as T
    for (idx, f) in data.into_iter().enumerate() {
        // TODO: the mapping could be done in float4's
        mem_map[idx] = *f;
    }

    mem_map.unmap().enq()?;
    Ok(())
}
