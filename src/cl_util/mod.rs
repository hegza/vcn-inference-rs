#![allow(dead_code)]
mod cl_types;
mod info;

use util::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};
use ocl::flags::*;
pub use self::cl_types::*;
pub use self::info::*;

const PROFILING: bool = false;
const KERNEL_PATH: &str = "src/cl";

pub fn init<T>(
    kernel_srcs: &[&str],
    addt_cmplr_opts: &[&str],
    device_type: Option<DeviceType>,
) -> (Queue, Program, Context)
where
    T: ClVecTypeName,
{
    // Select device
    let platform = Platform::default();
    let device = match device_type {
        Some(dt) => *Device::list(platform, Some(dt)).unwrap().first().unwrap(),
        None => Device::first(platform).unwrap(),
    };
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    let mut program_b = Program::builder();
    // Add default compiler options
    configure_program::<T>(&mut program_b, &device);

    // Additional compiler options
    for &opt in addt_cmplr_opts {
        program_b.cmplr_opt(opt);
    }

    // Input the kernel source files
    for src in kernel_srcs {
        program_b.src_file(&format!("src/cl/{}", src));
    }

    let program = program_b.build(&context).unwrap();

    // Create the queue for the default device
    let profile_flag = match PROFILING {
        true => Some(flags::CommandQueueProperties::PROFILING_ENABLE),
        false => None,
    };
    let queue = Queue::new(&context, device, profile_flag).unwrap();
    (queue, program, context)
}

pub fn configure_program<T>(program_b: &mut ProgramBuilder, device: &Device)
where
    T: ClVecTypeName,
{
    program_b
        .devices(device.clone())
        .cmplr_opt("-I./src/cl")
        .cmplr_opt("-cl-std=CL1.2")
        .cmplr_opt(format!("-D CL_PRIM={}", T::cl_type_name()))
        .cmplr_opt(format!("-D CL_PRIM2={}", T::cl_vec2_type_name()))
        .cmplr_opt(format!("-D CL_PRIM4={}", T::cl_vec4_type_name()))
        .cmplr_opt(format!("-D CL_PRIM8={}", T::cl_vec8_type_name()))
        .cmplr_opt(format!("-D CL_PRIM16={}", T::cl_vec16_type_name()));
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
