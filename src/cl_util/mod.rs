#![allow(dead_code)]
mod cl_types;
mod info;
pub mod standalone;

pub use self::cl_types::*;
pub use self::info::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::flags::*;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};
use std::fs;
use std::io::prelude::*;
use crate::util::*;

const PROFILING: bool = false;
const KERNEL_PATH_PREFIX: &str = "";

pub fn init<T>(
    kernel_srcs: &[&str],
    addt_cmplr_opts: &[&str],
    device_type: Option<DeviceType>,
) -> (Queue, Program, Context)
where
    T: ClVecTypeName,
{
    init_from_file::<T>(kernel_srcs, addt_cmplr_opts, device_type)
}

pub fn init_from_sources<T>(
    kernel_srcs: &[&str],
    addt_cmplr_opts: &[&str],
    device_type: Option<DeviceType>,
) -> (Queue, Program, Context)
where
    T: ClVecTypeName,
{
    // Select device
    let device = resolve_device(device_type);
    let platform = Platform::default();
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    let mut program_b = Program::builder();

    // Add default compiler options
    configure_program::<T, Device>(&mut program_b, device);

    // Additional compiler options
    for &opt in addt_cmplr_opts {
        program_b.cmplr_opt(opt);
    }

    // Input the kernel source files
    for &src in kernel_srcs {
        program_b.src(src);
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

pub fn init_from_file<T>(
    kernel_files: &[&str],
    addt_cmplr_opts: &[&str],
    device_type: Option<DeviceType>,
) -> (Queue, Program, Context)
where
    T: ClVecTypeName,
{
    let sources = kernel_files
        .iter()
        .map(|&fname| {
            let mut f = fs::File::open(&format!("{}{}", KERNEL_PATH_PREFIX, fname)).unwrap();
            let mut contents = String::new();
            f.read_to_string(&mut contents).unwrap();
            contents
        }).collect::<Vec<String>>();
    init_from_sources::<T>(
        &sources.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
        addt_cmplr_opts,
        device_type,
    )
}

pub fn configure_program<T, D>(program_b: &mut ProgramBuilder, devices: D)
where
    T: ClVecTypeName,
    D: Into<DeviceSpecifier>,
{
    program_b
        .devices(devices)
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
    // Create a host-accessible input buffer for writing the data into device memory
    let mut mem_map = buf.map().flags(flags::MAP_WRITE).len(buf.len()).enq()?;

    // Read the input into the input_buf as T
    for (idx, f) in data.into_iter().enumerate() {
        mem_map[idx] = *f;
    }

    mem_map.unmap().enq()?;
    Ok(())
}

pub fn resolve_device(device_type: Option<DeviceType>) -> Device {
    let platform = Platform::default();

    match device_type {
        Some(dt) => *Device::list(platform, Some(dt)).unwrap().first().unwrap(),
        None => Device::first(platform).unwrap(),
    }
}

pub enum DevicePreference {
    PreferGpu,
    RequireGpu,
    RequireCpu,
}

pub fn select_device(preference: DevicePreference) -> Device {
    let platform = Platform::default();

    use self::DevicePreference::*;
    match preference {
        // The first device listed by OpenCL usually seems  to be the GPU
        PreferGpu => Device::first(platform).unwrap(),
        RequireGpu => *Device::list(platform, Some(DeviceType::GPU))
            .unwrap()
            .first()
            .unwrap(),
        RequireCpu => *Device::list(platform, Some(DeviceType::CPU))
            .unwrap()
            .first()
            .unwrap(),
    }
}
