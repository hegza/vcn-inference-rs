#![allow(dead_code)]
use util::*;
use ocl;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};
use ocl::enums::*;

const KERNEL_PATH: &str = "src/cl";

/// Define which platform and device(s) to use. Create a context, queue, and program.
pub fn init(kernel_files: &[&str]) -> ocl::Result<(Queue, Program, Context)> {
    // The platform is the thing that's provided by whatever vendor.
    let platform = Platform::default();
    let devices = Device::list_all(&platform)?;
    let device_names: Vec<String> = devices
        .iter()
        .map(|&device| device.name().unwrap())
        .collect();
    debug!("Available OpenCL devices: {:?}.", device_names);

    let device = Device::first(platform)?;
    info!(
        "Using {} \"{}\".",
        match device.info(DeviceInfo::Type).unwrap() {
            DeviceInfoResult::Type(t) => match t {
                flags::DeviceType::CPU => "CPU",
                flags::DeviceType::GPU => "GPU",
                _ => "unknown device type",
            },
            _ => panic!("ocl did not return the expected type"),
        },
        device.name().unwrap()
    );

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    let mut program = Program::builder();
    program
        .devices(device)
        .cmplr_opt("-I./src/cl")
        .cmplr_opt("-cl-std=CL1.2");
    // Input the kernel source files
    kernel_files.iter().for_each(|&src| {
        program.src_file(&format!("{}/{}", KERNEL_PATH, src));
    });
    let program = program.build(&context)?;

    // Create the queue for the default device
    let queue = Queue::new(
        &context,
        device,
        Some(flags::CommandQueueProperties::PROFILING_ENABLE),
    )?;

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
