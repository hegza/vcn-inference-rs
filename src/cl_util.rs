#![allow(dead_code)]
use util::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};

const PROFILING: bool = true;
const KERNEL_PATH: &str = "src/cl";

/// Define which platform and device(s) to use. Create a context, queue, and program.
pub fn init(
    kernel_files: &[&str],
    addt_cmplr_defs: &[(&str, i32)],
) -> ocl::Result<(Queue, Program, Context)> {
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
        .cmplr_opt("-cl-std=CL1.2");
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

/// Returns the max work-group-size of the primary OpenCL device.
pub fn max_wgs(device: Option<Device>) -> usize {
    let device = match device {
        Some(d) => d,
        None => {
            let platform = Platform::default();
            Device::first(platform).unwrap()
        }
    };

    match device.info(DeviceInfo::MaxWorkGroupSize).unwrap() {
        DeviceInfoResult::MaxWorkGroupSize(max_wgs) => max_wgs,
        e => panic!("ocl library returned invalid enum {:?}", e),
    }
}

fn describe_device(device: &Device) -> ocl::Result<()> {
    let device_type = match device.info(DeviceInfo::Type)? {
        DeviceInfoResult::Type(t) => match t {
            flags::DeviceType::CPU => "CPU",
            flags::DeviceType::GPU => "GPU",
            _ => "unknown device type",
        },
        _ => panic!("ocl did not return the expected type"),
    };
    info!("Using {} \"{}\".", device_type, device.name()?);
    debug!(
        "Maximum work-item-sizes: {}",
        device.info(DeviceInfo::MaxWorkItemSizes)?
    );
    Ok(())
}
