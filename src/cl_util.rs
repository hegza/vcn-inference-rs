use util::*;
use ocl;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};

/// Define which platform and device(s) to use. Create a context, queue, and program.
pub fn init() -> ocl::Result<(Queue, Program, Context)> {
    let kernel_sources = read_file("kernel/original_kernels.cl");

    // The platform is the thing that's provided by whatever vendor.
    let platform = Platform::default();
    let device_names: Vec<String> = Device::list(&platform, None)?
        .iter()
        .map(|&device| device.name())
        .collect();
    debug!("Available OpenCL devices: {:?}.", device_names);

    // TODO: currently, the first available device is selected, make configurable
    let device = Device::first(platform);
    info!("Using device \"{}\".", device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;

    // This is the 'deck of cards' that can be dealt to the 'players'
    let program = Program::builder()
        .devices(device)
        .src(kernel_sources)
        // HACK: Allow the .cl files to include the contents of files in the working directory
        .cmplr_opt("-I./input")
        .cmplr_opt("-cl-std=CL1.2")
        .build(&context)?;

    // TODO: make separate queues for all the associated devices
    // Create the queue for the default device
    let queue = Queue::new(
        &context,
        device,
        Some(flags::CommandQueueProperties::PROFILING_ENABLE),
    )?;

    Ok((queue, program, context))
}

pub fn create_buffer<T>(
    name: &'static str,
    length: usize,
    flags: flags::MemFlags,
    queue: &Queue,
) -> ocl::Result<Buffer<T>>
where
    T: OclPrm,
{
    debug!(
        "Create buffer with {} elements for {}. Flags: {:?}.",
        length, name, flags
    );
    Buffer::<T>::builder()
        .queue(queue.clone())
        .flags(flags)
        .dims(length)
        .build()
}

#[allow(dead_code)]
fn write_buf_to_file(filename: &str, buf: ocl::Buffer<f32>) -> ocl::Result<()> {
    unsafe {
        let mut mem_map = buf.map().flags(flags::MAP_READ).len(buf.len()).enq()?;
        write_file_f32s(filename, &mem_map);

        mem_map.unmap().enq()?;
    };
    Ok(())
}