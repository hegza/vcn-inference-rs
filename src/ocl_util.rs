use util::*;
use ocl;
use ocl::{flags, Buffer, Context, Device, MemFlags, OclPrm, Platform, Program, Queue, SpatialDims};

/// Define which platform and device(s) to use. Create a context, queue, and program.
pub fn init() -> ocl::Result<(Queue, Program, Context)> {
    let kernel_sources = read_file("kernel/original_kernels.cl");

    // The platform is the thing that's provided by whatever vendor.
    let platform = Platform::default();
    let device_names: Vec<String> = Device::list(&platform, None)?
        .iter()
        .map(|&device| device.name())
        .collect();
    println!("Available OpenCL devices: {:?}.", device_names);

    // TODO: currently, the first available device is selected, make configurable
    let device = Device::first(platform);
    println!("Using device \"{}\".", device.name());

    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;

    // This is the 'deck of cards' that can be dealt to the 'players'
    let program = Program::builder()
        .devices(device)
        .src(kernel_sources)
        // HACK: Allow the .cl files to include the contents of files in the working directory
        .cmplr_opt("-I.")
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
