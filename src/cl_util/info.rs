use crate::util::*;
use ocl;
use ocl::builders::*;
use ocl::enums::*;
use ocl::{flags, Buffer, Context, Device, OclPrm, Platform, Program, Queue};

/// Returns the max work-group-size of the given OpenCL device (primary by default).
pub fn max_wgs(device: Option<&Device>) -> usize {
    let device = match device {
        Some(d) => *d,
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

pub fn describe_device(device: Device) -> ocl::Result<()> {
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
