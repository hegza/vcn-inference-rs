use super::*;
use geometry::*;
use ocl::flags::DeviceType;
use ocl::{flags, Context, Device, Kernel, Platform, Program, SpatialDims};

// Test that Maxpool + ReLU produces the correct output
#[test]
fn mxp_returns_baseline() {
    let in_img = f32::read_csv("src/tests/in/img-4x4_mono-norm.csv");
    const SIDE: usize = 4;
    let in_shape = ImageGeometry::new(SIDE, 1);

    // Make sure the input image matches with the assumed input shape
    assert_eq!(in_shape.num_elems(), in_img.len());

    let mxp = MaxpoolLayer::new(&in_shape, 2);

    // Implement mxp on GPU if possible
    let dev_max_wgs = cl_util::max_wgs(None);
    let cl_impl = mxp.impl_standalone(
        &["src/cl/test/mxp.cl"],
        "max_pool",
        &[],
        None,
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    let gpu_out = cl_impl.run_with_input(&in_img);

    // Run maxpool on CPU
    let cpu_out = mxp.compute(&in_img);

    // Verify match between confirmed correct output and GPU and CPU outputs
    let correct = f32::read_csv("src/tests/out/img-4x4_mono-norm-mxp2.csv");
    assert!(is_within_margin(&gpu_out, &correct, RESULT_MARGIN));
    assert!(is_within_margin(&cpu_out, &correct, RESULT_MARGIN));
}
