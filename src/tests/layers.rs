use super::*;
use crate::geometry::*;
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
    let device = cl_util::select_device(cl_util::DevicePreference::PreferGpu);
    let dev_max_wgs = cl_util::max_wgs(Some(&device));

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

use crate::network::sparse::WEIGHTS_DIR;
use ndarray::Array;
use ocl::flags::*;
#[test]
fn conv2d_cl_returns_tf_baseline() {
    // Load image with padding and in (height, width, channels)-order
    let padded_input: Vec<f32> = {
        let raw_input: Vec<f32> = load_jpeg_hwc("input/baseline/sparse-f32/in.jpg");
        let mut padded = Array::zeros((100, 100, 3));
        padded
            .slice_mut(s![2..-2, 2..-2, ..])
            .assign(&Array::from_shape_vec((96, 96, 3), raw_input).unwrap());

        padded.into_iter().cloned().collect::<Vec<f32>>()
    };

    // Load filters in (out channels, height, width, channels)-order
    let filters = {
        let raw_filters = Array::from_shape_vec(
            (32, 3, 5, 5),
            f32::read_csv(&format!("{}/{}", WEIGHTS_DIR, "conv1-f32-dcwh.csv")),
        )
        .unwrap();
        let dhwc_order = raw_filters.permuted_axes((0, 3, 2, 1));

        dhwc_order.into_iter().cloned().collect::<Vec<f32>>()
    };

    // Output is in (height, width, channels)-order
    let output = {
        let (q, prog, _ctx) = cl_util::init_from_sources::<f32>(
            &[&String::from_utf8_lossy(include_bytes!(
                "../cl/conv2d_mxp.cl"
            ))],
            &[
                "-D FILTER_WIDTH=5",
                "-D FILTER_HEIGHT=5",
                "-D IN_WIDTH=96",
                "-D IN_HEIGHT=96",
                "-D IN_CHANNELS=3",
                "-D OUT_CHANNELS=32",
            ],
            None,
        );
        let conv_cl = cl_util::standalone::BinOp::new(
            "conv2d_mxp",
            &filters,
            &padded_input,
            96 * 96 * 32,
            SpatialDims::Three(96, 96, 32),
            q.clone(),
            &prog,
        );
        relu(conv_cl.result())
    };

    // Load model out in (height, width, channels)-order
    let model_output = {
        let raw = Array::from_shape_vec(
            (32, 96, 96),
            f32::read_csv("input/baseline/sparse-f32/fm1-cwh.csv"),
        )
        .unwrap();
        let hwc_order = raw.permuted_axes((2, 1, 0));
        hwc_order.into_iter().cloned().collect::<Vec<f32>>()
    };

    verify(&output, &model_output, F32_GEMM_MAX_EPSILON);
}

/// Tests that when input is convoluted with filter, it produces output
/// input:
///     00, 00, 00, 00, 00
///     00, 11, 12, 13, 00
///     00, 21, 22, 23, 00
///     00, 31, 32, 33, 00
///     00, 00, 00, 00, 00
/// filter:
///     11, 12, 13
///     21, 22, 23
///     31, 32, 33
/// output:
///     988, 1580, 1124,
///     2372, 3750, 2636,
///     2708, 4220, 2924,
#[test]
fn conv2d_cl_convolves_simple() {
    let padded_input: Vec<f32> = {
        let raw_input =
            util::csv_str_to_vec(String::from_utf8_lossy(include_bytes!("in/rc_3x3.csv")));
        let mut padded = Array::zeros((5, 5));
        padded
            .slice_mut(s![1..-1, 1..-1])
            .assign(&Array::from_shape_vec((3, 3), raw_input).unwrap());
        padded.into_iter().cloned().collect::<Vec<f32>>()
    };

    let filter: Vec<f32> =
        util::csv_str_to_vec(String::from_utf8_lossy(include_bytes!("in/rc_3x3.csv")));

    let (q, prog, _ctx) = cl_util::init_from_sources::<f32>(
        &[&String::from_utf8_lossy(include_bytes!(
            "../cl/conv2d_mxp.cl"
        ))],
        &[
            "-D FILTER_WIDTH=3",
            "-D FILTER_HEIGHT=3",
            "-D IN_WIDTH=3",
            "-D IN_HEIGHT=3",
            "-D IN_CHANNELS=1",
            "-D OUT_CHANNELS=1",
            "-D NXCORR",
        ],
        None,
    );
    let conv_cl = cl_util::standalone::BinOp::new(
        "conv2d_mxp",
        &filter,
        &padded_input,
        3 * 3,
        SpatialDims::Three(3, 3, 1),
        q.clone(),
        &prog,
    );
    let output = Array::from_shape_vec((3, 3), conv_cl.result().clone()).unwrap();

    let model_output = Array::from_shape_vec(
        (3, 3),
        vec![
            988f32, 1580f32, 1124f32, 2372f32, 3750f32, 2636f32, 2708f32, 4220f32, 2924f32,
        ],
    )
    .unwrap();

    verify(
        &output.iter().cloned().collect::<Vec<f32>>(),
        &model_output.iter().cloned().collect::<Vec<f32>>(),
        RESULT_MARGIN,
    );
}
