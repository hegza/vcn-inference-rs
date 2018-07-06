use super::*;

const ONE: f32 = 1f32;
const ZERO: f32 = 0f32;
const MINUS_ONE: f32 = -1f32;

// `cargo test --feature test_quantize` to run this test
#[cfg_attr(not(feature = "test_quantize"), ignore)]
#[test]
fn f32_quantizes_into_u8() {
    // Test regular quantization
    let params = QuantizationParams::<f32, u8>::choose(-1f32, 1f32);

    // 1f32 quantizes into u8::MAX or -1
    let q_max: u8 = ONE.quantize(&params);
    assert!(q_max == 255 || q_max == 254);

    // 0f32 quantizes into a value at about the middle of the range
    let q_middle: u8 = ZERO.quantize(&params);
    assert!(q_middle == 127 || q_middle == 128);

    // -1f32 quantizes into u8::MIN (0) or +1
    let q_zero: u8 = MINUS_ONE.quantize(&params);
    assert!(q_zero == 0 || q_zero == 1);

    // Verify that other kinds of quantizations work as well
    let params2 = QuantizationParams::<f32, u8>::choose(0f32, 2f32);

    let q_custom_max: u8 = 2f32.quantize(&params2);
    assert!(q_custom_max == 255 || q_custom_max == 254);

    let q_custom_min: u8 = 0f32.quantize(&params2);
    assert!(q_custom_min == 0 || q_custom_min == 1);
}

// `cargo test --feature test_quantize` to run this test
#[cfg_attr(not(feature = "test_quantize"), ignore)]
#[test]
fn f32_quantizes_into_i8() {
    // Test regular quantization
    let params = QuantizationParams::<f32, i8>::choose(-1f32, 1f32);

    let q_max: i8 = ONE.quantize(&params);
    assert_eq!(q_max, 127);

    let q_middle: i8 = ZERO.quantize(&params);
    assert_eq!(q_middle, 0);

    let q_min: i8 = MINUS_ONE.quantize(&params);
    assert_eq!(q_min, -127);

    // Verify that other kinds of quantizations work as well
    let params2 = QuantizationParams::<f32, i8>::choose(0f32, 2f32);

    let q_custom_max: i8 = 2f32.quantize(&params2);
    assert_eq!(q_custom_max, 127);

    let q_custom_min: i8 = 0f32.quantize(&params2);
    assert!(q_custom_min == -127 || q_custom_min == -128);
}

// TODO: test unquantization

/*
#[test]
fn mtx_mul_normint_works() {
    let a = vec![255; 1];
    let b = vec![255; 1];
    let c = mtx_mul_normint(&a, &b, 1, 1);
    assert_eq!(c, vec![255; 1]);
}
*/
