use super::*;

#[test]
fn f32_quantizes_to_i8_correctly() {
    let one = 1f32;
    let zero = 0f32;
    let minus_one = -1f32;

    let q_one: i8 = one.quantize(-1f32, 1f32);
    let q_zero: i8 = zero.quantize(-1f32, 1f32);
    let q_minus_one: i8 = minus_one.quantize(-1f32, 1f32);
    let q_custom_max: i8 = 2f32.quantize(0f32, 2f32);
    let q_custom_min: i8 = 2f32.quantize(2f32, 4f32);
    let q_custom_center: i8 = 16.5f32.quantize(16f32, 17f32);

    assert_eq!(q_one, 127);
    assert_eq!(q_zero, 0);
    assert_eq!(q_minus_one, -128);
    assert_eq!(q_custom_max, 127);
    assert_eq!(q_custom_min, -128);
    assert_eq!(q_custom_center, 0);
}
