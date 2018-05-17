use super::*;

#[test]
fn f32_quantizes_to_i8_correctly() {
    let one = 1f32;
    let zero = 0f32;
    let minus_one = -1f32;

    let q_one: i8 = one.quantize();
    let q_zero: i8 = zero.quantize();
    let q_minus_one: i8 = minus_one.quantize();

    assert_eq!(q_one, 127);
    assert_eq!(q_zero, 0);
    assert_eq!(q_minus_one, -127);
}
