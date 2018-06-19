// Quantization based on https://github.com/google/gemmlowp
// Original work licensed with Apache License 2.0
// TODO: refactor comments to follow my conventions (passive voice)

use std::ops::{Mul, Sub};
use num_traits::Zero;
use num_traits::bounds::Bounded;

// HACK: implementations currently assume that T == u8

// A structure to hold quantization parameters 'scale' and 'zero_point'
// as discussed in doc/quantization.md. As explained there, the meaning
// of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
pub struct QuantizationParams<S, T>
where
    S: From<T>,
    T: Zero + Bounded,
{
    pub scale: S,
    pub zero_point: T,
}

pub trait QuantizeInto<T>: From<T>
where
    T: Zero + Bounded,
{
    fn quantize(self, params: &QuantizationParams<Self, T>) -> T;
}

// TODO: combine implementations as generic

// Given the min and max values of a float array, return
// reasonable quantization parameters to use for this array.
impl QuantizationParams<f32, u8> {
    pub fn choose(min: f32, max: f32) -> QuantizationParams<f32, u8> {
        // We extend the [min, max] interval to ensure that it contains 0.
        // Otherwise, we would not meet the requirement that 0 be an exactly
        // representable value.
        let min = min.min(f32::from(u8::zero()));
        let max = max.max(f32::from(u8::zero()));

        // the min and max quantized values, as floating-point values
        let qmin = f32::from(u8::min_value());
        let qmax = f32::from(u8::max_value());

        // First determine the scale.
        let scale = (max - min) as f64 / (qmax - qmin) as f64;

        // Zero-point computation.
        // First the initial floating-point computation. The zero-point can be
        // determined from solving an affine equation for any known pair
        // (real value, corresponding quantized value).
        // We know two such pairs: (rmin, qmin) and (rmax, qmax).
        // Let's use the first one here.
        let initial_zero_point = qmin as f64 - min as f64 / scale;

        // Now we need to nudge the zero point to be an integer
        // (our zero points are integer, and this is motivated by the requirement
        // to be able to represent the real value "0" exactly as a quantized value,
        // which is required in multiple places, for example in Im2col with SAME
        // padding).
        let nudged_zero_point: u8;
        // Note: qmin and qmax are guaranteed to be within range of T
        if initial_zero_point < qmin as f64 {
            nudged_zero_point = qmin as u8;
        } else if initial_zero_point > qmax as f64 {
            nudged_zero_point = qmax as u8;
        } else {
            // Note: I chose to use .floor() here instead of .round() of the original
            // implementation, because that brings the minimum value to 0 in the common case of
            // [0, 255].
            nudged_zero_point = (initial_zero_point as f32).floor() as u8;
        }

        QuantizationParams {
            scale: scale as f32,
            zero_point: nudged_zero_point,
        }
    }
}

impl QuantizationParams<f32, i8> {
    pub fn choose(min: f32, max: f32) -> QuantizationParams<f32, i8> {
        // We extend the [min, max] interval to ensure that it contains 0.
        // Otherwise, we would not meet the requirement that 0 be an exactly
        // representable value.
        let min = min.min(f32::from(i8::zero()));
        let max = max.max(f32::from(i8::zero()));

        // the min and max quantized values, as floating-point values
        let qmin = f32::from(i8::min_value());
        let qmax = f32::from(i8::max_value());

        // First determine the scale.
        let scale = (max - min) as f64 / (qmax - qmin) as f64;

        // Zero-point computation.
        // First the initial floating-point computation. The zero-point can be
        // determined from solving an affine equation for any known pair
        // (real value, corresponding quantized value).
        // We know two such pairs: (rmin, qmin) and (rmax, qmax).
        // Let's use the first one here.
        let initial_zero_point = qmin as f64 - min as f64 / scale;

        // Now we need to nudge the zero point to be an integer
        // (our zero points are integer, and this is motivated by the requirement
        // to be able to represent the real value "0" exactly as a quantized value,
        // which is required in multiple places, for example in Im2col with SAME
        // padding).
        let nudged_zero_point: i8;
        // Note: qmin and qmax are guaranteed to be within range of T
        if initial_zero_point < qmin as f64 {
            nudged_zero_point = qmin as i8;
        } else if initial_zero_point > qmax as f64 {
            nudged_zero_point = qmax as i8;
        } else {
            // Note: I chose to use .ceil() here instead of .round() of the original implementation,
            // because the signed integers have one more value at the negative side. Using .ceil()
            // brings the average case closer to symmetric around zero.
            nudged_zero_point = (initial_zero_point as f32).ceil() as i8;
        }

        QuantizationParams {
            scale: scale as f32,
            zero_point: nudged_zero_point,
        }
    }
}

impl QuantizeInto<u8> for f32 {
    fn quantize(self, params: &QuantizationParams<f32, u8>) -> u8 {
        let transformed_val: f32 = f32::from(params.zero_point) + self / params.scale;
        let clamped_val: f32 = transformed_val.min(255f32).max(0f32);
        clamped_val.round() as u8
    }
}

impl QuantizeInto<i8> for f32 {
    fn quantize(self, params: &QuantizationParams<f32, i8>) -> i8 {
        let transformed_val: f32 = f32::from(params.zero_point) + self / params.scale;
        let clamped_val: f32 = transformed_val.min(127f32).max(-128f32);
        clamped_val.round() as i8
    }
}

pub fn quantize_vec_u8(params: &QuantizationParams<f32, u8>, src: &[f32]) -> Vec<u8> {
    src.iter()
        .map(|&val| val.quantize(params))
        .collect::<Vec<u8>>()
}

pub fn quantize_vec_i8(params: &QuantizationParams<f32, i8>, src: &[f32]) -> Vec<i8> {
    src.iter()
        .map(|&val| val.quantize(params))
        .collect::<Vec<i8>>()
}
