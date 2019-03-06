use crate::math::GenericOps;
use ndarray::{Array, ArrayBase, Axis, Dim, Dimension, IntoDimension, Ix3, Ix4, OwnedRepr};
use num_traits::bounds::Bounded;
use num_traits::Zero;
use std::ops::{Index, Mul, Sub};

struct Shape {
    dims: Vec<usize>,
}

#[allow(dead_code)]
enum PaddingStyle {
    Same,
    Valid,
}

/// This function which takes an input (Tensor) and a kernel (Tensor)
/// and returns the convolution of them
/// Args:
///     conv_input: a numpy array of size [input # of channels, input_width, input_height].
///     conv_kernel: a numpy array of size [output # of channels, input # of channels,
///         kernel_width, kernel_height] represents the kernel of the Convolutional Layer's filter.
///     bias: a scalar value, represents the bias of the Convolutional Layer's filter.
///     strides: a tuple of (convolution horizontal stride, convolution vertical stride).
///     padding: type of the padding scheme: 'same' or 'valid'.
/// Returns:
///     output in [input # of channels, input_width, input_height].
#[allow(dead_code)]
fn xcorr2d<S>(
    conv_input: ArrayBase<OwnedRepr<f32>, Ix3>,
    conv_kernel: ArrayBase<OwnedRepr<f32>, Ix4>,
    bias: &[f32],
    strides: S,
    padding: PaddingStyle,
) -> ArrayBase<OwnedRepr<f32>, Ix3>
where
    S: IntoShape,
{
    let strides = strides.into_shape();

    assert_eq!(
        conv_kernel.shape()[1],
        conv_input.shape()[0],
        "the input and the kernel should have the same depth."
    );
    let output_depth = conv_kernel.shape()[0];
    assert_eq!(bias.len(), output_depth);

    let (input_w, input_h) = (conv_input.shape()[1], conv_input.shape()[2]); // input_width and input_height
    let (kernel_w, kernel_h) = (conv_kernel.shape()[2], conv_kernel.shape()[3]); // kernel_width and kernel_height

    let mut output;
    match padding {
        PaddingStyle::Same => {
            let output_height = (input_h as f32 / strides[1] as f32).ceil() as usize;
            let output_width = (input_w as f32 / strides[0] as f32).ceil() as usize;

            // Calculate the number of zeros which are needed to add as padding
            let pad_along_height: usize =
                ((output_height - 1) * strides[1] + kernel_h - input_h).max(0);
            let pad_along_width: usize =
                ((output_width - 1) * strides[0] + kernel_w - input_w).max(0);
            // Amount of zero padding in each direction
            let pad_top: usize = pad_along_height / 2; // NOTE: quotient without division
            let pad_bottom: usize = pad_along_height - pad_top;
            let pad_left: usize = pad_along_width / 2; // NOTE: quotient without division
            let pad_right: usize = pad_along_width - pad_left;
            // convolution output
            output = Array::zeros((output_depth, output_width, output_height));

            // Add zero padding to the input image
            let mut image_padded = Array::zeros((
                conv_input.shape()[0],
                conv_input.shape()[1] + pad_along_width,
                conv_input.shape()[2] + pad_along_height,
            ));
            image_padded
                .slice_mut(s![
                    ..,
                    pad_left as isize..-(pad_right as isize),
                    pad_top as isize..-(pad_bottom as isize)
                ])
                .assign(&conv_input);
            for ch in 0..output_depth {
                // Loop over every pixel of the output
                for x in 0..output_width {
                    for y in 0..output_height {
                        // element-wise multiplication of the kernel and the image
                        *output.get_mut((ch, x, y)).unwrap() =
                            (&conv_kernel.slice(s![ch, .., .., ..])
                                * &image_padded.slice(s![
                                    ..,
                                    x * strides[0]..x * strides[0] + kernel_w,
                                    y * strides[1]..y * strides[1] + kernel_h,
                                ]))
                                .scalar_sum()
                                + bias[ch];
                    }
                }
            }
            /*
            tensorflow  {
                for y in 0..out_height {
                    for x in 0..out_width {
                        gemm(n: 96*96, m: 32, k: 3, a: &filters[x, y], b: &input[x, y], c: &mut out);
                    }
                }
            }
            });
            */
        }
        PaddingStyle::Valid => {
            let output_height =
                (((input_h - kernel_h + 1) as f32 / (strides[1] as f32)).ceil()) as usize;
            let output_width =
                (((input_w - kernel_w + 1) as f32 / (strides[0] as f32)).ceil()) as usize;
            output = Array::zeros((output_depth, output_width, output_height)); // convolution output;
            for ch in 0..output_depth {
                for x in 0..output_width {
                    // Loop over every pixel of the output
                    for y in 0..output_height {
                        // element-wise multiplication of the kernel and the image
                        *output.get_mut((ch, x, y)).unwrap() =
                            (&conv_kernel.slice(s![ch, .., .., ..])
                                * &conv_input.slice(s![
                                    ..,
                                    x * strides[0]..x * strides[0] + kernel_w,
                                    y * strides[1]..y * strides[1] + kernel_h,
                                ]))
                                .scalar_sum()
                                + bias[ch];
                    }
                }
            }
        }
    }

    output
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, idx: usize) -> &usize {
        &self.dims[idx]
    }
}

trait IntoShape {
    fn into_shape(self) -> Shape;
}

impl IntoShape for Shape {
    fn into_shape(self) -> Shape {
        return self;
    }
}

impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Shape {
        Shape {
            dims: vec![self.0, self.1],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::relu;
    use crate::network::sparse::WEIGHTS_DIR;
    use crate::tests::F32_GEMM_MAX_EPSILON;
    use crate::util::{load_jpeg_chw, ReadCsv};
    use crate::verify;

    #[test]
    fn cross_correlation_works_for_sparse_baseline_l1() {
        // Load input in (channels, width, height)-order
        let input = Array::from_shape_vec(
            (3, 96, 96),
            load_jpeg_chw("input/baseline/sparse-f32/in.jpg"),
        )
        .unwrap()
        // Convert from (channels, height, width) to (channels, width, height)
        .permuted_axes((0, 2, 1));

        // Load filters in (out channels, in channels, width, height)-order
        let weights = {
            Array::from_shape_vec(
                (32, 3, 5, 5),
                f32::read_csv(&format!("{}/{}", WEIGHTS_DIR, "conv1-f32-dcwh.csv")),
            )
            .unwrap()
        };

        let output = relu(
            xcorr2d(input, weights, &vec![0f32; 32], (1, 1), PaddingStyle::Same)
                .into_iter()
                .cloned()
                .collect::<Vec<f32>>(),
        );

        // Load output in (channels, width, height)-order
        let model_output = f32::read_csv("input/baseline/sparse-f32/fm1-cwh.csv");

        verify(&output, &model_output, F32_GEMM_MAX_EPSILON);
    }
}
