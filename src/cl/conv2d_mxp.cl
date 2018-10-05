/*
 * Combines convolution of input with filters with max pooling and optional ReLU. Calculation is
 * done in NHWC order. See below for compile parameters.
 * Adapted from Jani's implementation of LIDE-C implementation written by Renjie Xie Described in
 * "Resource-Constrained Implementation and Optimization of a Deep Neural Network for Vehicle
 * Classification" by R. Xie, H. Huttunen, S. Lin, S. S. Bhattacharyya, J. Takala, EUSIPCO 2016
 */

// Compile with:
// -D FILTER_WIDTH={}           Filter/kernel width. Input padding must match (FILTER_WIDTH-1)/2
// -D FILTER_HEIGHT={}          Filter/kernel height. Input padding must match (FILTER_HEIGHT-1)/2
// -D IN_WIDTH={}               Input size on the x dimension without padding.
// -D IN_HEIGHT={}              Input size on the y dimension without padding.
// -D IN_CHANNELS={}            #-of input channels (size on the z-dimension). Filters must have
//                              equal amount of channels.

// Temporary
#define MXP_STRIDE 1
// -D CONV2D_MXP_STRIDE={}      Stride of the max pool operation
// -D CONV2D_RELU={}            {1,0} implement ReLU after layer

// Rename flags to local, extra defines
//#define MXP_STRIDE CONV2D_MXP_STRIDE
//#define IMPL_RELU CONV2D_RELU
#define OUT_WIDTH (IN_WIDTH / MXP_STRIDE)
#define OUT_HEIGHT (IN_HEIGHT / MXP_STRIDE)
#define FILTER_FLAT_LEN (FILTER_HEIGHT * FILTER_WIDTH * IN_CHANNELS)
#define PADDING_X ((FILTER_WIDTH-1)/2)
#define PADDING_Y ((FILTER_HEIGHT-1)/2)
#define PADDED_IN_WIDTH (PADDING_X*2 + IN_WIDTH)
#define IN_STRIDE_Y (PADDED_IN_WIDTH * IN_CHANNELS)
#define FILTER_STRIDE_Y (FILTER_WIDTH * IN_CHANNELS)
#define OUT_STRIDE_Z (OUT_HEIGHT * OUT_WIDTH)

/// filters: (output channels, filter height, filter width, input channels)
/// input: (padded input height, padded input width, input channels). input is assumed to be padded with zeroes such that
/// output: (output channels, output height, output width)
__kernel void conv2d_mxp(
    __constant float* restrict filters,
    __global float* restrict input,
    __global float* restrict output)
{
    // get_global_id(0) == feature map, channels
    const size_t out_z = get_global_id(0);
    // get_global_id(1) == output y, wrt. height
    const size_t out_y = get_global_id(1)/* * MXP_STRIDE*/;
    // get_global_id(2) == output x, wrt. width
    const size_t out_x = get_global_id(2)/* * MXP_STRIDE*/;

    // The filter for this work-item is at filters[out_z, :, :, :]
    const size_t filter_start_pos = out_z * FILTER_FLAT_LEN;

    float out = 0;

    // The input for this work-item is at input[out_y..out_y + filter_height, out_x..out_x + filter_width, :]
    for (size_t filter_y = 0; filter_y != FILTER_HEIGHT; ++filter_y) {
        size_t in_y = out_y + filter_y;
        for (size_t filter_x = 0; filter_x != FILTER_WIDTH; ++filter_x) {
            size_t in_x = out_x + filter_x;
            for (size_t in_channel = 0; in_channel != IN_CHANNELS; ++in_channel) {
                size_t filter_idx = filter_start_pos + filter_y * FILTER_STRIDE_Y + filter_x * IN_CHANNELS + in_channel;
                size_t input_idx = in_y * IN_STRIDE_Y + in_x * IN_CHANNELS + in_channel;
                out += filters[filter_idx] * input[input_idx];
            }
        }
    }

    // TODO: mxp
    // TODO: ReLU
    const size_t output_idx = out_z * OUT_STRIDE_Z + out_y * OUT_WIDTH + out_x;
    output[output_idx] = out;
}
