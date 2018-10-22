/*
 * Combines convolution of input with filters with max pooling and optional ReLU. See below for
 * compile parameters and input orders.
 *
 * Adapted from Jani's implementation of LIDE-C implementation written by Renjie Xie Described in
 * "Resource-Constrained Implementation and Optimization of a Deep Neural Network for Vehicle
 * Classification" by R. Xie, H. Huttunen, S. Lin, S. S. Bhattacharyya, J. Takala, EUSIPCO 2016.
 */

// Compile with:
// -D FILTER_WIDTH={}           Filter/kernel width. Input padding must match (FILTER_WIDTH-1)/2
// -D FILTER_HEIGHT={}          Filter/kernel height. Input padding must match (FILTER_HEIGHT-1)/2
// -D IN_WIDTH={}               Input size on the x dimension without padding.
// -D IN_HEIGHT={}              Input size on the y dimension without padding.
// -D IN_CHANNELS={}            #-of in channels (size on the z-dimension). Filters must have
//                              equal amount of channels.
// -D OUT_CHANNELS={}           #-of feature maps
// -D CL_PRIM={}                double, float
// -D NXCORR                    Use convolution instead of cross-correlation (xcorr is more
//                              efficient)

// Temporary
#define MXP_STRIDE 1
// -D CONV2D_MXP_STRIDE={}      Stride of the max pool operation
// -D CONV2D_RELU={}            {1,0} implement ReLU after layer

// Rename flags to local, extra defines
//#define MXP_STRIDE CONV2D_MXP_STRIDE
//#define IMPL_RELU CONV2D_RELU
#define OUT_WIDTH (IN_WIDTH / MXP_STRIDE)
#define OUT_HEIGHT (IN_HEIGHT / MXP_STRIDE)
#define PADDING_X ((FILTER_WIDTH-1)/2)
#define PADDING_Y ((FILTER_HEIGHT-1)/2)
#define PADDED_IN_WIDTH (PADDING_X*2 + IN_WIDTH)
#ifdef NXCORR
#define FILTER_X (FILTER_WIDTH - filter_x - 1)
#define FILTER_Y (FILTER_HEIGHT - filter_y - 1)
#else
#define FILTER_X filter_x
#define FILTER_Y filter_y
#endif


/// filters: (out channels, filter height, filter width, in channels) [FM,HWC]
/// input: (padded input height, padded input width, in channels) [HWC]
/// output: (output height, output width, output channels) [HWC]
__kernel void conv2d_mxp(
    __constant CL_PRIM* restrict filters,
    __global CL_PRIM* restrict input,
    __global CL_PRIM* restrict output)
{
    // get_global_id(0) == out y
    const size_t out_y = get_global_id(0)/* * MXP_STRIDE*/;
    // get_global_id(1) == out x
    const size_t out_x = get_global_id(1)/* * MXP_STRIDE*/;
    // get_global_id(2) == out channel, feature map
    const size_t out_c = get_global_id(2);

    const size_t filter_start_pos = out_c * (FILTER_HEIGHT * FILTER_WIDTH * IN_CHANNELS);

    CL_PRIM out = 0;

    for (size_t filter_y = 0; filter_y != FILTER_HEIGHT; ++filter_y) {
        size_t in_y = out_y + filter_y;
        for (size_t filter_x = 0; filter_x != FILTER_WIDTH; ++filter_x) {
            size_t in_x = out_x + filter_x;
            for (size_t in_channel = 0; in_channel != IN_CHANNELS; ++in_channel) {
                size_t filter_idx = filter_start_pos + FILTER_Y * (FILTER_WIDTH * IN_CHANNELS) + FILTER_X * IN_CHANNELS + in_channel;
                size_t in_idx = in_y * (PADDED_IN_WIDTH * IN_CHANNELS) + in_x * IN_CHANNELS + in_channel;
                out += filters[filter_idx] * input[in_idx];
            }
        }
    }

    // TODO: mxp
    // TODO: ReLU
    const size_t out_idx = out_y * OUT_WIDTH * OUT_CHANNELS + out_x * OUT_CHANNELS + out_c;
    output[out_idx] = out;
}
