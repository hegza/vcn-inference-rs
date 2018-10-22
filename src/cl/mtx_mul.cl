// Compile with:
// -D CL_PRIM={}    type {float, char}
// -D VECN={}       the N after the vector type (eg. 4, 8, ...), default = 1 (or empty)
// -D NORM_INT={}   (optional) if one should use the integer normalization method rather than floating point multiplication

// HACK: should be based passed in as a flag (hyperparams.h for classic, sepconv.h for sepconv)
#include "common.h"
#include "hyperparams.h"

// Does matrix multiplication: ´weights´x´input´=´output´
// input/B: (channels, height, width) [CHW]
// output/c_mul: () [const]
// weights/A: (channels, height, width, const) [CHWN]
__kernel void mtx_mul(
    __global CL_PRIM_N* restrict input,
    __global CL_PRIM* restrict output,
    __global CL_PRIM_N* restrict weights) {

    // Divide by number of primitives in vector
    const size_t DATA_SIZE = PATCH3SQ * FM_COUNT / VECN;

    size_t gid = get_global_id(0);

    CL_PRIM acc = 0.0;
    for (int data_idx = 0; data_idx != DATA_SIZE; ++data_idx) {
        const size_t wgt_idx = DATA_SIZE * gid + data_idx;
        // Do type-independent dot-product
        acc += gen_dot(weights[wgt_idx], input[data_idx]);
    }

    const CL_PRIM with_relu = acc > 0 ? acc : 0;
    output[gid] = with_relu;
}
