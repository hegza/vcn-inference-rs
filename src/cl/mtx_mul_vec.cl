// Compiled with:
//  #define CL_PRIM                 type {float, char}
//  #define CL_PRIM<N>              vector type (eg. float2, char2)
//  #define VECN                    the N after the vector type (eg. 4, 8, ...)

// HACK: should be based passed in as a flag (cnn.h for classic, sepconv.h for sepconv)
#include "cnn.h"

// This allows us to generate the type independent implementations.
#define CAT_I(a,b) a##b
#define CAT(a,b) CAT_I(a, b)
#define CL_PRIMN CAT(CL_PRIM, VECN)

__kernel void mtx_mul_vec(
        __global CL_PRIMN* restrict input,
        __global CL_PRIM* restrict output,
        __global CL_PRIMN* restrict weights) {

    // Divide by number of primitives in vector
    const int DATA_SIZE = PATCH3SQ * FM_COUNT / VECN;

    size_t gid = get_global_id(0);

    CL_PRIM acc = 0.0;
    for (int data_idx = 0; data_idx != DATA_SIZE; ++data_idx) {
        const size_t wgt_idx = DATA_SIZE * gid + data_idx;
        acc += dot(weights[wgt_idx], input[data_idx]);
    }
    output[gid] = acc > 0 ? acc : 0;

}
