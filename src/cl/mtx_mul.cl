// Compiled with:
//  #define CL_PRIM                 type {float, char}
//  #define CL_PRIM<N>              vector type (eg. float2, char2)

// HACK: should be based passed in as a flag (cnn.h for classic, sepconv.h for sepconv)
#include "cnn.h"

__kernel void mtx_mul(
        __global CL_PRIM* restrict input,
        __global CL_PRIM* restrict output,
        __global CL_PRIM* restrict weights) {

    const int DATA_SIZE = PATCH3SQ * FM_COUNT;

    size_t gid = get_global_id(0);

    CL_PRIM acc = 0.0;
    for (int data_idx = 0; data_idx != DATA_SIZE; ++data_idx) {
        acc += weights[DATA_SIZE*gid + data_idx] * input[data_idx];
    }
    output[gid] = acc > 0 ? acc : 0;
}
