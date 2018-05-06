// Compiled with:
//  #define CL_PRIM                 type {float, char}
//  #define CL_PRIM<N>              vector type (eg. float2, char2)

// HACK: should be based passed in as a flag (cnn.h for classic, sepconv.h for sepconv)
#include "cnn.h"

// `restrict` makes sure OpenCL knows that the pointers must not overlap
__kernel void mtx_mul(
        __global CL_PRIM* restrict input,
        __global CL_PRIM* restrict output,
        __global CL_PRIM* restrict weights) {

    const int DATA_COUNT = PATCH3SQ * FM_COUNT;

    size_t gid = get_global_id(0);

    CL_PRIM acc = 0.0;
    for (int z = 0; z < DATA_COUNT; z++) {
        acc += weights[DATA_COUNT*gid + z] * input[z];
    }
    output[gid] = acc;

}

__kernel void mtx_mul_vec16(
        __global CL_PRIM16* restrict input,
        __global CL_PRIM16* restrict output,
        __global CL_PRIM16* restrict weights) {

    const int DATA_COUNT = PATCH3SQ * FM_COUNT;

    size_t gid = get_global_id(0);

    CL_PRIM16 acc = (CL_PRIM16)(0.0);
    for (int z = 0; z < DATA_COUNT; ++z) {
        acc += weights[DATA_COUNT * gid + z] * input[z];
    }
    output[gid] = acc;
}
