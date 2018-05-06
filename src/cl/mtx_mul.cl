// Compiled with:
//  #define CL_PRIM                 type {float, char}
//  #define CL_PRIM<N>              vector type (eg. float2, char2)

// HACK: should be based passed in as a flag (cnn.h for classic, sepconv.h for sepconv)
#include "cnn.h"

// `restrict` makes sure OpenCL knows that the pointers must not overlap
__kernel void mtx_mul(__global CL_PRIM* restrict B, __global CL_PRIM* restrict c_mul,
                       __global CL_PRIM* restrict A) {

    const int Mdim = MAGIC;
    const int Kdim = 1;
    const int Ndim = PATCH3SQ * FM_COUNT;

    int i = get_global_id(0);

    CL_PRIM acc = 0.0;
    for (int z = 0; z < Ndim; z++) {
        acc += A[Ndim*i + z] * B[z];
    }
    c_mul[i] = acc;
}
