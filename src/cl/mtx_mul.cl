// HACK: should be based passed in as a flag (cnn.h for classic, sepconv.h for sepconv)
#include "cnn.h"

// `restrict` makes sure OpenCL knows that the pointers must not overlap
__kernel void mtx_mul_f32(__global float* restrict B, __global float* restrict c_mul,
                       __global float* restrict A) {

    const int Mdim = MAGIC;
    const int Kdim = 1;
    const int Ndim = PATCH3SQ * FM_COUNT;

    int i = get_global_id(0);

    float acc = 0.0;
    for (int z = 0; z < Ndim; z++) {
        acc += A[Ndim*i + z] * B[z];
    }
    c_mul[i] = acc;
}
