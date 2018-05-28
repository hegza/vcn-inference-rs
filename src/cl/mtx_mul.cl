// Compile with:
//  -D CL_PRIM      type {float, char}
//  -D VECN         the N after the vector type (eg. 4, 8, ...), default = 1 (or empty)
//  -D NORM_INT     (optional) if one should use the integer normalization method rather than floating point multiplication

// HACK: should be based passed in as a flag (hyperparams.h for classic, sepconv.h for sepconv)
#include "common.h"
#include "hyperparams.h"

#ifdef NORM_INT
    // TODO: see if changing between int and short produces any effect
    // TODO: this should likely be defined on the host
    #define ACCUMULATOR_T int
#else
    #define ACCUMULATOR_T CL_PRIM
#endif

__kernel void mtx_mul(
        __global CL_PRIM_N* restrict input,
        __global CL_PRIM* restrict output,
        __global CL_PRIM_N* restrict weights) {

    // Divide by number of primitives in vector
    const int DATA_SIZE = PATCH3SQ * FM_COUNT / VECN;

    size_t gid = get_global_id(0);

    ACCUMULATOR_T acc = 0.0;
    for (int data_idx = 0; data_idx != DATA_SIZE; ++data_idx) {
        const size_t wgt_idx = DATA_SIZE * gid + data_idx;
        // Do type-independent dot-product
        acc += gen_dot(weights[wgt_idx], input[data_idx]);
    }

    const ACCUMULATOR_T with_relu = acc > 0 ? acc : 0;
#ifdef NORM_INT
    output[gid] = with_relu >> 24);
#else
    output[gid] = with_relu;
#endif
}
