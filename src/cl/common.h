// Compile with:
//  -D CL_PRIM      type {float, char}
//  -D VECN         the N after the vector type (eg. 4, 8, ...), default = 1 (or empty)

#pragma once

// Catenate: this allows for generating vector-type independent implementation names.
#define CAT_I(a, b) a##b
#define CAT(a, b) CAT_I(a, b)

// Generates CL_PRIM_N, eg. float2, char2
#ifdef VECN
#define CL_PRIM_N CAT(CL_PRIM, VECN)
#else
#define VECN 1
#define CL_PRIM_N CL_PRIM
#endif

// Select a different multiplication method for single types and vector-types
CL_PRIM gen_dot(CL_PRIM_N a, CL_PRIM_N b)
{
#if VECN == 1
    return a * b;
#else
    return dot(a, b);
#endif
}
