// Compile with:
// -D WIDTH={}  The vector-width (in number of floats)
// -D TS={}     The square-root of the 2D tile-size (== work-group dims)

#include "macros.h"

#if WIDTH == 1
typedef float floatX;
#elif WIDTH == 2
typedef float2 floatX;
#elif WIDTH == 4
typedef float4 floatX;
#endif

// Use wider data types
__kernel void myGEMM4(const int M, const int N, const int K,
                      const __global floatX* A,
                      const __global floatX* B,
                      __global floatX* C) {

    // Work-item identifiers
    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH
    const int globalCol = TS*get_group_id(1) + col; // 0..N

    // Local memory to fit a tile of TS*TS elements of A and B
    __local floatX Asub[TS][TS/WIDTH];
    __local floatX Bsub[TS][TS/WIDTH];

    // Initialise the accumulation registers
#if WIDTH == 1
    floatX acc = 0.0f;
#elif WIDTH == 2
    floatX acc = { 0.0f, 0.0f };
#elif WIDTH == 4
    floatX acc = { 0.0f, 0.0f, 0.0f, 0.0f };
#endif

    // Loop over all tiles
    const int numTiles = K/TS;
    for (int tile = 0; tile < numTiles; tile++) {

        // Load one tile of A and B into local memory
        const int tiledRow = (TS/WIDTH)*tile + row;
        const int tiledCol = TS*tile + col;
        Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];
        Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        floatX vecA, vecB;
        float valB;
        for (int k=0; k<TS/WIDTH; k++) {
            vecB = Bsub[col][k];
            for (int w=0; w<WIDTH; w++) {
                vecA = Asub[WIDTH*k + w][row];
#if WIDTH == 1
                valB = vecB;
                acc += vecA * valB;
#elif WIDTH == 2
                switch (w) {
                case 0:
                    valB = vecB.x;
                    break;
                case 1:
                    valB = vecB.y;
                    break;
                }
                acc.x += vecA.x * valB;
                acc.y += vecA.y * valB;
#elif WIDTH == 4
                switch (w) {
                case 0:
                    valB = vecB.x;
                    break;
                case 1:
                    valB = vecB.y;
                    break;
                case 2:
                    valB = vecB.z;
                    break;
                case 3:
                    valB = vecB.w;
                    break;
                }
                acc.x += vecA.x * valB;
                acc.y += vecA.y * valB;
                acc.z += vecA.z * valB;
                acc.w += vecA.w * valB;
#endif
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final results in C
    C[globalCol*(M/WIDTH) + globalRow] = acc;
}
