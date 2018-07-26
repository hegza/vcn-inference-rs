// Compile with:
// -D TSM={}    The tile-size in dimension M
// -D TSN={}    The tile-size in dimension N
// -D TSK={}    The tile-size in dimension K
// -D WPTM={}   The amount of work-per-work-item in dimension M
// -D WPTN={}   The amount of work-per-work-item in dimension N

// Local memory usage:
// local_memory_bytes = 4 * TSK * TSM + 4 * (TSK + 2) * TSN

#include "macros.h"

#define RTSM (TSM / WPTM)                  // The reduced tile-size in dimension M (== number of work-items)
#define RTSN (TSN / WPTN)                  // The reduced tile-size in dimension N (== number of work-items)
#define LPTA ((TSK * WPTM * WPTN) / (TSN)) // The amount of loads-per-work-item for A
#define LPTB ((TSK * WPTM * WPTN) / (TSM)) // The amount of loads-per-work-item for B

// Use 2D register blocking (further increase in work per work-item)
__kernel void myGEMM6(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    // Work-item identifiers
    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM == RTSM)
    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN == RTSN)
    const int offsetM = TSM*get_group_id(0); // Work-group offset
    const int offsetN = TSN*get_group_id(1); // Work-group offset

    // Local memory to fit a tile of A and B
    __local float Asub[TSK][TSM];
    __local float Bsub[TSN][TSK+2];

    // Allocate register space
    float Areg;
    float Breg[WPTN];
    float acc[WPTM][WPTN];

    // Initialise the accumulation registers
#pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
#pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Loop over all tiles
    const int numTiles = K/TSK;
    int t=0;
    do {
        // Load one tile of A and B into local memory
#pragma unroll
        for (int la=0; la<LPTA; la++) {
            int tid = tidn*RTSM + tidm;
            // This can be either `volatile int id` or `int id`, makes no difference on a
            // AMD Radeon HD 7800 Series.
            int id = la*RTSN*RTSM + tid;
            int row = MOD2(id,TSM);
            int col = DIV2(id,TSM);
            int tiledIndex = TSK*t + col;
            Asub[col][row] = A[tiledIndex*M + offsetM + row];
            Bsub[row][col] = B[tiledIndex*N + offsetN + row];
        }

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Loop over the values of a single tile
        for (int k=0; k<TSK; k++) {

            // Cache the values of Bsub in registers
#pragma unroll
            for (int wn=0; wn<WPTN; wn++) {
                int col = tidn + wn*RTSN;
                Breg[wn] = Bsub[col][k];
            }

            // Perform the computation
#pragma unroll
            for (int wm=0; wm<WPTM; wm++) {
                int row = tidm + wm*RTSM;
                Areg = Asub[k][row];
#pragma unroll
                for (int wn=0; wn<WPTN; wn++) {
                    acc[wm][wn] += Areg * Breg[wn];
                }
            }
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next tile
        t++;
    } while (t<numTiles);

    // Store the final results in C
#pragma unroll
    for (int wm=0; wm<WPTM; wm++) {
        int globalRow = offsetM + tidm + wm*RTSM;
#pragma unroll
        for (int wn=0; wn<WPTN; wn++) {
            int globalCol = offsetN + tidn + wn*RTSN;
            C[globalCol*M + globalRow] = acc[wm][wn];
        }
    }
}
