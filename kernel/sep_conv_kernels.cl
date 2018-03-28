#include "cnn.h"

#define WIDTH 96
#define HEIGHT 96

//Fixed parameters
#define ROWS_BLOCKDIM_X  96
#define ROWS_BLOCKDIM_Y  4
#define ROWS_2_BLOCKDIM_X  48
#define ROWS_2_BLOCKDIM_Y  4

#define COLUMNS_BLOCKDIM_X  32
#define COLUMNS_BLOCKDIM_Y  8

#define COLUMNS_2_BLOCKDIM_X  16
#define COLUMNS_2_BLOCKDIM_Y  8

#define KERNEL_RADIUS 2
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define C1 3
#define C2 7
#define C3 32
#define MP1_BLOCK_DIM 32
#define MP2_BLOCK_DIM 16

__kernel void rowConv(__global float *d_Src, __global float *d_Dst, __constant float *c_rowKernel) {

    __local float l_data[ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X + KERNEL_RADIUS * 2];

    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (ROWS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
                          get_group_id(0) * ROWS_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH) * (HEIGHT); // global index
    d_Src +=
        (ROWS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) + get_group_id(0) * ROWS_BLOCKDIM_X;

    float sum = 0;

    l_data[liy][lix] = 0;

    if (get_local_id(0) < KERNEL_RADIUS * 2)
        l_data[liy][lix + ROWS_BLOCKDIM_X] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < C2; c++) {

        l_data[liy][lix + KERNEL_RADIUS] = d_Src[lix + c * WIDTH * HEIGHT];

        barrier(CLK_LOCAL_MEM_FENCE);

        float C_sum = 0;

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            C_sum += c_rowKernel[KERNEL_RADIUS + j + c * KERNEL_LENGTH +
                                 get_group_id(2) * KERNEL_LENGTH * C2] *
                     l_data[liy][lix + j + KERNEL_RADIUS];
        }

        sum += C_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    d_Dst[dst_idx] = sum;
}

__kernel void colConv(__global float *d_Src, __global float *d_Dst, __constant float *c_colKernel) {

    __local float l_data[COLUMNS_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_BLOCKDIM_X];
    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (COLUMNS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
                          get_group_id(0) * COLUMNS_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH) * (HEIGHT);
    d_Src += (COLUMNS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
             get_group_id(0) * COLUMNS_BLOCKDIM_X;

    float sum = 0;

    l_data[liy][lix] = 0;

    if (get_local_id(1) < KERNEL_RADIUS * 2)
        l_data[liy + COLUMNS_BLOCKDIM_Y][lix] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < C1; c++) {

        if (get_group_id(1) == 0) {

            l_data[liy + KERNEL_RADIUS][lix] = d_Src[lix + c * WIDTH * HEIGHT];

            if (get_local_id(1) < KERNEL_RADIUS)

                l_data[liy + COLUMNS_BLOCKDIM_Y + KERNEL_RADIUS][lix] =
                    d_Src[lix + COLUMNS_BLOCKDIM_Y * WIDTH + c * WIDTH * HEIGHT];

        }

        else if (get_group_id(1) > 0 && get_group_id(1) < get_num_groups(1) - 1) {

            l_data[liy][lix] = d_Src[lix - KERNEL_RADIUS * WIDTH + c * WIDTH * HEIGHT];

            if (get_local_id(1) < KERNEL_RADIUS * 2)

                l_data[liy + COLUMNS_BLOCKDIM_Y][lix] =
                    d_Src[lix - KERNEL_RADIUS * WIDTH + COLUMNS_BLOCKDIM_Y * WIDTH +
                          c * WIDTH * HEIGHT];
        }

        else {

            l_data[liy][lix] = d_Src[lix - KERNEL_RADIUS * WIDTH + c * WIDTH * HEIGHT];

            if (get_local_id(1) < KERNEL_RADIUS)

                l_data[liy + COLUMNS_BLOCKDIM_Y][lix] =
                    d_Src[lix - KERNEL_RADIUS * WIDTH + COLUMNS_BLOCKDIM_Y * WIDTH +
                          c * WIDTH * HEIGHT];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float C_sum = 0;

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

            C_sum += c_colKernel[KERNEL_RADIUS + j + c * KERNEL_LENGTH +
                                 get_group_id(2) * KERNEL_LENGTH * C1] *
                     l_data[liy + j + KERNEL_RADIUS][lix];
        }
        sum += C_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    d_Dst[dst_idx] = sum;
}

__kernel void rowConv2(__global float *d_Src, __global float *d_Dst,
                       __constant float *c_row2Kernel) {

    __local float l_data[ROWS_2_BLOCKDIM_Y][ROWS_2_BLOCKDIM_X + KERNEL_RADIUS * 2];
    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (ROWS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
                          get_group_id(0) * ROWS_2_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH / 2) * (HEIGHT / 2); // global index

    d_Src += (ROWS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
             get_group_id(0) * ROWS_2_BLOCKDIM_X;

    float sum = 0;

    l_data[liy][lix] = 0;

    if (get_local_id(0) < KERNEL_RADIUS * 2)
        l_data[liy][lix + ROWS_2_BLOCKDIM_X] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < C2; c++) {

        l_data[liy][lix + KERNEL_RADIUS] = d_Src[lix + c * (WIDTH / 2) * (HEIGHT / 2)];

        barrier(CLK_LOCAL_MEM_FENCE);

        float C_sum = 0;

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

            C_sum += c_row2Kernel[KERNEL_RADIUS + j + c * KERNEL_LENGTH +
                                  get_group_id(2) * KERNEL_LENGTH * C2] *
                     l_data[liy][lix + j + KERNEL_RADIUS];
        }

        sum += C_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    d_Dst[dst_idx] = sum;
}

__kernel void colConv2(__global float *d_Src, __global float *d_Dst,
                       __constant float *c_col2Kernel) {

    __local float l_data[COLUMNS_2_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_2_BLOCKDIM_X];
    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (COLUMNS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
                          get_group_id(0) * COLUMNS_2_BLOCKDIM_X + lix;             // global index
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH / 2) * (HEIGHT / 2); // global index
    d_Src += (COLUMNS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
             get_group_id(0) * COLUMNS_2_BLOCKDIM_X;

    l_data[liy][lix] = 0;

    if (get_local_id(1) < KERNEL_RADIUS * 2)
        l_data[liy + COLUMNS_2_BLOCKDIM_Y][lix] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0;

    for (int c = 0; c < C3; c++) {

        if (get_group_id(1) == 0) {

            l_data[liy + KERNEL_RADIUS][lix] = d_Src[lix + c * WIDTH / 2 * HEIGHT / 2];

            if (get_local_id(1) < KERNEL_RADIUS)

                l_data[liy + COLUMNS_2_BLOCKDIM_Y + KERNEL_RADIUS][lix] =
                    d_Src[lix + COLUMNS_2_BLOCKDIM_Y * WIDTH / 2 + c * WIDTH / 2 * HEIGHT / 2];

        }

        else if (get_group_id(1) > 0 && get_group_id(1) < get_num_groups(1) - 1) {

            l_data[liy][lix] = d_Src[lix - KERNEL_RADIUS * WIDTH / 2 + c * WIDTH / 2 * HEIGHT / 2];

            if (get_local_id(1) < KERNEL_RADIUS * 2)

                l_data[liy + COLUMNS_2_BLOCKDIM_Y][lix] =
                    d_Src[lix - KERNEL_RADIUS * WIDTH / 2 + COLUMNS_2_BLOCKDIM_Y * WIDTH / 2 +
                          c * WIDTH / 2 * HEIGHT / 2];
        }

        else {

            l_data[liy][lix] = d_Src[lix - KERNEL_RADIUS * WIDTH / 2 + c * WIDTH / 2 * HEIGHT / 2];

            if (get_local_id(1) < KERNEL_RADIUS)

                l_data[liy + COLUMNS_2_BLOCKDIM_Y][lix] =
                    d_Src[lix - KERNEL_RADIUS * WIDTH / 2 + COLUMNS_2_BLOCKDIM_Y * WIDTH / 2 +
                          c * WIDTH / 2 * HEIGHT / 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        float C_sum = 0;

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {

            C_sum += c_col2Kernel[KERNEL_RADIUS + j + c * KERNEL_LENGTH +
                                  get_group_id(2) * KERNEL_LENGTH * C3] *
                     l_data[liy + j + KERNEL_RADIUS][lix];
        }
        sum += C_sum;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    d_Dst[dst_idx] = sum;
}

// Update test_mxp.cl if changed
__kernel void MaxPool1(__global const float *src, __global float *dst) {
    dst += get_group_id(0) * get_local_size(0) / 2 +
           get_group_id(1) * WIDTH / 2 * get_local_size(1) / 2 +
           get_group_id(2) * WIDTH / 2 * WIDTH / 2;
    src += get_group_id(0) * get_local_size(0) + get_group_id(1) * get_local_size(1) * WIDTH +
           get_group_id(2) * WIDTH * HEIGHT;

    __local float sh_data[MP1_BLOCK_DIM][MP1_BLOCK_DIM];

    sh_data[get_local_id(1)][get_local_id(0)] = src[get_local_id(1) * WIDTH + get_local_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < MP1_BLOCK_DIM / 2 && get_local_id(1) < MP1_BLOCK_DIM / 2) {

        float locMax = sh_data[get_local_id(1) * 2][get_local_id(0) * 2];

        if (locMax < sh_data[get_local_id(1) * 2][get_local_id(0) * 2 + 1])
            locMax = sh_data[get_local_id(1) * 2][get_local_id(0) * 2 + 1];

        if (locMax < sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2])
            locMax = sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2];

        if (locMax < sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2 + 1])
            locMax = sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2 + 1];

        // With ReLU
        dst[get_local_id(1) * WIDTH / 2 + get_local_id(0)] = locMax > 0? locMax : 0;
        // Without ReLU
        //dst[get_local_id(1) * WIDTH / 2 + get_local_id(0)] = locMax;
    }
}

//<<< (3,3,32) , (16,16) >>>
__kernel void MaxPool2(__global const float *src, __global float *dst) {

    dst += get_group_id(0) * get_local_size(0) / 2 +
           get_group_id(1) * WIDTH / 4 * get_local_size(1) / 2 +
           get_group_id(2) * WIDTH / 4 * HEIGHT / 4;
    src += get_group_id(0) * get_local_size(0) + get_group_id(1) * get_local_size(1) * WIDTH / 2 +
           get_group_id(2) * WIDTH / 2 * HEIGHT / 2;

    __local float sh_data[MP2_BLOCK_DIM][MP2_BLOCK_DIM];

    sh_data[get_local_id(1)][get_local_id(0)] = src[get_local_id(1) * WIDTH / 2 + get_local_id(0)];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < MP2_BLOCK_DIM / 2 && get_local_id(1) < MP2_BLOCK_DIM / 2) {

        float locMax = sh_data[get_local_id(1) * 2][get_local_id(0) * 2];

        if (locMax < sh_data[get_local_id(1) * 2][get_local_id(0) * 2 + 1])
            locMax = sh_data[get_local_id(1) * 2][get_local_id(0) * 2 + 1];

        if (locMax < sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2])
            locMax = sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2];

        if (locMax < sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2 + 1])
            locMax = sh_data[get_local_id(1) * 2 + 1][get_local_id(0) * 2 + 1];

        // With ReLU
        dst[get_local_id(1) * WIDTH / 4 + get_local_id(0)] = locMax > 0 ? locMax : 0;
        // Without ReLU
        //dst[get_local_id(1) * WIDTH / 4 + get_local_id(0)] = locMax;
    }
}

__kernel void mtx_mulf(__global float* restrict B, __global float* restrict c_mul,
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
