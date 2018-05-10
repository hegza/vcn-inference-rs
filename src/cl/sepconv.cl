// Compiled with:
//  #define WIDTH                   i32
//  #define HEIGHT                  i32
//  #define ROWS_BLOCKDIM_Y         i32
//  #define ROWS_2_BLOCKDIM_Y       i32
//  #define CL_PRIM                 type {float, char, ...}
//  #define CL_PRIM<N>              vector type (eg. float2, char2)

#define ROWS_BLOCKDIM_X 96
#define ROWS_2_BLOCKDIM_X 48

#define COLUMNS_BLOCKDIM_X 32
#define COLUMNS_BLOCKDIM_Y 8

#define COLUMNS_2_BLOCKDIM_X 16
#define COLUMNS_2_BLOCKDIM_Y 8

#define KERNEL_RADIUS 2
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
#define C1 3
#define C2 7
#define C3 32

__kernel void row_conv(const __global CL_PRIM *d_Src, __global CL_PRIM *d_Dst, __constant CL_PRIM *c_rowKernel) {

    __local CL_PRIM l_data[ROWS_BLOCKDIM_Y][ROWS_BLOCKDIM_X + KERNEL_RADIUS * 2];

    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (ROWS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
                          get_group_id(0) * ROWS_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH) * (HEIGHT); // global index
    d_Src +=
        (ROWS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) + get_group_id(0) * ROWS_BLOCKDIM_X;

    CL_PRIM sum = 0;

    l_data[liy][lix] = 0;

    if (get_local_id(0) < KERNEL_RADIUS * 2)
        l_data[liy][lix + ROWS_BLOCKDIM_X] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < C2; c++) {

        l_data[liy][lix + KERNEL_RADIUS] = d_Src[lix + c * WIDTH * HEIGHT];

        barrier(CLK_LOCAL_MEM_FENCE);

        CL_PRIM C_sum = 0;

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

__kernel void col_conv(const __global CL_PRIM *d_Src, __global CL_PRIM *d_Dst, __constant CL_PRIM *c_colKernel) {

    __local CL_PRIM l_data[COLUMNS_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_BLOCKDIM_X];
    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (COLUMNS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
                          get_group_id(0) * COLUMNS_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH) * (HEIGHT);
    d_Src += (COLUMNS_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH) +
             get_group_id(0) * COLUMNS_BLOCKDIM_X;

    CL_PRIM sum = 0;

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

        CL_PRIM C_sum = 0;

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

__kernel void row_conv_2(const __global CL_PRIM *d_Src, __global CL_PRIM *d_Dst,
                       __constant CL_PRIM *c_row2Kernel) {

    __local CL_PRIM l_data[ROWS_2_BLOCKDIM_Y][ROWS_2_BLOCKDIM_X + KERNEL_RADIUS * 2];
    const int lix = get_local_id(0);
    const int liy = get_local_id(1);
    const int giy = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int block_idx = (ROWS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
                          get_group_id(0) * ROWS_2_BLOCKDIM_X + lix;
    const int dst_idx = block_idx + (get_group_id(2)) * (WIDTH / 2) * (HEIGHT / 2); // global index

    d_Src += (ROWS_2_BLOCKDIM_Y * get_group_id(1) + liy) * (WIDTH / 2) +
             get_group_id(0) * ROWS_2_BLOCKDIM_X;

    CL_PRIM sum = 0;

    l_data[liy][lix] = 0;

    if (get_local_id(0) < KERNEL_RADIUS * 2)
        l_data[liy][lix + ROWS_2_BLOCKDIM_X] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int c = 0; c < C2; c++) {

        l_data[liy][lix + KERNEL_RADIUS] = d_Src[lix + c * (WIDTH / 2) * (HEIGHT / 2)];

        barrier(CLK_LOCAL_MEM_FENCE);

        CL_PRIM C_sum = 0;

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

__kernel void col_conv_2(const __global CL_PRIM *d_Src, __global CL_PRIM *d_Dst,
                       __constant CL_PRIM *c_col2Kernel) {

    __local CL_PRIM l_data[COLUMNS_2_BLOCKDIM_Y + KERNEL_RADIUS * 2][COLUMNS_2_BLOCKDIM_X];
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

    CL_PRIM sum = 0;

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

        CL_PRIM C_sum = 0;

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
