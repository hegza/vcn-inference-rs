#define WIDTH 4
#define HEIGHT 4
#define MP1_BLOCK_DIM 32

// Based on max_pool_1 from sepconv.cl with minor non-breaking modifications
__kernel void max_pool(__global const float *src, __global float *dst) {
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
        //dst[get_local_id(1) * WIDTH / 2 + get_local_id(0)] = locMax > 0? locMax : 0;
        // Without ReLU
        dst[get_local_id(1) * WIDTH / 2 + get_local_id(0)] = locMax;
    }
}