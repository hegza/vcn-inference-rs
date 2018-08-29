// Compile with:
// -D PADDINGX={}
// -D PADDINGY={}

// Pad the P * Q matrix with zeroes to form a P_XL * Q_XL matrix
__kernel void paddingAddZeroes(const int P, const int Q,
                               const __global float* input,
                               const int P_XL, const int Q_XL,
                               __global float* output) {

    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P_XL in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q_XL in blocks of PADDINGY

    // Check whether we are within bounds of the XL matrix
    if (tx < P_XL && ty < Q_XL) {

        // Copy the input or pad a zero
        float value;
        if (tx < P && ty < Q) {
            value = input[ty*P + tx];
        }
        else {
            value = 0.0f;
        }

        // Store the result
        output[ty*P_XL + tx] = value;
    }
}

// =================================================================================================

// Remove padded values from a P_XL * Q_XL matrix to form a P * Q matrix
__kernel void paddingRemoveZeroes(const int P_XL, const int Q_XL,
                                  const __global float* input,
                                  const int P, const int Q,
                                  __global float* output) {

    // Thread identifiers
    const int tx = get_group_id(0)*PADDINGX + get_local_id(0); // 0..P in blocks of PADDINGX
    const int ty = get_group_id(1)*PADDINGY + get_local_id(1); // 0..Q in blocks of PADDINGY


    // Only store the result if within P * Q bounds
    if (tx < P && ty < Q) {
        output[ty*P + tx] = input[ty*P_XL + tx];
    }
}
