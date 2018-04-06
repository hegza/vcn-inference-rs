// TODO: figure out the cnn.h dependency thing
#include "cnn.h"

__kernel void conv_relu_1(
    __global float* restrict fifo_in,
    __global float* restrict fifo_out,
    __constant float* restrict wgt)
{
    const int nInWidth = PATCH1 + (2 * PAD_NUM);
    const int nWidth = PATCH1;
    const int nHeight = PATCH1;
    const int nFilterWidth = CONV1SIZE;

    int fm = get_global_id(0);
    int row = get_global_id(1)*PAD_NUM;
    int col = get_global_id(2)*PAD_NUM;
/*
    for (int fm = 0; fm < FM_COUNT; fm++) {
    for (int row = 0; row < nHeight; row += PAD_NUM) {
    for (int col = 0; col < nWidth; col += PAD_NUM) {
*/
    float tmp = 0;
    if ((row > PAD_NUM) && (col > PAD_NUM) && (row < PATCH1+(2*PAD_NUM)) && (col < PATCH1+(2*PAD_NUM))) {
        int rr = row - 2*PAD_NUM;
        int cc = col - 2*PAD_NUM;
        float gssample[PAD_NUM*PAD_NUM] = {0};
        for (int c = 0; c < CHANNELS; c++) {
            for (int rst = 0; rst < PAD_NUM; rst ++) {
                for (int cst = 0; cst < PAD_NUM; cst ++) {

                    for (int i = 0; i < nFilterWidth; i++) {
                        for (int j = 0; j < nFilterWidth; j++) {
                            gssample[rst*PAD_NUM + cst] +=
                                matrix1(fifo_in, (c*PATCH1SQPAD), nInWidth, (rr+rst+i), (cc+cst+j)) *
                                matrix1(wgt, (CONV1SQ * CHANNELS * fm + c * CONV1SQ), nFilterWidth, (nFilterWidth-1-i), (nFilterWidth-1-j));
                        }
                    }

                }
            }
        }


        tmp = tmp > gssample[0] ? tmp : gssample[0];
        tmp = tmp > gssample[1] ? tmp : gssample[1];
        tmp = tmp > gssample[2] ? tmp : gssample[2];
        tmp = tmp > gssample[3] ? tmp : gssample[3];
    }
    matrix1(fifo_out, (fm*PATCH2SQPAD), (PATCH2+2*PAD_NUM), (row/2), (col/2)) = tmp;
/*
    } // col
    } // row
    } // fm
*/
}

__kernel void conv_relu_2(
    __global float* restrict fifo_in,
    __global float* restrict fifo_out,
    __global float* restrict wgt)
{
    const int nInWidth = PATCH2 + (2 * PAD_NUM);
    const int nWidth = PATCH2;
    const int nHeight = PATCH2;
     const int nFilterWidth = CONV2SIZE;

    int fm = get_global_id(0);
    int row = get_global_id(1)*PAD_NUM;
    int col = get_global_id(2)*PAD_NUM;
/*
    for (int fm = 0; fm < FM_COUNT; fm++) {
    for (int row = 0; row < nHeight; row += PAD_NUM) {
    for (int col = 0; col < nWidth; col += PAD_NUM) {
*/

    float gssample[PAD_NUM*PAD_NUM] = {0};
    for (int c = 0; c < FM_COUNT; c++) {
        for (int rst = 0; rst < PAD_NUM; rst ++) {
            for (int cst = 0; cst < PAD_NUM; cst ++) {

                for (int i = 0; i < nFilterWidth; i++) {
                    for (int j = 0; j < nFilterWidth; j++) {
                        gssample[rst*PAD_NUM + cst] +=
                            matrix1(fifo_in, (c*PATCH2SQPAD), nInWidth, (row+rst+i), (col+cst+j)) *
                            matrix1(wgt, (CONV2SQ * FM_COUNT * fm + c * CONV2SQ), nFilterWidth, (nFilterWidth-1-i), (nFilterWidth-1-j));
                    }
                }

            }
        }
    }

    float tmp = 0;
    tmp = tmp > gssample[0] ? tmp : gssample[0];
    tmp = tmp > gssample[1] ? tmp : gssample[1];
    tmp = tmp > gssample[2] ? tmp : gssample[2];
    tmp = tmp > gssample[3] ? tmp : gssample[3];
    matrix1(fifo_out, fm*PATCH3SQ, (PATCH2/TILE_NUM), (row/2), (col/2)) = tmp;
/*
    } // col
    } // row
    } // fm
*/
}
