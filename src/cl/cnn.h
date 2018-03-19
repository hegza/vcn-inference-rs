#pragma once

#define CHANNELS 3
#define PATCH1 96
#define PATCH1SQ (PATCH1 * PATCH1)
#define PATCH2 (PATCH1 / 2)
#define PATCH2SQ (PATCH2 * PATCH2)
#define PATCH3 (PATCH2 / 2)
#define PATCH3SQ (PATCH3 * PATCH3)
#define CONV1SIZE 5
#define PATCH1SQPAD (PATCH1 + CONV1SIZE - 1) * (PATCH1 + CONV1SIZE - 1)
#define CONV1SQ (CONV1SIZE * CONV1SIZE)
#define CONV2SIZE 5
#define PATCH2SQPAD (PATCH2 + CONV2SIZE - 1) * (PATCH2 + CONV2SIZE - 1)
#define CONV2SQ (CONV2SIZE * CONV2SIZE)
#define FM_COUNT 32
#define TILE_NUM 2
#define MAGIC 100
#define CLASSES 4
#define SL1SIZE (FM_COUNT * PATCH1SQ)
#define PAD_NUM 2

#define matrix(pointer, length, row, col) pointer[row * length + col]
#define matrix1(pointer, offset, length, row, col) pointer[offset + row * length + col]
