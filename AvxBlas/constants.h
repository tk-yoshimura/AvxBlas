#pragma once

#define SUCCESS (0)
#define FAILURE_BADPARAM (-1)
#define FAILURE_BADALLOC (-2)
#define UNEXECUTED (-3)

#define AVX1_ALIGNMENT (16u)
#define AVX2_ALIGNMENT (32u)

#define AVX2_FLOAT_STRIDE (8u)
#define AVX2_FLOAT_BATCH_MASK (~7u)
#define AVX2_FLOAT_REMAIN_MASK (7u)

#define AVX2_DOUBLE_STRIDE (4u)
#define AVX2_DOUBLE_BATCH_MASK (~3u)
#define AVX2_DOUBLE_REMAIN_MASK (3u)

#define MAX_BATCHES (1073741824u)
#define MAX_CHANNELS (16777216u)
#define MAX_KERNEL_SIZE (65537u)
#define MAX_MAP_SIZE (16777216u)
#define MAX_POOL_STRIDE (64u)