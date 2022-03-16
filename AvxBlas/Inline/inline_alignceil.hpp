#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

__forceinline size_t alignceil(size_t n) {
    return ((n + (AVX2_ALIGNMENT - 1)) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT;
}
