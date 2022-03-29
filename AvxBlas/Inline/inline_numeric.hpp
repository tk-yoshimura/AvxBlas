#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

__forceinline size_t alignceil(size_t n) {
    return ((n + (AVX2_ALIGNMENT - 1)) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT;
}

__forceinline uint padclip(uint index, uint size, uint pad) {
    if (index < pad) {
        return 0;
    }

    uint coord = index - pad;

    return (coord < size) ? coord : size - 1;
}