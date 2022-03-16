#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

__forceinline size_t alignceil(size_t n) {
    return ((n + (AVX2_ALIGNMENT - 1)) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT;
}

__forceinline unsigned int padclip(unsigned int index, unsigned int size, unsigned int pad) {
    if (index < pad) {
        return 0;
    }

    unsigned int coord = index - pad;

    return (coord < size) ? coord : size - 1;
}