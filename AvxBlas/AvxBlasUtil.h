#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

namespace AvxBlas {
    extern __m256i masktable_m256(int k);
    extern __m128i masktable_m128(int k);
}