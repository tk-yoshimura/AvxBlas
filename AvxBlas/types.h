#pragma once

#include <immintrin.h>

typedef unsigned int uint;

typedef const float* __restrict infloats;
typedef const double* __restrict indoubles;
typedef float* __restrict outfloats;
typedef double* __restrict outdoubles;

static_assert(sizeof(float) == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");
static_assert(sizeof(uint) == 4, "sizeof uint must be 4");

union _m32 {
    float f;
    unsigned __int32 i;

    constexpr _m32(unsigned __int32 i) : i(i) { }
};

union _m64 {
    double f;
    unsigned __int64 i;

    constexpr _m64(unsigned __int64 i) : i(i) { }
};

struct __m256dx2 {
    __m256d lo, hi;

    constexpr __m256dx2(__m256d lo, __m256d hi) : lo(lo), hi(hi) { }
};