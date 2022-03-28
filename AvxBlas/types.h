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

struct __m256x2 {
    __m256 imm0, imm1;

    constexpr __m256x2(__m256 imm0, __m256 imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m256dx2 {
    __m256d imm0, imm1;

    constexpr __m256dx2(__m256d imm0, __m256d imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m256x3 {
    __m256 imm0, imm1, imm2;

    constexpr __m256x3(__m256 imm0, __m256 imm1, __m256 imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m256dx3 {
    __m256d imm0, imm1, imm2;

    constexpr __m256dx3(__m256d imm0, __m256d imm1, __m256d imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m256x4 {
    __m256 imm0, imm1, imm2, imm3;

    constexpr __m256x4(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m256dx4 {
    __m256d imm0, imm1, imm2, imm3;

    constexpr __m256dx4(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m256x5 {
    __m256 imm0, imm1, imm2, imm3, imm4;

    constexpr __m256x5(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3, __m256 imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct __m256dx5 {
    __m256d imm0, imm1, imm2, imm3, imm4;

    constexpr __m256dx5(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3, __m256d imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};