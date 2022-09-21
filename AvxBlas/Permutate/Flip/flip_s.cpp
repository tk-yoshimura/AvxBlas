#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "../../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

__m256i _mm256_flipperm_ps(const uint n) {
#ifdef _DEBUG
    if (n <= 0 || n >= AVX2_FLOAT_STRIDE) {
        throw std::exception();
    }
#endif // _DEBUG

    int x[AVX2_FLOAT_STRIDE];
    for (uint i = 0; i < AVX2_FLOAT_STRIDE; i++) {
        x[i] = (i < n) ? (n - i - 1) : i;
    }

    return _mm256_loadu_epi32(x);
}

int flip_s2_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute_ps(x0, _MM_PERM_CDAB);
        x1 = _mm256_permute_ps(x1, _MM_PERM_CDAB);
        x2 = _mm256_permute_ps(x2, _MM_PERM_CDAB);
        x3 = _mm256_permute_ps(x3, _MM_PERM_CDAB);

        _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0 = _mm256_permute_ps(x0, _MM_PERM_CDAB);
        x1 = _mm256_permute_ps(x1, _MM_PERM_CDAB);

        _mm256_stream_x2_ps(y_ptr, x0, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0 = _mm256_permute_ps(x0, _MM_PERM_CDAB);

        _mm256_stream_x1_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps((r * 2) & AVX2_FLOAT_REMAIN_MASK);

        x0 = _mm256_maskload_ps(x_ptr, mask);
        x0 = _mm256_permute_ps(x0, _MM_PERM_CDAB);

        _mm256_maskstore_ps(y_ptr, mask, x0);
    }

    return SUCCESS;
}

int flip_s3_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(2, 1, 0, 5, 4, 3, 6, 7);

    while (r >= 9) {
        x0 = _mm256_loadu_ps(x_ptr);
        x1 = _mm256_loadu_ps(x_ptr + 6);
        x2 = _mm256_loadu_ps(x_ptr + 12);
        x3 = _mm256_loadu_ps(x_ptr + 18);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_storeu_ps(y_ptr, x0);
        _mm256_storeu_ps(y_ptr + 6, x1);
        _mm256_storeu_ps(y_ptr + 12, x2);
        _mm256_storeu_ps(y_ptr + 18, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 3;
        y_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= 8;
    }
    while (r >= 2) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_maskstore_ps(y_ptr, mask, x0);

        x_ptr += AVX2_FLOAT_STRIDE * 3 / 4;
        y_ptr += AVX2_FLOAT_STRIDE * 3 / 4;
        r -= 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(3);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_maskstore_ps(y_ptr, mask, x0);
    }

    return SUCCESS;
}

int flip_s4_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute_ps(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute_ps(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute_ps(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute_ps(x3, _MM_PERM_ABCD);

        _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0 = _mm256_permute_ps(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute_ps(x1, _MM_PERM_ABCD);

        _mm256_stream_x2_ps(y_ptr, x0, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0 = _mm256_permute_ps(x0, _MM_PERM_ABCD);

        _mm256_stream_x1_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps((r * 4) & AVX2_FLOAT_REMAIN_MASK);

        x0 = _mm256_maskload_ps(x_ptr, mask);
        x0 = _mm256_permute_ps(x0, _MM_PERM_ABCD);

        _mm256_maskstore_ps(y_ptr, mask, x0);
    }

    return SUCCESS;
}

int flip_s5_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(4, 3, 2, 1, 0, 5, 6, 7);

    while (r >= 5) {
        x0 = _mm256_loadu_ps(x_ptr);
        x1 = _mm256_loadu_ps(x_ptr + 5);
        x2 = _mm256_loadu_ps(x_ptr + 10);
        x3 = _mm256_loadu_ps(x_ptr + 15);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_storeu_ps(y_ptr, x0);
        _mm256_storeu_ps(y_ptr + 5, x1);
        _mm256_storeu_ps(y_ptr + 10, x2);
        _mm256_storeu_ps(y_ptr + 15, x3);

        x_ptr += 20;
        y_ptr += 20;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(5);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_maskstore_ps(y_ptr, mask, x0);

        x_ptr += 5;
        y_ptr += 5;
        r--;
    }

    return SUCCESS;
}

int flip_s6_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(5, 4, 3, 2, 1, 0, 6, 7);

    while (r >= 5) {
        x0 = _mm256_loadu_ps(x_ptr);
        x1 = _mm256_loadu_ps(x_ptr + 6);
        x2 = _mm256_loadu_ps(x_ptr + 12);
        x3 = _mm256_loadu_ps(x_ptr + 18);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_storeu_ps(y_ptr, x0);
        _mm256_storeu_ps(y_ptr + 6, x1);
        _mm256_storeu_ps(y_ptr + 12, x2);
        _mm256_storeu_ps(y_ptr + 18, x3);

        x_ptr += 24;
        y_ptr += 24;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_maskstore_ps(y_ptr, mask, x0);

        x_ptr += 6;
        y_ptr += 6;
        r--;
    }

    return SUCCESS;
}

int flip_s7_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(6, 5, 4, 3, 2, 1, 0, 7);

    while (r >= 5) {
        x0 = _mm256_loadu_ps(x_ptr);
        x1 = _mm256_loadu_ps(x_ptr + 7);
        x2 = _mm256_loadu_ps(x_ptr + 14);
        x3 = _mm256_loadu_ps(x_ptr + 21);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_storeu_ps(y_ptr, x0);
        _mm256_storeu_ps(y_ptr + 7, x1);
        _mm256_storeu_ps(y_ptr + 14, x2);
        _mm256_storeu_ps(y_ptr + 21, x3);

        x_ptr += 28;
        y_ptr += 28;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(7);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_maskstore_ps(y_ptr, mask, x0);

        x_ptr += 7;
        y_ptr += 7;
        r--;
    }

    return SUCCESS;
}

int flip_s8_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    while (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);

        _mm256_stream_x2_ps(y_ptr, x0, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r >= AVX2_FLOAT_STRIDE / 8) {
        _mm256_load_x1_ps(x_ptr, x0);

        x0 = _mm256_permutevar8x32_ps(x0, perm);

        _mm256_stream_x1_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 8;
    }

    return SUCCESS;
}

int flip_s9to15_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1;

    uint r = n;

    const __m256i mask = _mm256_setmask_ps(s & AVX2_FLOAT_REMAIN_MASK);

    const __m256i fperm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i rperm = _mm256_flipperm_ps(s & AVX2_FLOAT_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);

        x0 = _mm256_permutevar8x32_ps(x0, fperm);
        x1 = _mm256_permutevar8x32_ps(x1, rperm);

        _mm256_maskstore_ps(y_ptr, mask, x1);
        _mm256_storeu_ps(y_ptr + s - AVX2_FLOAT_STRIDE, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s16_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    while (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_stream_x4_ps(y_ptr, x1, x0, x3, x2);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r >= AVX2_FLOAT_STRIDE / 8) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);

        _mm256_stream_x2_ps(y_ptr, x1, x0);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 8;
    }

    return SUCCESS;
}

int flip_s17to23_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2;

    uint r = n;

    const __m256i mask = _mm256_setmask_ps(s & AVX2_FLOAT_REMAIN_MASK);

    const __m256i fperm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i rperm = _mm256_flipperm_ps(s & AVX2_FLOAT_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);

        x0 = _mm256_permutevar8x32_ps(x0, fperm);
        x1 = _mm256_permutevar8x32_ps(x1, fperm);
        x2 = _mm256_permutevar8x32_ps(x2, rperm);

        _mm256_maskstore_ps(y_ptr, mask, x2);
        _mm256_storeu_x2_ps(y_ptr + s - AVX2_FLOAT_STRIDE * 2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s24_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    while (r > 0) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);

        _mm256_stream_x3_ps(y_ptr, x2, x1, x0);

        x_ptr += AVX2_FLOAT_STRIDE * 3;
        y_ptr += AVX2_FLOAT_STRIDE * 3;
        r--;
    }

    return SUCCESS;
}

int flip_s25to31_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_ps(s & AVX2_FLOAT_REMAIN_MASK);

    const __m256i fperm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i rperm = _mm256_flipperm_ps(s & AVX2_FLOAT_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);

        x0 = _mm256_permutevar8x32_ps(x0, fperm);
        x1 = _mm256_permutevar8x32_ps(x1, fperm);
        x2 = _mm256_permutevar8x32_ps(x2, fperm);
        x3 = _mm256_permutevar8x32_ps(x3, rperm);

        _mm256_maskstore_ps(y_ptr, mask, x3);
        _mm256_storeu_x3_ps(y_ptr + s - AVX2_FLOAT_STRIDE * 3, x2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s32_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    while (r > 0) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permutevar8x32_ps(x0, perm);
        x1 = _mm256_permutevar8x32_ps(x1, perm);
        x2 = _mm256_permutevar8x32_ps(x2, perm);
        x3 = _mm256_permutevar8x32_ps(x3, perm);

        _mm256_stream_x4_ps(y_ptr, x3, x2, x1, x0);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r--;
    }

    return SUCCESS;
}

int flip_s33to_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_ps(s & AVX2_FLOAT_REMAIN_MASK);

    const __m256i fperm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i rperm = _mm256_flipperm_ps(s & AVX2_FLOAT_REMAIN_MASK);

    const uint block = s / (AVX2_FLOAT_STRIDE * 4) * (AVX2_FLOAT_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permutevar8x32_ps(x0, fperm);
            x1 = _mm256_permutevar8x32_ps(x1, fperm);
            x2 = _mm256_permutevar8x32_ps(x2, fperm);
            x3 = _mm256_permutevar8x32_ps(x3, fperm);

            _mm256_storeu_x4_ps(y_ptr + s - i - AVX2_FLOAT_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_maskload_x4_ps(x_ptr + block, x0, x1, x2, x3, mask);

            x0 = _mm256_permutevar8x32_ps(x0, fperm);
            x1 = _mm256_permutevar8x32_ps(x1, fperm);
            x2 = _mm256_permutevar8x32_ps(x2, fperm);
            x3 = _mm256_permutevar8x32_ps(x3, rperm);

            _mm256_maskstore_ps(y_ptr, mask, x3);
            _mm256_storeu_x3_ps(y_ptr + (remain & AVX2_FLOAT_REMAIN_MASK), x2, x1, x0);
        }
        else if (remain >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_maskload_x3_ps(x_ptr + block, x0, x1, x2, mask);

            x0 = _mm256_permutevar8x32_ps(x0, fperm);
            x1 = _mm256_permutevar8x32_ps(x1, fperm);
            x2 = _mm256_permutevar8x32_ps(x2, rperm);

            _mm256_maskstore_ps(y_ptr, mask, x2);
            _mm256_storeu_x2_ps(y_ptr + (remain & AVX2_FLOAT_REMAIN_MASK), x1, x0);
        }
        else if (remain >= AVX2_FLOAT_STRIDE) {
            _mm256_maskload_x2_ps(x_ptr + block, x0, x1, mask);

            x0 = _mm256_permutevar8x32_ps(x0, fperm);
            x1 = _mm256_permutevar8x32_ps(x1, rperm);

            _mm256_maskstore_ps(y_ptr, mask, x1);
            _mm256_storeu_ps(y_ptr + (remain & AVX2_FLOAT_REMAIN_MASK), x0);
        }
        else {
            _mm256_maskload_x1_ps(x_ptr + block, x0, mask);

            x0 = _mm256_permutevar8x32_ps(x0, rperm);

            _mm256_maskstore_ps(y_ptr, mask, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s8x_s(
    const uint n, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    const uint block = s / (AVX2_FLOAT_STRIDE * 4) * (AVX2_FLOAT_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permutevar8x32_ps(x0, perm);
            x1 = _mm256_permutevar8x32_ps(x1, perm);
            x2 = _mm256_permutevar8x32_ps(x2, perm);
            x3 = _mm256_permutevar8x32_ps(x3, perm);

            _mm256_store_x4_ps(y_ptr + s - i - AVX2_FLOAT_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_load_x3_ps(x_ptr + block, x0, x1, x2);

            x0 = _mm256_permutevar8x32_ps(x0, perm);
            x1 = _mm256_permutevar8x32_ps(x1, perm);
            x2 = _mm256_permutevar8x32_ps(x2, perm);

            _mm256_store_x3_ps(y_ptr, x2, x1, x0);
        }
        else if (remain >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(x_ptr + block, x0, x1);

            x0 = _mm256_permutevar8x32_ps(x0, perm);
            x1 = _mm256_permutevar8x32_ps(x1, perm);

            _mm256_store_x2_ps(y_ptr, x1, x0);
        }
        else if (remain >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(x_ptr + block, x0);

            x0 = _mm256_permutevar8x32_ps(x0, perm);

            _mm256_store_x1_ps(y_ptr, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Permutate::Flip(UInt32 n, UInt32 s, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || s <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, s);
    Util::CheckLength(n * s, x, y);
    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (s == 1) {
        Elementwise::Copy(n * s, x, y);
        return;
    }
    else if(s == 2){
        ret = flip_s2_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 3) {
        ret = flip_s3_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 4) {
        ret = flip_s4_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 5) {
        ret = flip_s5_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 6) {
        ret = flip_s6_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 7) {
        ret = flip_s7_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 8) {
        ret = flip_s8_s(n, s, x_ptr, y_ptr);
    }
    else if (s < 16) {
        ret = flip_s9to15_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 16) {
        ret = flip_s16_s(n, s, x_ptr, y_ptr);
    }
    else if (s < 24) {
        ret = flip_s17to23_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 24) {
        ret = flip_s24_s(n, s, x_ptr, y_ptr);
    }
    else if (s < 32) {
        ret = flip_s25to31_s(n, s, x_ptr, y_ptr);
    }
    else if (s == 32) {
        ret = flip_s32_s(n, s, x_ptr, y_ptr);
    }
    else if ((s % AVX2_FLOAT_STRIDE) == 0) {
        ret = flip_s8x_s(n, s, x_ptr, y_ptr);
    }
    else {
        ret = flip_s33to_s(n, s, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}