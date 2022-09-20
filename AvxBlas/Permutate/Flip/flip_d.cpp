#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "../../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int flip_s2_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_CDAB);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_CDAB);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_CDAB);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_CDAB);

        _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_CDAB);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_CDAB);

        _mm256_stream_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x1_pd(x_ptr, x0);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_CDAB);

        _mm256_stream_x1_pd(y_ptr, x0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd((r * 2) & AVX2_DOUBLE_REMAIN_MASK);

        x0 = _mm256_maskload_pd(x_ptr, mask);
        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_CDAB);

        _mm256_maskstore_pd(y_ptr, mask, x0);
    }

    return SUCCESS;
}

int flip_s3_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(2, 1, 0, 5, 4, 3, 6, 7);

    while (r >= 5) {
        x0 = _mm256_loadu_pd(x_ptr);
        x1 = _mm256_loadu_pd(x_ptr + 3);
        x2 = _mm256_loadu_pd(x_ptr + 6);
        x3 = _mm256_loadu_pd(x_ptr + 9);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_DABC);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_DABC);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_DABC);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_DABC);

        _mm256_storeu_pd(y_ptr, x0);
        _mm256_storeu_pd(y_ptr + 3, x1);
        _mm256_storeu_pd(y_ptr + 6, x2);
        _mm256_storeu_pd(y_ptr + 9, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 3;
        y_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= 4;
    }
    while (r >= 1) {
        const __m256i mask = _mm256_setmask_pd(3);

        x0 = _mm256_maskload_pd(x_ptr, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_DABC);

        _mm256_maskstore_pd(y_ptr, mask, x0);

        x_ptr += AVX2_DOUBLE_STRIDE * 3 / 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 3 / 4;
        r -= 1;
    }

    return SUCCESS;
}

int flip_s4_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

        _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);

        _mm256_stream_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 4) {
        _mm256_load_x1_pd(x_ptr, x0);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);

        _mm256_stream_x1_pd(y_ptr, x0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 4;
    }

    return SUCCESS;
}

int flip_s5_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);

        _mm256_maskstore_pd(y_ptr, mask, x1);
        _mm256_storeu_pd(y_ptr + s - AVX2_DOUBLE_STRIDE, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s6_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_DCAB);

        _mm256_maskstore_pd(y_ptr, mask, x1);
        _mm256_storeu_pd(y_ptr + s - AVX2_DOUBLE_STRIDE, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s7_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_DABC);

        _mm256_maskstore_pd(y_ptr, mask, x1);
        _mm256_storeu_pd(y_ptr + s - AVX2_DOUBLE_STRIDE, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s8_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

        _mm256_stream_x4_pd(y_ptr, x1, x0, x3, x2);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 4) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);

        _mm256_stream_x2_pd(y_ptr, x1, x0);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE / 4;
    }

    return SUCCESS;
}

int flip_s9_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);

        _mm256_maskstore_pd(y_ptr, mask, x2);
        _mm256_storeu_x2_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s10_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_DCAB);

        _mm256_maskstore_pd(y_ptr, mask, x2);
        _mm256_storeu_x2_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s11_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_DABC);

        _mm256_maskstore_pd(y_ptr, mask, x2);
        _mm256_storeu_x2_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s12_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2;

    uint r = n;

    while (r > 0) {
        _mm256_load_x3_pd(x_ptr, x0, x1, x2);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);

        _mm256_stream_x3_pd(y_ptr, x2, x1, x0);

        x_ptr += AVX2_DOUBLE_STRIDE * 3;
        y_ptr += AVX2_DOUBLE_STRIDE * 3;
        r--;
    }

    return SUCCESS;
}

int flip_s13_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);

        _mm256_maskstore_pd(y_ptr, mask, x3);
        _mm256_storeu_x3_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 3, x2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s14_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_DCAB);

        _mm256_maskstore_pd(y_ptr, mask, x3);
        _mm256_storeu_x3_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 3, x2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s15_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    while (r > 0) {
        _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_DABC);

        _mm256_maskstore_pd(y_ptr, mask, x3);
        _mm256_storeu_x3_pd(y_ptr + s - AVX2_DOUBLE_STRIDE * 3, x2, x1, x0);

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s16_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r > 0) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
        x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
        x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
        x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

        _mm256_stream_x4_pd(y_ptr, x3, x2, x1, x0);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r--;
    }

    return SUCCESS;
}

int flip_s17to_smod1_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    const uint block = s / (AVX2_DOUBLE_STRIDE * 4) * (AVX2_DOUBLE_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

            _mm256_storeu_x4_pd(y_ptr + s - i - AVX2_DOUBLE_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_maskload_x4_pd(x_ptr + block, x0, x1, x2, x3, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);

            _mm256_maskstore_pd(y_ptr, mask, x3);
            _mm256_storeu_x3_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x2, x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_maskload_x3_pd(x_ptr + block, x0, x1, x2, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);

            _mm256_maskstore_pd(y_ptr, mask, x2);
            _mm256_storeu_x2_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE) {
            _mm256_maskload_x2_pd(x_ptr + block, x0, x1, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);

            _mm256_maskstore_pd(y_ptr, mask, x1);
            _mm256_storeu_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x0);
        }
        else {
            _mm256_maskload_x1_pd(x_ptr + block, x0, mask);

            _mm256_maskstore_pd(y_ptr, mask, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s17to_smod2_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    const uint block = s / (AVX2_DOUBLE_STRIDE * 4) * (AVX2_DOUBLE_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

            _mm256_storeu_x4_pd(y_ptr + s - i - AVX2_DOUBLE_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_maskload_x4_pd(x_ptr + block, x0, x1, x2, x3, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_DCAB);

            _mm256_maskstore_pd(y_ptr, mask, x3);
            _mm256_storeu_x3_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x2, x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_maskload_x3_pd(x_ptr + block, x0, x1, x2, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_DCAB);

            _mm256_maskstore_pd(y_ptr, mask, x2);
            _mm256_storeu_x2_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE) {
            _mm256_maskload_x2_pd(x_ptr + block, x0, x1, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_DCAB);

            _mm256_maskstore_pd(y_ptr, mask, x1);
            _mm256_storeu_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x0);
        }
        else {
            _mm256_maskload_x1_pd(x_ptr + block, x0, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_DCAB);

            _mm256_maskstore_pd(y_ptr, mask, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s17to_smod3_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i mask = _mm256_setmask_pd(s & AVX2_DOUBLE_REMAIN_MASK);

    const uint block = s / (AVX2_DOUBLE_STRIDE * 4) * (AVX2_DOUBLE_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

            _mm256_storeu_x4_pd(y_ptr + s - i - AVX2_DOUBLE_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_maskload_x4_pd(x_ptr + block, x0, x1, x2, x3, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_DABC);

            _mm256_maskstore_pd(y_ptr, mask, x3);
            _mm256_storeu_x3_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x2, x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_maskload_x3_pd(x_ptr + block, x0, x1, x2, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_DABC);

            _mm256_maskstore_pd(y_ptr, mask, x2);
            _mm256_storeu_x2_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE) {
            _mm256_maskload_x2_pd(x_ptr + block, x0, x1, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_DABC);

            _mm256_maskstore_pd(y_ptr, mask, x1);
            _mm256_storeu_pd(y_ptr + (remain & AVX2_DOUBLE_REMAIN_MASK), x0);
        }
        else {
            _mm256_maskload_x1_pd(x_ptr + block, x0, mask);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_DABC);

            _mm256_maskstore_pd(y_ptr, mask, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

int flip_s4x_d(
    const uint n, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    const uint block = s / (AVX2_DOUBLE_STRIDE * 4) * (AVX2_DOUBLE_STRIDE * 4);
    const uint remain = s - block;

    while (r > 0) {
        for (uint i = 0; i < block; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(x_ptr + i, x0, x1, x2, x3);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);
            x3 = _mm256_permute4x64_pd(x3, _MM_PERM_ABCD);

            _mm256_store_x4_pd(y_ptr + s - i - AVX2_DOUBLE_STRIDE * 4, x3, x2, x1, x0);
        }
        if (remain >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_load_x3_pd(x_ptr + block, x0, x1, x2);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);
            x2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABCD);

            _mm256_store_x3_pd(y_ptr, x2, x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(x_ptr + block, x0, x1);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);
            x1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABCD);

            _mm256_store_x2_pd(y_ptr, x1, x0);
        }
        else if (remain >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(x_ptr + block, x0);

            x0 = _mm256_permute4x64_pd(x0, _MM_PERM_ABCD);

            _mm256_store_x1_pd(y_ptr, x0);
        }

        x_ptr += s;
        y_ptr += s;
        r--;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Permutate::Flip(UInt32 n, UInt32 s, Array<double>^ x, Array<double>^ y) {
    if (n <= 0 || s <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, s);
    Util::CheckLength(n * s, x, y);
    Util::CheckDuplicateArray(x, y);

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (s == 1) {
        Elementwise::Copy(n * s, x, y);
        return;
    }
    else if (s == 2) {
        ret = flip_s2_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 3) {
        ret = flip_s3_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 4) {
        ret = flip_s4_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 5) {
        ret = flip_s5_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 6) {
        ret = flip_s6_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 7) {
        ret = flip_s7_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 8) {
        ret = flip_s8_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 9) {
        ret = flip_s9_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 10) {
        ret = flip_s10_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 11) {
        ret = flip_s11_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 12) {
        ret = flip_s12_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 13) {
        ret = flip_s13_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 14) {
        ret = flip_s14_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 15) {
        ret = flip_s15_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 16) {
        ret = flip_s16_d(n, s, x_ptr, y_ptr);
    }
    else if (s == 16) {
        ret = flip_s16_d(n, s, x_ptr, y_ptr);
    }
    else if ((s % AVX2_DOUBLE_STRIDE) == 0) {
        ret = flip_s4x_d(n, s, x_ptr, y_ptr);
    }
    else if ((s % AVX2_DOUBLE_STRIDE) == 1) {
        ret = flip_s17to_smod1_d(n, s, x_ptr, y_ptr);
    }
    else if ((s % AVX2_DOUBLE_STRIDE) == 2) {
        ret = flip_s17to_smod2_d(n, s, x_ptr, y_ptr);
    }
    else if ((s % AVX2_DOUBLE_STRIDE) == 3) {
        ret = flip_s17to_smod3_d(n, s, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}