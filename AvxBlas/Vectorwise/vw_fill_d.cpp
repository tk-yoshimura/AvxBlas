#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_set_d.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int vw_fill_stride2_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 2) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d v = _mm256_set2_pd(v_ptr[0], v_ptr[1]);

    const uint maskn = (2 * n) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_stream_pd(y_ptr, v);

        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_maskstore_pd(y_ptr, mask, v);
    }

    return SUCCESS;
}

int vw_fill_stride3_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 3) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256dx3 v = _mm256_set3_pd(v_ptr[0], v_ptr[1], v_ptr[2]);

    const uint maskn = (3 * n) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_stream_x3_pd(y_ptr, v.imm0, v.imm1, v.imm2);

        y_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= 3) { // 3 * r >= AVX2_DOUBLE_STRIDE * 2
        _mm256_maskstore_x3_pd(y_ptr, v.imm0, v.imm1, v.imm2, mask);
    }
    else if (r >= 2) { // 3 * r >= AVX2_DOUBLE_STRIDE
        _mm256_maskstore_x2_pd(y_ptr, v.imm0, v.imm1, mask);
    }
    else if (r >= 1) {
        _mm256_maskstore_x1_pd(y_ptr, v.imm0, mask);
    }

    return SUCCESS;
}

int vw_fill_stride4_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d v = _mm256_load_pd(v_ptr);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE / 4) {
        _mm256_stream_pd(y_ptr, v);

        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 4;
    }

    return SUCCESS;
}

int vw_fill_stride5_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 5) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256dx5 v = _mm256_set5_pd(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3], v_ptr[4]);

    const uint maskn = (5 * n) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_stream_x5_pd(y_ptr, v.imm0, v.imm1, v.imm2, v.imm3, v.imm4);

        y_ptr += AVX2_DOUBLE_STRIDE * 5;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= 3) { // 5 * r >= AVX2_DOUBLE_STRIDE * 3
        _mm256_maskstore_x4_pd(y_ptr, v.imm0, v.imm1, v.imm2, v.imm3, mask);
    }
    else if (r >= 2) { // 5 * r >= AVX2_DOUBLE_STRIDE * 2
        _mm256_maskstore_x3_pd(y_ptr, v.imm0, v.imm1, v.imm2, mask);
    }
    else if (r >= 1) { // 5 * r >= AVX2_DOUBLE_STRIDE
        _mm256_maskstore_x2_pd(y_ptr, v.imm0, v.imm1, mask);
    }

    return SUCCESS;
}

int vw_fill_stride6_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 6) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256dx3 v = _mm256_set6_pd(v_ptr[0], v_ptr[1], v_ptr[2], v_ptr[3], v_ptr[4], v_ptr[5]);

    const uint maskn = (6 * n) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_stream_x3_pd(y_ptr, v.imm0, v.imm1, v.imm2);

        y_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r >= 1) { // 6 * r >= AVX2_DOUBLE_STRIDE
        _mm256_maskstore_x2_pd(y_ptr, v.imm0, v.imm1, mask);
    }

    return SUCCESS;
}

int vw_fill_stride7_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride != 7) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(3);

    __m256d v0, v1;
    _mm256_maskload_x2_pd(v_ptr, v0, v1, mask);

    for (uint i = 0; i < n; i++) {
        _mm256_maskstore_x2_pd(y_ptr, v0, v1, mask);

        y_ptr += 7;
    }

    return SUCCESS;
}

int vw_fill_stride8_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 2) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d v0, v1;
    _mm256_load_x2_pd(v_ptr, v0, v1);

    for (uint i = 0; i < n; i++) {
        _mm256_stream_x2_pd(y_ptr, v0, v1);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
    }

    return SUCCESS;
}

int vw_fill_stride9to11_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    __m256d v0, v1, v2;
    _mm256_maskload_x3_pd(v_ptr, v0, v1, v2, mask);

    for (uint i = 0; i < n; i++) {
        _mm256_maskstore_x3_pd(y_ptr, v0, v1, v2, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_fill_stride12_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 3) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d v0, v1, v2;
    _mm256_load_x3_pd(v_ptr, v0, v1, v2);

    for (uint i = 0; i < n; i++) {
        _mm256_stream_x3_pd(y_ptr, v0, v1, v2);

        y_ptr += AVX2_DOUBLE_STRIDE * 3;
    }

    return SUCCESS;
}

int vw_fill_stride13to15_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    __m256d v0, v1, v2, v3;
    _mm256_maskload_x4_pd(v_ptr, v0, v1, v2, v3, mask);

    for (uint i = 0; i < n; i++) {
        _mm256_maskstore_x4_pd(y_ptr, v0, v1, v2, v3, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int vw_fill_stride16_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 4) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d v0, v1, v2, v3;
    _mm256_load_x4_pd(v_ptr, v0, v1, v2, v3);

    for (uint i = 0; i < n; i++) {
        _mm256_stream_x4_pd(y_ptr, v0, v1, v2, v3);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
    }

    return SUCCESS;
}

int vw_fill_strideleq8_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

    if (stride == 2) {
        return vw_fill_stride2_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == 3) {
        return vw_fill_stride3_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == 4) {
        return vw_fill_stride4_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == 5) {
        return vw_fill_stride5_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == 6) {
        return vw_fill_stride6_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == 7) {
        return vw_fill_stride7_d(n, stride, v_ptr, y_ptr);
    }
    else if (stride == AVX2_DOUBLE_STRIDE) {
        return vw_fill_stride8_d(n, stride, v_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int vw_fill_aligned_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

    if (stride == AVX2_DOUBLE_STRIDE) {
        return vw_fill_stride4_d(n, stride, v_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 2) {
        return vw_fill_stride8_d(n, stride, v_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 3) {
        return vw_fill_stride12_d(n, stride, v_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 4) {
        return vw_fill_stride16_d(n, stride, v_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((stride & AVX2_DOUBLE_REMAIN_MASK) != 0) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d v0, v1, v2, v3;

    for (uint i = 0; i < n; i++) {
        const double* vc_ptr = v_ptr;

        uint r = stride;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(vc_ptr, v0, v1, v2, v3);

            _mm256_stream_x4_pd(y_ptr, v0, v1, v2, v3);

            y_ptr += AVX2_DOUBLE_STRIDE * 4;
            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(vc_ptr, v0, v1);

            _mm256_stream_x2_pd(y_ptr, v0, v1);

            y_ptr += AVX2_DOUBLE_STRIDE * 2;
            vc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(vc_ptr, v0);

            _mm256_stream_x1_pd(y_ptr, v0);

            y_ptr += AVX2_DOUBLE_STRIDE;
        }
    }

    return SUCCESS;
}

int vw_fill_unaligned_d(
    const uint n, const uint stride,
    indoubles v_ptr, outdoubles y_ptr) {

    if (stride <= AVX2_DOUBLE_STRIDE * 2) {
        return vw_fill_strideleq8_d(n, stride, v_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 3) {
        return vw_fill_stride9to11_d(n, stride, v_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 4) {
        return vw_fill_stride13to15_d(n, stride, v_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((stride & AVX2_DOUBLE_REMAIN_MASK) == 0) || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    __m256d v0, v1, v2, v3;

    for (uint i = 0; i < n; i++) {
        const double* vc_ptr = v_ptr;

        uint r = stride;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(vc_ptr, v0, v1, v2, v3);

            _mm256_storeu_x4_pd(y_ptr, v0, v1, v2, v3);

            y_ptr += AVX2_DOUBLE_STRIDE * 4;
            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(vc_ptr, v0, v1);

            _mm256_storeu_x2_pd(y_ptr, v0, v1);

            y_ptr += AVX2_DOUBLE_STRIDE * 2;
            vc_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(vc_ptr, v0);

            _mm256_storeu_x1_pd(y_ptr, v0);

            y_ptr += AVX2_DOUBLE_STRIDE;
            vc_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            _mm256_load_x1_pd(vc_ptr, v0);

            _mm256_maskstore_x1_pd(y_ptr, v0, mask);

            y_ptr += r;
        }
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Vectorwise::Fill(UInt32 n, UInt32 stride, Array<double>^ v, Array<double>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, stride);

    Util::CheckLength(n * stride, y);
    Util::CheckLength(stride, v);

    Util::CheckDuplicateArray(v, y);

    if (stride == 1u) {
        Initialize::Clear(n, v[0], y);
        return;
    }

    const double* v_ptr = (const double*)(v->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = vw_fill_aligned_d(n, stride, v_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = vw_fill_unaligned_d(n, stride, v_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
