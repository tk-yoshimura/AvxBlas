#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int transpose_stride1_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k;

            for (uint j = 0; j < s; j++) {
                y_ptr[offset] = *x_ptr;

                x_ptr++;
                offset += r;
            }
        }

        y_ptr += s * r;
    }

    return SUCCESS;
}

int transpose_stride2_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 2;

            for (uint j = 0; j < s; j++) {
                __m128d x = _mm_load_pd(x_ptr);

                _mm_stream_pd(y_ptr + offset, x);

                x_ptr += 2;
                offset += r * 2;
            }
        }

        y_ptr += s * r * 2;
    }

    return SUCCESS;
}

int transpose_stride3_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

    const __m256i mask = _mm256_setmask_pd(3);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 3;

            for (uint j = 0; j < s; j++) {
                __m256d x = _mm256_loadu_pd(x_ptr);

                _mm256_maskstore_pd(y_ptr + offset, mask, x);

                x_ptr += 3;
                offset += r * 3;
            }
        }

        y_ptr += s * r * 3;
    }

    return SUCCESS;
}

int transpose_stride4_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 4;

            for (uint j = 0; j < s; j++) {
                __m256d x = _mm256_load_pd(x_ptr);

                _mm256_stream_pd(y_ptr + offset, x);

                x_ptr += 4;
                offset += r * 4;
            }
        }

        y_ptr += s * r * 4;
    }

    return SUCCESS;
}

int transpose_stride5to7_d(
    const uint n, const uint r, const uint s, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE || stride >= AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1;
                _mm256_loadu_x2_pd(x_ptr, x0, x1);
                _mm256_maskstore_x2_pd(y_ptr + offset, x0, x1, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride8_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_DOUBLE_STRIDE * 2;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1;
                _mm256_load_x2_pd(x_ptr, x0, x1);
                _mm256_stream_x2_pd(y_ptr + offset, x0, x1);

                x_ptr += AVX2_DOUBLE_STRIDE * 2;
                offset += r * AVX2_DOUBLE_STRIDE * 2;
            }
        }

        y_ptr += s * r * AVX2_DOUBLE_STRIDE * 2;
    }

    return SUCCESS;
}

int transpose_stride9to11_d(
    const uint n, const uint r, const uint s, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1, x2;
                _mm256_loadu_x3_pd(x_ptr, x0, x1, x2);
                _mm256_maskstore_x3_pd(y_ptr + offset, x0, x1, x2, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride12_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_DOUBLE_STRIDE * 3;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1, x2;
                _mm256_load_x3_pd(x_ptr, x0, x1, x2);
                _mm256_stream_x3_pd(y_ptr + offset, x0, x1, x2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                offset += r * AVX2_DOUBLE_STRIDE * 3;
            }
        }

        y_ptr += s * r * AVX2_DOUBLE_STRIDE * 3;
    }

    return SUCCESS;
}

int transpose_stride13to15_d(
    const uint n, const uint r, const uint s, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1, x2, x3;
                _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
                _mm256_maskstore_x4_pd(y_ptr + offset, x0, x1, x2, x3, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride16_d(
    const uint n, const uint r, const uint s,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_DOUBLE_STRIDE * 4;

            for (uint j = 0; j < s; j++) {
                __m256d x0, x1, x2, x3;
                _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_pd(y_ptr + offset, x0, x1, x2, x3);

                x_ptr += AVX2_DOUBLE_STRIDE * 4;
                offset += r * AVX2_DOUBLE_STRIDE * 4;
            }
        }

        y_ptr += s * r * AVX2_DOUBLE_STRIDE * 4;
    }

    return SUCCESS;
}

int transpose_aligned_d(
    const uint n, const uint r, const uint s, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_DOUBLE_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 4) {
        return transpose_stride4_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 8) {
        return transpose_stride8_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 12) {
        return transpose_stride12_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 16) {
        return transpose_stride16_d(n, r, s, x_ptr, y_ptr);
    }

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_DOUBLE_STRIDE * 4) {
                    __m256d x0, x1, x2, x3;
                    _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_pd(y_ptr + offset + index, x0, x1, x2, x3);

                    x_ptr += AVX2_DOUBLE_STRIDE * 4;
                    index += AVX2_DOUBLE_STRIDE * 4;
                    sr -= AVX2_DOUBLE_STRIDE * 4;
                }
                if (sr >= AVX2_DOUBLE_STRIDE * 2) {
                    __m256d x0, x1;
                    _mm256_load_x2_pd(x_ptr, x0, x1);
                    _mm256_stream_x2_pd(y_ptr + offset + index, x0, x1);

                    x_ptr += AVX2_DOUBLE_STRIDE * 2;
                    index += AVX2_DOUBLE_STRIDE * 2;
                    sr -= AVX2_DOUBLE_STRIDE * 2;
                }
                if (sr >= AVX2_DOUBLE_STRIDE) {
                    __m256d x0;
                    _mm256_load_x1_pd(x_ptr, x0);
                    _mm256_stream_x1_pd(y_ptr + offset + index, x0);

                    _mm256_stream_pd(y_ptr + offset + index, x0);
                }

                x_ptr += sr;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_unaligned_d(
    const uint n, const uint r, const uint s, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 1) {
        return transpose_stride1_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 2) {
        return transpose_stride2_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 3) {
        return transpose_stride3_d(n, r, s, x_ptr, y_ptr);
    }
    if (stride < 8) {
        return transpose_stride5to7_d(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 12) {
        return transpose_stride9to11_d(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 16) {
        return transpose_stride13to15_d(n, r, s, stride, x_ptr, y_ptr);
    }

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_DOUBLE_STRIDE * 4) {
                    __m256d x0, x1, x2, x3;
                    _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_pd(y_ptr + offset + index, x0, x1, x2, x3);

                    x_ptr += AVX2_DOUBLE_STRIDE * 4;
                    index += AVX2_DOUBLE_STRIDE * 4;
                    sr -= AVX2_DOUBLE_STRIDE * 4;
                }
                if (sr >= AVX2_DOUBLE_STRIDE * 2) {
                    __m256d x0, x1;
                    _mm256_loadu_x2_pd(x_ptr, x0, x1);
                    _mm256_storeu_x2_pd(y_ptr + offset + index, x0, x1);

                    x_ptr += AVX2_DOUBLE_STRIDE * 2;
                    index += AVX2_DOUBLE_STRIDE * 2;
                    sr -= AVX2_DOUBLE_STRIDE * 2;
                }
                if (sr >= AVX2_DOUBLE_STRIDE) {
                    __m256d x0;
                    _mm256_loadu_x1_pd(x_ptr, x0);
                    _mm256_storeu_x1_pd(y_ptr + offset + index, x0);

                    x_ptr += AVX2_DOUBLE_STRIDE;
                    index += AVX2_DOUBLE_STRIDE;
                    sr -= AVX2_DOUBLE_STRIDE;
                }
                if (sr > 0) {
                    __m256d x0 = _mm256_loadu_pd(x_ptr);

                    _mm256_maskstore_pd(y_ptr + offset + index, mask, x0);
                }

                x_ptr += sr;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Transform::Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<double>^ x, Array<double>^ y) {
    if (n <= 0 || r <= 0 || s <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, r, s, stride);
    Util::CheckLength(n * r * s * stride, x, y);
    Util::CheckDuplicateArray(x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = transpose_aligned_d(n, r, s, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = transpose_unaligned_d(n, r, s, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}