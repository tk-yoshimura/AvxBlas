#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "../../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

int transpose_stride1_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

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

int transpose_stride2_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 2;

            for (uint j = 0; j < s; j++) {
                y_ptr[offset] = x_ptr[0];
                y_ptr[offset + 1] = x_ptr[1];

                x_ptr += 2;
                offset += r * 2;
            }
        }

        y_ptr += s * r * 2;
    }

    return SUCCESS;
}

int transpose_stride3_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(3);

    __m128 x;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 3;

            for (uint j = 0; j < s; j++) {
                x = _mm_loadu_ps(x_ptr);

                _mm_maskstore_ps(y_ptr + offset, mask, x);

                x_ptr += 3;
                offset += r * 3;
            }
        }

        y_ptr += s * r * 3;
    }

    return SUCCESS;
}

int transpose_stride4_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 4) || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 4;

            for (uint j = 0; j < s; j++) {
                x = _mm_load_ps(x_ptr);

                _mm_stream_ps(y_ptr + offset, x);

                x_ptr += 4;
                offset += r * 4;
            }
        }

        y_ptr += s * r * 4;
    }

    return SUCCESS;
}

int transpose_stride5to7_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE / 2 || stride >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride);

    __m256 x;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                x = _mm256_loadu_ps(x_ptr);

                _mm256_maskstore_ps(y_ptr + offset, mask, x);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride8_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE;

            for (uint j = 0; j < s; j++) {
                x = _mm256_load_ps(x_ptr);

                _mm256_stream_ps(y_ptr + offset, x);

                x_ptr += AVX2_FLOAT_STRIDE;
                offset += r * AVX2_FLOAT_STRIDE;
            }
        }

        y_ptr += s * r * AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int transpose_stride9to15_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                _mm256_loadu_x2_ps(x_ptr, x0, x1);
                _mm256_maskstore_x2_ps(y_ptr + offset, x0, x1, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride16_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 2;

            for (uint j = 0; j < s; j++) {
                _mm256_load_x2_ps(x_ptr, x0, x1);
                _mm256_stream_x2_ps(y_ptr + offset, x0, x1);

                x_ptr += AVX2_FLOAT_STRIDE * 2;
                offset += r * AVX2_FLOAT_STRIDE * 2;
            }
        }

        y_ptr += s * r * AVX2_FLOAT_STRIDE * 2;
    }

    return SUCCESS;
}

int transpose_stride17to23_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
                _mm256_maskstore_x3_ps(y_ptr + offset, x0, x1, x2, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride24_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 3;

            for (uint j = 0; j < s; j++) {
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);
                _mm256_stream_x3_ps(y_ptr + offset, x0, x1, x2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                offset += r * AVX2_FLOAT_STRIDE * 3;
            }
        }

        y_ptr += s * r * AVX2_FLOAT_STRIDE * 3;
    }

    return SUCCESS;
}

int transpose_stride25to31_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_maskstore_x4_ps(y_ptr + offset, x0, x1, x2, x3, mask);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride32_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 4;

            for (uint j = 0; j < s; j++) {
                _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_ps(y_ptr + offset, x0, x1, x2, x3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                offset += r * AVX2_FLOAT_STRIDE * 4;
            }
        }

        y_ptr += s * r * AVX2_FLOAT_STRIDE * 4;
    }

    return SUCCESS;
}

int transpose_strideleq8_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == 1) {
        return transpose_stride1_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == 2) {
        return transpose_stride2_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == 3) {
        return transpose_stride3_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == 4) {
        return transpose_stride4_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 8) {
        return transpose_stride5to7_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == 8) {
        return transpose_stride8_s(n, r, s, stride, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int transpose_aligned_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == AVX2_FLOAT_STRIDE) {
        return transpose_stride8_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 2) {
        return transpose_stride16_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 3) {
        return transpose_stride24_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 4) {
        return transpose_stride32_s(n, r, s, stride, x_ptr, y_ptr);
    }

    __m256 x0, x1, x2, x3;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(y_ptr + offset + index, x0, x1, x2, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    index += AVX2_FLOAT_STRIDE * 4;
                    sr -= AVX2_FLOAT_STRIDE * 4;
                }
                if (sr >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_load_x2_ps(x_ptr, x0, x1);
                    _mm256_stream_x2_ps(y_ptr + offset + index, x0, x1);

                    x_ptr += AVX2_FLOAT_STRIDE * 2;
                    index += AVX2_FLOAT_STRIDE * 2;
                    sr -= AVX2_FLOAT_STRIDE * 2;
                }
                if (sr >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(x_ptr, x0);
                    _mm256_stream_x1_ps(y_ptr + offset + index, x0);
                }

                x_ptr += sr;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_unaligned_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride <= AVX2_FLOAT_STRIDE) {
        return transpose_strideleq8_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 2) {
        return transpose_stride9to15_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 3) {
        return transpose_stride17to23_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 4) {
        return transpose_stride25to31_s(n, r, s, stride, x_ptr, y_ptr);
    }

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3;

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_ps(y_ptr + offset + index, x0, x1, x2, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    index += AVX2_FLOAT_STRIDE * 4;
                    sr -= AVX2_FLOAT_STRIDE * 4;
                }
                if (sr >= AVX2_FLOAT_STRIDE * 2) {
                    __m256 x0, x1;
                    _mm256_loadu_x2_ps(x_ptr, x0, x1);
                    _mm256_storeu_x2_ps(y_ptr + offset + index, x0, x1);

                    x_ptr += AVX2_FLOAT_STRIDE * 2;
                    index += AVX2_FLOAT_STRIDE * 2;
                    sr -= AVX2_FLOAT_STRIDE * 2;
                }
                if (sr >= AVX2_FLOAT_STRIDE) {
                    __m256 x0;
                    _mm256_loadu_x1_ps(x_ptr, x0);
                    _mm256_storeu_x1_ps(y_ptr + offset + index, x0);

                    x_ptr += AVX2_FLOAT_STRIDE;
                    index += AVX2_FLOAT_STRIDE;
                    sr -= AVX2_FLOAT_STRIDE;
                }
                if (sr > 0) {
                    __m256 x0 = _mm256_loadu_ps(x_ptr);

                    _mm256_maskstore_ps(y_ptr + offset + index, mask, x0);
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

void AvxBlas::Permutate::Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || r <= 0 || s <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, r, s, stride);
    Util::CheckLength(n * r * s * stride, x, y);
    Util::CheckDuplicateArray(x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = transpose_aligned_s(n, r, s, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = transpose_unaligned_s(n, r, s, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}