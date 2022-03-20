#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int transpose_stride1_s(
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

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
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

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
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const __m128i mask = _mm_setmask_ps(3);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 3;

            for (uint j = 0; j < s; j++) {
                __m128 x = _mm_loadu_ps(x_ptr);

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
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * 4;

            for (uint j = 0; j < s; j++) {
                __m128 x = _mm_load_ps(x_ptr);

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

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256 x = _mm256_loadu_ps(x_ptr);

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
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE;

            for (uint j = 0; j < s; j++) {
                __m256 x = _mm256_load_ps(x_ptr);

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

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_loadu_ps(x_ptr);
                __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);

                _mm256_storeu_ps(y_ptr + offset, x0);
                _mm256_maskstore_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, mask, x1);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride16_s(
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 2;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_load_ps(x_ptr);
                __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);

                _mm256_stream_ps(y_ptr + offset, x0);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, x1);

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

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_loadu_ps(x_ptr);
                __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);
                __m256 x2 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);

                _mm256_storeu_ps(y_ptr + offset, x0);
                _mm256_storeu_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, x1);
                _mm256_maskstore_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 2, mask, x2);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride24_s(
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 3;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_load_ps(x_ptr);
                __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
                __m256 x2 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);

                _mm256_stream_ps(y_ptr + offset, x0);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, x1);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 2, x2);

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

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_loadu_ps(x_ptr);
                __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);
                __m256 x2 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
                __m256 x3 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

                _mm256_storeu_ps(y_ptr + offset, x0);
                _mm256_storeu_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, x1);
                _mm256_storeu_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 2, x2);
                _mm256_maskstore_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 3, mask, x3);

                x_ptr += stride;
                offset += r * stride;
            }
        }

        y_ptr += s * r * stride;
    }

    return SUCCESS;
}

int transpose_stride32_s(
    const uint n, const uint r, const uint s,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * AVX2_FLOAT_STRIDE * 4;

            for (uint j = 0; j < s; j++) {
                __m256 x0 = _mm256_load_ps(x_ptr);
                __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
                __m256 x2 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
                __m256 x3 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

                _mm256_stream_ps(y_ptr + offset, x0);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE, x1);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 2, x2);
                _mm256_stream_ps(y_ptr + offset + AVX2_FLOAT_STRIDE * 3, x3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                offset += r * AVX2_FLOAT_STRIDE * 4;
            }
        }

        y_ptr += s * r * AVX2_FLOAT_STRIDE * 4;
    }

    return SUCCESS;
}

int transpose_aligned_s(
    const uint n, const uint r, const uint s, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 8) {
        return transpose_stride8_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 16) {
        return transpose_stride16_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 24) {
        return transpose_stride24_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 32) {
        return transpose_stride32_s(n, r, s, x_ptr, y_ptr);
    }

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_FLOAT_STRIDE * 4) {
                    __m256 x0 = _mm256_load_ps(x_ptr);
                    __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
                    __m256 x2 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
                    __m256 x3 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

                    _mm256_stream_ps(y_ptr + offset + index, x0);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 2, x2);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 3, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    index += AVX2_FLOAT_STRIDE * 4;
                    sr -= AVX2_FLOAT_STRIDE * 4;
                }
                if (sr >= AVX2_FLOAT_STRIDE * 3) {
                    __m256 x0 = _mm256_load_ps(x_ptr);
                    __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
                    __m256 x2 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);

                    _mm256_stream_ps(y_ptr + offset + index, x0);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 2, x2);
                }
                else if (sr >= AVX2_FLOAT_STRIDE * 2) {
                    __m256 x0 = _mm256_load_ps(x_ptr);
                    __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);

                    _mm256_stream_ps(y_ptr + offset + index, x0);
                    _mm256_stream_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                }
                else if (sr >= AVX2_FLOAT_STRIDE) {
                    __m256 x0 = _mm256_load_ps(x_ptr);

                    _mm256_stream_ps(y_ptr + offset + index, x0);
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

    if (stride == 1) {
        return transpose_stride1_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 2) {
        return transpose_stride2_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 3) {
        return transpose_stride3_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride == 4) {
        return transpose_stride4_s(n, r, s, x_ptr, y_ptr);
    }
    if (stride < 8) {
        return transpose_stride5to7_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 16) {
        return transpose_stride9to15_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 24) {
        return transpose_stride17to23_s(n, r, s, stride, x_ptr, y_ptr);
    }
    if (stride < 32) {
        return transpose_stride25to31_s(n, r, s, stride, x_ptr, y_ptr);
    }

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint th = 0; th < n; th++) {

        for (uint k = 0; k < r; k++) {
            uint offset = k * stride;

            for (uint j = 0; j < s; j++) {
                uint sr = stride, index = 0;

                while (sr >= AVX2_FLOAT_STRIDE * 4) {
                    __m256 x0 = _mm256_loadu_ps(x_ptr);
                    __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);
                    __m256 x2 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
                    __m256 x3 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

                    _mm256_storeu_ps(y_ptr + offset + index, x0);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 2, x2);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 3, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    index += AVX2_FLOAT_STRIDE * 4;
                    sr -= AVX2_FLOAT_STRIDE * 4;
                }
                if (sr >= AVX2_FLOAT_STRIDE * 3) {
                    __m256 x0 = _mm256_loadu_ps(x_ptr);
                    __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);
                    __m256 x2 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
                    __m256 x3 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

                    _mm256_storeu_ps(y_ptr + offset + index, x0);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 2, x2);
                    _mm256_maskstore_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 3, mask, x3);
                }
                else if (sr >= AVX2_FLOAT_STRIDE * 2) {
                    __m256 x0 = _mm256_loadu_ps(x_ptr);
                    __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);
                    __m256 x2 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);

                    _mm256_storeu_ps(y_ptr + offset + index, x0);
                    _mm256_storeu_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, x1);
                    _mm256_maskstore_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE * 2, mask, x2);
                }
                else if (sr >= AVX2_FLOAT_STRIDE) {
                    __m256 x0 = _mm256_loadu_ps(x_ptr);
                    __m256 x1 = _mm256_loadu_ps(x_ptr + AVX2_FLOAT_STRIDE);

                    _mm256_storeu_ps(y_ptr + offset + index, x0);
                    _mm256_maskstore_ps(y_ptr + offset + index + AVX2_FLOAT_STRIDE, mask, x1);
                }
                else {
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

void AvxBlas::Transform::Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<float>^ x, Array<float>^ y) {
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