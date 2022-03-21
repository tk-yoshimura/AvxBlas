#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_set_s.hpp"
#include "../Inline/inline_sum_s.hpp"
#include "../Inline/inline_dotmul_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int affine_dotmul_stride1_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)b_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint nbb = nb & AVX2_FLOAT_BATCH_MASK, nbr = nb - nbb;
    const __m256i mask = _mm256_setmask_ps(nbr);

    if (nb <= 4) {
        for (uint i = 0; i < na; i++) {
            for (uint j = 0; j < nb; j++) {
                float y = a_ptr[i] * b_ptr[j];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (uint i = 0; i < na; i++) {
            __m256 a = _mm256_set1_ps(a_ptr[i]);

            for (uint j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
                __m256 b = _mm256_loadu_ps(b_ptr + j);

                __m256 y = _mm256_mul_ps(a, b);

                _mm256_storeu_ps(y_ptr, y);

                y_ptr += AVX2_FLOAT_STRIDE;
            }
            {
                __m256 b = _mm256_maskload_ps(b_ptr + nbb, mask);

                __m256 y = _mm256_mul_ps(a, b);

                _mm256_maskstore_ps(y_ptr, mask, y);

                y_ptr += nbr;
            }
        }
    }
    else {
        for (uint i = 0; i < na; i++) {
            __m256 a = _mm256_set1_ps(a_ptr[i]);

            for (uint j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
                __m256 b = _mm256_load_ps(b_ptr + j);

                __m256 y = _mm256_mul_ps(a, b);

                _mm256_store_ps(y_ptr, y);

                y_ptr += AVX2_FLOAT_STRIDE;
            }
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride2_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)b_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint nbb = nb / 4 * 4, nbr = nb - nbb;
    const __m256i masksrc = _mm256_setmask_ps(nbr * 2);
    const __m128i maskdst = _mm_setmask_ps(nbr);

    if (nb <= 2) {
        for (uint i = 0, nas = na * 2; i < nas; i += 2) {
            for (uint j = 0, nbs = nb * 2; j < nbs; j += 2) {
                float y = a_ptr[i] * b_ptr[j] + a_ptr[i + 1] * b_ptr[j + 1];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (uint i = 0, nas = na * 2; i < nas; i += 2) {
            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);

            for (uint j = 0, nbs = nbb * 2; j < nbs; j += 8) {
                __m256 b = _mm256_loadu_ps(b_ptr + j);

                __m128 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));

                _mm_storeu_ps(y_ptr, y);

                y_ptr += 4;
            }
            {
                __m256 b = _mm256_maskload_ps(b_ptr + nbb * 2, masksrc);

                __m128 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));

                _mm_maskstore_ps(y_ptr, maskdst, y);

                y_ptr += nbr;
            }
        }
    }
    else {
        for (uint i = 0, nas = na * 2; i < nas; i += 2) {
            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);

            for (uint j = 0, nbs = nbb * 2; j < nbs; j += 8) {
                __m256 b = _mm256_load_ps(b_ptr + j);

                __m128 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));

                _mm_store_ps(y_ptr, y);

                y_ptr += 4;
            }
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride3_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

    const __m128i mask = _mm_setmask_ps(3);

    for (uint i = 0, nas = na * 3; i < nas; i += 3) {
        __m128 a = _mm_maskload_ps(a_ptr + i, mask);

        for (uint j = 0, nbs = nb * 3; j < nbs; j += 3) {
            __m128 b = _mm_maskload_ps(b_ptr + j, mask);

            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride4_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0, nas = na * 4; i < nas; i += 4) {
        __m128 a = _mm_load_ps(a_ptr + i);

        for (uint j = 0, nbs = nb * 4; j < nbs; j += 4) {
            __m128 b = _mm_load_ps(b_ptr + j);

            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride5to7_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE / 2 || stride >= AVX2_FLOAT_STRIDE) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride);

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        __m256 a = _mm256_maskload_ps(a_ptr + i, mask);

        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256 b = _mm256_maskload_ps(b_ptr + j, mask);

            float y = _mm256_sum8to1_ps(_mm256_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride8_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0, nas = na * AVX2_FLOAT_STRIDE; i < nas; i += AVX2_FLOAT_STRIDE) {
        __m256 a = _mm256_load_ps(a_ptr + i);

        for (uint j = 0, nbs = nb * AVX2_FLOAT_STRIDE; j < nbs; j += AVX2_FLOAT_STRIDE) {
            __m256 b = _mm256_load_ps(b_ptr + j);

            float y = _mm256_sum8to1_ps(_mm256_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride9to15_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 a0, a1, b0, b1;

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        _mm256_maskload_x2_ps(a_ptr + i, a0, a1, mask);
        
        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            _mm256_maskload_x2_ps(b_ptr + i, b0, b1, mask);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);

            float y = _mm256_sum16to1_ps(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride16_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 a0, a1, b0, b1;

    for (uint i = 0, nas = na * AVX2_FLOAT_STRIDE * 2; i < nas; i += AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(a_ptr + i, a0, a1);

        for (uint j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 2; j < nbs; j += AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(b_ptr + i, b0, b1);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);

            float y = _mm256_sum16to1_ps(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride17to23_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 a0, a1, a2, b0, b1, b2;

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        _mm256_maskload_x3_ps(a_ptr + i, a0, a1, a2, mask);

        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            _mm256_maskload_x3_ps(b_ptr + i, b0, b1, b2, mask);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);
            __m256 ab2 = _mm256_mul_ps(a2, b2);

            float y = _mm256_sum24to1_ps(ab0, ab1, ab2);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride24_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 a0, a1, a2, b0, b1, b2;

    for (uint i = 0, nas = na * AVX2_FLOAT_STRIDE * 3; i < nas; i += AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(a_ptr + i, a0, a1, a2);

        for (uint j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 3; j < nbs; j += AVX2_FLOAT_STRIDE * 3) {
            _mm256_load_x3_ps(b_ptr + i, b0, b1, b2);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);
            __m256 ab2 = _mm256_mul_ps(a2, b2);

            float y = _mm256_sum24to1_ps(ab0, ab1, ab2);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride25to31_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        _mm256_maskload_x4_ps(a_ptr + i, a0, a1, a2, a3, mask);
        
        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            _mm256_maskload_x4_ps(b_ptr + i, b0, b1, b2, b3, mask);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);
            __m256 ab2 = _mm256_mul_ps(a2, b2);
            __m256 ab3 = _mm256_mul_ps(a3, b3);

            float y = _mm256_sum32to1_ps(ab0, ab1, ab2, ab3);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_stride32_s(
    const uint na, const uint nb,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;

    for (uint i = 0, nas = na * AVX2_FLOAT_STRIDE * 4; i < nas; i += AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(a_ptr + i, a0, a1, a2, a3);

        for (uint j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 4; j < nbs; j += AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(b_ptr + i, b0, b1, b2, b3);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);
            __m256 ab2 = _mm256_mul_ps(a2, b2);
            __m256 ab3 = _mm256_mul_ps(a3, b3);

            float y = _mm256_sum32to1_ps(ab0, ab1, ab2, ab3);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_aligned_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 8) {
        return affine_dotmul_stride8_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 16) {
        return affine_dotmul_stride16_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 24) {
        return affine_dotmul_stride24_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 32) {
        return affine_dotmul_stride32_s(na, nb, a_ptr, b_ptr, y_ptr);
    }

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            float y = dotmul_aligned_s(stride, a_ptr + i, b_ptr + j);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_dotmul_unaligned_s(
    const uint na, const uint nb, const uint stride,
    infloats a_ptr, infloats b_ptr, outfloats y_ptr) {

    if (stride == 1) {
        return affine_dotmul_stride1_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 2) {
        return affine_dotmul_stride2_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 3) {
        return affine_dotmul_stride3_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 4) {
        return affine_dotmul_stride4_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 8) {
        return affine_dotmul_stride5to7_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 16) {
        return affine_dotmul_stride9to15_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 24) {
        return affine_dotmul_stride17to23_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 32) {
        return affine_dotmul_stride25to31_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0, nas = na * stride; i < nas; i += stride) {
        for (uint j = 0, nbs = nb * stride; j < nbs; j += stride) {
            float y = dotmul_unaligned_s(stride, a_ptr + i, b_ptr + j, mask);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}


#pragma managed

void AvxBlas::Affine::Dotmul(UInt32 na, UInt32 nb, UInt32 stride, Array<float>^ a, Array<float>^ b, Array<float>^ y) {
    if (na <= 0 || nb <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(na, stride);
    Util::CheckProdOverflow(nb, stride);
    Util::CheckProdOverflow(na, nb);

    Util::CheckLength(na * stride, a);
    Util::CheckLength(nb * stride, b);
    Util::CheckLength(na * nb, y);

    Util::CheckDuplicateArray(a, y);
    Util::CheckDuplicateArray(b, y);

    const float* a_ptr = (const float*)(a->Ptr.ToPointer());
    const float* b_ptr = (const float*)(b->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = affine_dotmul_aligned_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = affine_dotmul_unaligned_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
