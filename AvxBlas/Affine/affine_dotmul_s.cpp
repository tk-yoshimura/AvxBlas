#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_set.hpp"
#include "../Inline/inline_sum.hpp"
#include "../Inline/inline_dotmul.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int affine_stride1_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

    const unsigned int nbb = nb & AVX2_FLOAT_BATCH_MASK, nbr = nb - nbb;
    const __m256i mask = _mm256_set_mask(nbr);

    if (nb <= 4) {
        for (unsigned int i = 0; i < na; i++) {
            for (unsigned int j = 0; j < nb; j++) {
                float y = a_ptr[i] * b_ptr[j];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (unsigned int i = 0; i < na; i++) {
            __m256 a = _mm256_set1_ps(a_ptr[i]);

            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
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
        for (unsigned int i = 0; i < na; i++) {
            __m256 a = _mm256_set1_ps(a_ptr[i]);

            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
                __m256 b = _mm256_load_ps(b_ptr + j);

                __m256 y = _mm256_mul_ps(a, b);

                _mm256_store_ps(y_ptr, y);

                y_ptr += AVX2_FLOAT_STRIDE;
            }
        }
    }

    return SUCCESS;
}

int affine_stride2_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)b_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int nbb = nb / 4 * 4, nbr = nb - nbb;
    const __m256i masksrc = _mm256_set_mask(nbr * 2);
    const __m128i maskdst = _mm_set_mask(nbr);

    if (nb <= 2) {
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            for (unsigned int j = 0, nbs = nb * 2; j < nbs; j += 2) {
                float y = a_ptr[i] * b_ptr[j] + a_ptr[i + 1] * b_ptr[j + 1];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);

            for (unsigned int j = 0, nbs = nbb * 2; j < nbs; j += 8) {
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
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);

            for (unsigned int j = 0, nbs = nbb * 2; j < nbs; j += 8) {
                __m256 b = _mm256_load_ps(b_ptr + j);

                __m128 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));

                _mm_store_ps(y_ptr, y);

                y_ptr += 4;
            }
        }
    }

    return SUCCESS;
}

int affine_stride3_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

    const __m128i mask = _mm_set_mask(3);

    for (unsigned int i = 0, nas = na * 3; i < nas; i += 3) {
        __m128 a = _mm_maskload_ps(a_ptr + i, mask);

        for (unsigned int j = 0, nbs = nb * 3; j < nbs; j += 3) {
            __m128 b = _mm_maskload_ps(b_ptr + j, mask);

            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride4_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * 4; i < nas; i += 4) {
        __m128 a = _mm_load_ps(a_ptr + i);

        for (unsigned int j = 0, nbs = nb * 4; j < nbs; j += 4) {
            __m128 b = _mm_load_ps(b_ptr + j);

            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride5to7_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE / 2 || stride >= AVX2_FLOAT_STRIDE) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(stride);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256 a = _mm256_maskload_ps(a_ptr + i, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256 b = _mm256_maskload_ps(b_ptr + j, mask);

            float y = _mm256_sum8to1_ps(_mm256_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride8_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_FLOAT_STRIDE; i < nas; i += AVX2_FLOAT_STRIDE) {
        __m256 a = _mm256_load_ps(a_ptr + i);

        for (unsigned int j = 0, nbs = nb * AVX2_FLOAT_STRIDE; j < nbs; j += AVX2_FLOAT_STRIDE) {
            __m256 b = _mm256_load_ps(b_ptr + j);

            float y = _mm256_sum8to1_ps(_mm256_mul_ps(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride9to15_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(stride & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256 a0 = _mm256_loadu_ps(a_ptr + i);
        __m256 a1 = _mm256_maskload_ps(a_ptr + i + AVX2_FLOAT_STRIDE, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256 b0 = _mm256_loadu_ps(b_ptr + j);
            __m256 b1 = _mm256_maskload_ps(b_ptr + j + AVX2_FLOAT_STRIDE, mask);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);

            float y = _mm256_sum16to1_ps(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride16_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_FLOAT_STRIDE * 2; i < nas; i += AVX2_FLOAT_STRIDE * 2) {
        __m256 a0 = _mm256_load_ps(a_ptr + i);
        __m256 a1 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE);

        for (unsigned int j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 2; j < nbs; j += AVX2_FLOAT_STRIDE * 2) {
            __m256 b0 = _mm256_load_ps(b_ptr + j);
            __m256 b1 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE);

            __m256 ab0 = _mm256_mul_ps(a0, b0);
            __m256 ab1 = _mm256_mul_ps(a1, b1);

            float y = _mm256_sum16to1_ps(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride17to23_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(stride & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256 a0 = _mm256_loadu_ps(a_ptr + i);
        __m256 a1 = _mm256_loadu_ps(a_ptr + i + AVX2_FLOAT_STRIDE);
        __m256 a2 = _mm256_maskload_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 2, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256 b0 = _mm256_loadu_ps(b_ptr + j);
            __m256 b1 = _mm256_loadu_ps(b_ptr + j + AVX2_FLOAT_STRIDE);
            __m256 b2 = _mm256_maskload_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 2, mask);

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

int affine_stride24_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_FLOAT_STRIDE * 3; i < nas; i += AVX2_FLOAT_STRIDE * 3) {
        __m256 a0 = _mm256_load_ps(a_ptr + i);
        __m256 a1 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE);
        __m256 a2 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 2);

        for (unsigned int j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 3; j < nbs; j += AVX2_FLOAT_STRIDE * 3) {
            __m256 b0 = _mm256_load_ps(b_ptr + j);
            __m256 b1 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE);
            __m256 b2 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 2);

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

int affine_stride25to31_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(stride & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256 a0 = _mm256_loadu_ps(a_ptr + i);
        __m256 a1 = _mm256_loadu_ps(a_ptr + i + AVX2_FLOAT_STRIDE);
        __m256 a2 = _mm256_loadu_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 2);
        __m256 a3 = _mm256_maskload_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 3, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256 b0 = _mm256_loadu_ps(b_ptr + j);
            __m256 b1 = _mm256_loadu_ps(b_ptr + j + AVX2_FLOAT_STRIDE);
            __m256 b2 = _mm256_loadu_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 2);
            __m256 b3 = _mm256_maskload_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 3, mask);

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

int affine_stride32_dotmul_s(
    const unsigned int na, const unsigned int nb,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_FLOAT_STRIDE * 4; i < nas; i += AVX2_FLOAT_STRIDE * 4) {
        __m256 a0 = _mm256_load_ps(a_ptr + i);
        __m256 a1 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE);
        __m256 a2 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 2);
        __m256 a3 = _mm256_load_ps(a_ptr + i + AVX2_FLOAT_STRIDE * 3);

        for (unsigned int j = 0, nbs = nb * AVX2_FLOAT_STRIDE * 4; j < nbs; j += AVX2_FLOAT_STRIDE * 4) {
            __m256 b0 = _mm256_load_ps(b_ptr + j);
            __m256 b1 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE);
            __m256 b2 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 2);
            __m256 b3 = _mm256_load_ps(b_ptr + j + AVX2_FLOAT_STRIDE * 3);

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

int affine_alignment_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 8) {
        return affine_stride8_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 16) {
        return affine_stride16_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 24) {
        return affine_stride24_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 32) {
        return affine_stride32_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            float y = dotmul_alignment_s(stride, a_ptr + i, b_ptr + j);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_disorder_dotmul_s(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {

    if (stride == 1) {
        return affine_stride1_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 2) {
        return affine_stride2_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 3) {
        return affine_stride3_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 4) {
        return affine_stride4_dotmul_s(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 8) {
        return affine_stride5to7_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 16) {
        return affine_stride9to15_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 24) {
        return affine_stride17to23_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 32) {
        return affine_stride25to31_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }

    const __m256i mask = _mm256_set_mask(stride & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            float y = dotmul_disorder_s(stride, a_ptr + i, b_ptr + j, mask);

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

    float* a_ptr = (float*)(a->Ptr.ToPointer());
    float* b_ptr = (float*)(b->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type alignment");
#endif // _DEBUG

        affine_alignment_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
        return;
    }

#ifdef _DEBUG
    Console::WriteLine("type disorder");
#endif // _DEBUG

    affine_disorder_dotmul_s(na, nb, stride, a_ptr, b_ptr, y_ptr);
    return;
}
