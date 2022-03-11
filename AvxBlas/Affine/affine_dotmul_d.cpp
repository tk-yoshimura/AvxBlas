#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_set.hpp"
#include "../Inline/inline_sum.hpp"
#include "../Inline/inline_dotmul.hpp"
#include <memory.h>

using namespace System;

#pragma unmanaged

int affine_stride1_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

    const unsigned int nbb = nb & AVX2_DOUBLE_BATCH_MASK, nbr = nb - nbb;
    const __m256i mask = _mm256_set_mask(nbr * 2);

    if (nb <= 2) {
        for (unsigned int i = 0; i < na; i++) {
            for (unsigned int j = 0; j < nb; j++) {
                double y = a_ptr[i] * b_ptr[j];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (unsigned int i = 0; i < na; i++) {
            __m256d a = _mm256_set1_pd(a_ptr[i]);

            for (unsigned int j = 0; j < nbb; j += AVX2_DOUBLE_STRIDE) {
                __m256d b = _mm256_loadu_pd(b_ptr + j);

                __m256d y = _mm256_mul_pd(a, b);

                _mm256_storeu_pd(y_ptr, y);

                y_ptr += AVX2_DOUBLE_STRIDE;
            }
            {
                __m256d b = _mm256_maskload_pd(b_ptr + nbb, mask);

                __m256d y = _mm256_mul_pd(a, b);

                _mm256_maskstore_pd(y_ptr, mask, y);

                y_ptr += nbr;
            }
        }
    }
    else {
        for (unsigned int i = 0; i < na; i++) {
            __m256d a = _mm256_set1_pd(a_ptr[i]);

            for (unsigned int j = 0; j < nbb; j += AVX2_DOUBLE_STRIDE) {
                __m256d b = _mm256_load_pd(b_ptr + j);

                __m256d y = _mm256_mul_pd(a, b);

                _mm256_store_pd(y_ptr, y);

                y_ptr += AVX2_DOUBLE_STRIDE;
            }
        }
    }

    return SUCCESS;
}

int affine_stride2_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)b_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int nbb = nb / 2 * 2, nbr = nb - nbb;
    const __m256i masksrc = _mm256_set_mask(nbr * 4);
    const __m128i maskdst = _mm_set_mask(nbr * 2);

    if (nb <= 2) {
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            for (unsigned int j = 0, nbs = nb * 2; j < nbs; j += 2) {
                double y = a_ptr[i] * b_ptr[j] + a_ptr[i + 1] * b_ptr[j + 1];

                *y_ptr = y;
                y_ptr++;
            }
        }
    }
    else if (nbr > 0) {
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            __m256d a = _mm256_set2_pd(a_ptr[i], a_ptr[i + 1]);

            for (unsigned int j = 0, nbs = nbb * 2; j < nbs; j += 4) {
                __m256d b = _mm256_loadu_pd(b_ptr + j);

                __m128d y = _mm256_hadd2_pd(_mm256_mul_pd(a, b));

                _mm_storeu_pd(y_ptr, y);

                y_ptr += 2;
            }
            {
                __m256d b = _mm256_maskload_pd(b_ptr + nbb * 2, masksrc);

                __m128d y = _mm256_hadd2_pd(_mm256_mul_pd(a, b));

                _mm_maskstore_pd(y_ptr, maskdst, y);

                y_ptr += nbr;
            }
        }
    }
    else {
        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
            __m256d a = _mm256_set2_pd(a_ptr[i], a_ptr[i + 1]);

            for (unsigned int j = 0, nbs = nbb * 2; j < nbs; j += 4) {
                __m256d b = _mm256_load_pd(b_ptr + j);

                __m128d y = _mm256_hadd2_pd(_mm256_mul_pd(a, b));

                _mm_store_pd(y_ptr, y);

                y_ptr += 2;
            }
        }
    }

    return SUCCESS;
}

int affine_stride3_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

    const __m256i mask = _mm256_set_mask(6);

    for (unsigned int i = 0, nas = na * 3; i < nas; i += 3) {
        __m256d a = _mm256_maskload_pd(a_ptr + i, mask);

        for (unsigned int j = 0, nbs = nb * 3; j < nbs; j += 3) {
            __m256d b = _mm256_maskload_pd(b_ptr + j, mask);

            double y = _mm256_sum4to1_pd(_mm256_mul_pd(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride4_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * 4; i < nas; i += 4) {
        __m256d a = _mm256_load_pd(a_ptr + i);

        for (unsigned int j = 0, nbs = nb * 4; j < nbs; j += 4) {
            __m256d b = _mm256_load_pd(b_ptr + j);

            double y = _mm256_sum4to1_pd(_mm256_mul_pd(a, b));

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride5to7_dotmul_d(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE || stride >= AVX2_DOUBLE_STRIDE * 2) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask((stride & AVX2_DOUBLE_REMAIN_MASK) * 2);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256d a0 = _mm256_loadu_pd(a_ptr + i);
        __m256d a1 = _mm256_maskload_pd(a_ptr + i + AVX2_DOUBLE_STRIDE, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256d b0 = _mm256_loadu_pd(b_ptr + j);
            __m256d b1 = _mm256_maskload_pd(b_ptr + j + AVX2_DOUBLE_STRIDE, mask);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);

            double y = _mm256_sum8to1_pd(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride8_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_DOUBLE_STRIDE * 2; i < nas; i += AVX2_DOUBLE_STRIDE * 2) {
        __m256d a0 = _mm256_load_pd(a_ptr + i);
        __m256d a1 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE);

        for (unsigned int j = 0, nbs = nb * AVX2_DOUBLE_STRIDE * 2; j < nbs; j += AVX2_DOUBLE_STRIDE * 2) {
            __m256d b0 = _mm256_load_pd(b_ptr + j);
            __m256d b1 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);

            double y = _mm256_sum8to1_pd(ab0, ab1);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride9to11_dotmul_d(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask((stride & AVX2_DOUBLE_REMAIN_MASK) * 2);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256d a0 = _mm256_loadu_pd(a_ptr + i);
        __m256d a1 = _mm256_loadu_pd(a_ptr + i + AVX2_DOUBLE_STRIDE);
        __m256d a2 = _mm256_maskload_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 2, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256d b0 = _mm256_loadu_pd(b_ptr + j);
            __m256d b1 = _mm256_loadu_pd(b_ptr + j + AVX2_DOUBLE_STRIDE);
            __m256d b2 = _mm256_maskload_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 2, mask);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);
            __m256d ab2 = _mm256_mul_pd(a2, b2);

            double y = _mm256_sum12to1_pd(ab0, ab1, ab2);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride12_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_DOUBLE_STRIDE * 3; i < nas; i += AVX2_DOUBLE_STRIDE * 3) {
        __m256d a0 = _mm256_load_pd(a_ptr + i);
        __m256d a1 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE);
        __m256d a2 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 2);

        for (unsigned int j = 0, nbs = nb * AVX2_DOUBLE_STRIDE * 3; j < nbs; j += AVX2_DOUBLE_STRIDE * 3) {
            __m256d b0 = _mm256_load_pd(b_ptr + j);
            __m256d b1 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE);
            __m256d b2 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 2);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);
            __m256d ab2 = _mm256_mul_pd(a2, b2);

            double y = _mm256_sum12to1_pd(ab0, ab1, ab2);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride13to15_dotmul_d(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask((stride & AVX2_DOUBLE_REMAIN_MASK) * 2);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        __m256d a0 = _mm256_loadu_pd(a_ptr + i);
        __m256d a1 = _mm256_loadu_pd(a_ptr + i + AVX2_DOUBLE_STRIDE);
        __m256d a2 = _mm256_loadu_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 2);
        __m256d a3 = _mm256_maskload_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 3, mask);

        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            __m256d b0 = _mm256_loadu_pd(b_ptr + j);
            __m256d b1 = _mm256_loadu_pd(b_ptr + j + AVX2_DOUBLE_STRIDE);
            __m256d b2 = _mm256_loadu_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 2);
            __m256d b3 = _mm256_maskload_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 3, mask);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);
            __m256d ab2 = _mm256_mul_pd(a2, b2);
            __m256d ab3 = _mm256_mul_pd(a3, b3);

            double y = _mm256_sum16to1_pd(ab0, ab1, ab2, ab3);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_stride16_dotmul_d(
    const unsigned int na, const unsigned int nb,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0, nas = na * AVX2_DOUBLE_STRIDE * 4; i < nas; i += AVX2_DOUBLE_STRIDE * 4) {
        __m256d a0 = _mm256_load_pd(a_ptr + i);
        __m256d a1 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE);
        __m256d a2 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 2);
        __m256d a3 = _mm256_load_pd(a_ptr + i + AVX2_DOUBLE_STRIDE * 3);

        for (unsigned int j = 0, nbs = nb * AVX2_DOUBLE_STRIDE * 4; j < nbs; j += AVX2_DOUBLE_STRIDE * 4) {
            __m256d b0 = _mm256_load_pd(b_ptr + j);
            __m256d b1 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE);
            __m256d b2 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 2);
            __m256d b3 = _mm256_load_pd(b_ptr + j + AVX2_DOUBLE_STRIDE * 3);

            __m256d ab0 = _mm256_mul_pd(a0, b0);
            __m256d ab1 = _mm256_mul_pd(a1, b1);
            __m256d ab2 = _mm256_mul_pd(a2, b2);
            __m256d ab3 = _mm256_mul_pd(a3, b3);

            double y = _mm256_sum16to1_pd(ab0, ab1, ab2, ab3);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_alignment_dotmul_d(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((stride & AVX2_DOUBLE_REMAIN_MASK) != 0) || ((size_t)a_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)b_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (stride == 4) {
        return affine_stride4_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 8) {
        return affine_stride8_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 12) {
        return affine_stride12_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 16) {
        return affine_stride16_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            double y = dotmul_alignment_d(stride, a_ptr + i, b_ptr + j);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}

int affine_disorder_dotmul_d(
    const unsigned int na, const unsigned int nb, const unsigned int stride,
    const double* __restrict a_ptr, const double* __restrict b_ptr, double* __restrict y_ptr) {

    if (stride == 1) {
        return affine_stride1_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 2) {
        return affine_stride2_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride == 3) {
        return affine_stride3_dotmul_d(na, nb, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 8) {
        return affine_stride5to7_dotmul_d(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 12) {
        return affine_stride9to11_dotmul_d(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }
    if (stride < 16) {
        return affine_stride13to15_dotmul_d(na, nb, stride, a_ptr, b_ptr, y_ptr);
    }

    const __m256i mask = _mm256_set_mask((stride & AVX2_DOUBLE_REMAIN_MASK) * 2);

    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
            double y = dotmul_disorder_d(stride, a_ptr + i, b_ptr + j, mask);

            *y_ptr = y;
            y_ptr++;
        }
    }

    return SUCCESS;
}


#pragma managed

void AvxBlas::Affine::Dotmul(UInt32 na, UInt32 nb, UInt32 stride, Array<double>^ a, Array<double>^ b, Array<double>^ y) {
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

    double* a_ptr = (double*)(a->Ptr.ToPointer());
    double* b_ptr = (double*)(b->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type alignment");
#endif // _DEBUG

        affine_alignment_dotmul_d(na, nb, stride, a_ptr, b_ptr, y_ptr);
        return;
    }

#ifdef _DEBUG
    Console::WriteLine("type disorder");
#endif // _DEBUG

    affine_disorder_dotmul_d(na, nb, stride, a_ptr, b_ptr, y_ptr);
    return;
}