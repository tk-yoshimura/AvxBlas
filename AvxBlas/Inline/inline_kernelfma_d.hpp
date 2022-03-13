#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n16x_d(
    const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_store_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr),
                    y,
                    _mm256_load_pd(w_ptr)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
                )
            );

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
    }
}

__forceinline void kernelfma_aligned_d(
    const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_store_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr),
                    y,
                    _mm256_load_pd(w_ptr)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
                )
            );

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r == AVX2_DOUBLE_STRIDE * 3) {
            _mm256_store_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr),
                    y,
                    _mm256_load_pd(w_ptr)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
        }
        else if (r == AVX2_DOUBLE_STRIDE * 2) {
            _mm256_store_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr),
                    y,
                    _mm256_load_pd(w_ptr)
                )
            );
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
        }
        else if (r == AVX2_DOUBLE_STRIDE) {
            _mm256_store_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_load_pd(src_ptr),
                    y,
                    _mm256_load_pd(w_ptr)
                )
            );
        }

        w_ptr += r;
    }
}

__forceinline void kernelfma_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_storeu_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr),
                    y,
                    _mm256_loadu_pd(w_ptr)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
                )
            );

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_storeu_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr),
                    y,
                    _mm256_loadu_pd(w_ptr)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3,
                mask,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
                )
            );
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_storeu_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr),
                    y,
                    _mm256_loadu_pd(w_ptr)
                )
            );
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2,
                mask,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
                )
            );
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_storeu_pd(w_ptr,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr),
                    y,
                    _mm256_loadu_pd(w_ptr)
                )
            );
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE,
                mask,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                    y,
                    _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
                )
            );
        }
        else {
            _mm256_maskstore_pd(w_ptr,
                mask,
                _mm256_fmadd_pd(
                    _mm256_loadu_pd(src_ptr),
                    y,
                    _mm256_loadu_pd(w_ptr)
                )
            );
        }

        w_ptr += r;
    }
}