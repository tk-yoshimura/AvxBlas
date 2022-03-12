#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void n32x_kernelfma_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_store_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr),
                    y,
                    _mm256_load_ps(w_ptr)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
                )
            );

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
    }
}

__forceinline void alignment_kernelfma_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_store_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr),
                    y,
                    _mm256_load_ps(w_ptr)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
                )
            );

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r == AVX2_FLOAT_STRIDE * 3) {
            _mm256_store_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr),
                    y,
                    _mm256_load_ps(w_ptr)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
        }
        else if (r == AVX2_FLOAT_STRIDE * 2) {
            _mm256_store_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr),
                    y,
                    _mm256_load_ps(w_ptr)
                )
            );
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
        }
        else if (r == AVX2_FLOAT_STRIDE) {
            _mm256_store_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_load_ps(src_ptr),
                    y,
                    _mm256_load_ps(w_ptr)
                )
            );
        }

        w_ptr += r;
    }
}

__forceinline void disorder_kernelfma_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_storeu_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr),
                    y,
                    _mm256_loadu_ps(w_ptr)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
                )
            );

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_storeu_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr),
                    y,
                    _mm256_loadu_ps(w_ptr)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
                mask,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
                )
            );
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_storeu_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr),
                    y,
                    _mm256_loadu_ps(w_ptr)
                )
            );
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
                mask,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
                )
            );
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_storeu_ps(w_ptr,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr),
                    y,
                    _mm256_loadu_ps(w_ptr)
                )
            );
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE,
                mask,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                    y,
                    _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
                )
            );
        }
        else {
            _mm256_maskstore_ps(w_ptr,
                mask,
                _mm256_fmadd_ps(
                    _mm256_loadu_ps(src_ptr),
                    y,
                    _mm256_loadu_ps(w_ptr)
                )
            );
        }

        w_ptr += r;
    }
}