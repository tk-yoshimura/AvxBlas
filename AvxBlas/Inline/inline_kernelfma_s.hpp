#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dilate.hpp"
#include "inline_set.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n1_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 1 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set1_ps(x_ptr[0]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE * 4;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE * 2;
        w_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr),
                _mm256_loadu_ps(w_ptr)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE;
        w_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if(r > 0){
        _mm256_maskstore_ps(w_ptr,
            mask,
            _mm256_fmadd_ps(
                x,
                _mm256_loadu_ps(src_ptr),
                _mm256_loadu_ps(w_ptr)
            )
        );
    }
}

__forceinline void kernelfma_n2_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 2 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set2_ps(x_ptr[0], x_ptr[1]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE)),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3 / 2)),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE * 2;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE;
        w_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );

        src_ptr += AVX2_FLOAT_STRIDE / 2;
        w_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_maskstore_ps(w_ptr,
            mask,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
    }
}

__forceinline void kernelfma_n3_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 3 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set3_ps(x_ptr[0], x_ptr[1], x_ptr[2]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= 8) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + 6,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 2)),
                _mm256_loadu_ps(w_ptr + 6)
            )
        );
        _mm256_storeu_ps(w_ptr + 12,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 4)),
                _mm256_loadu_ps(w_ptr + 12)
            )
        );
        _mm256_storeu_ps(w_ptr + 18,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 6)),
                _mm256_loadu_ps(w_ptr + 18)
            )
        );

        src_ptr += 8;
        w_ptr += 24;
        r -= 8;
    }
    if (r >= 4) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + 6,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 2)),
                _mm256_loadu_ps(w_ptr + 6)
            )
        );

        src_ptr += 4;
        w_ptr += 12;
        r -= 4;
    }
    if (r >= 2) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );

        src_ptr += 2;
        w_ptr += 6;
        r -= 2;
    }
    if (r >= 1) {
        _mm256_maskstore_ps(w_ptr,
            mask,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
    }
}

__forceinline void kernelfma_n4_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 4 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set4_ps(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= 8) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + 8,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
                _mm256_loadu_ps(w_ptr + 8)
            )
        );
        _mm256_storeu_ps(w_ptr + 16,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 4)),
                _mm256_loadu_ps(w_ptr + 16)
            )
        );
        _mm256_storeu_ps(w_ptr + 24,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 6)),
                _mm256_loadu_ps(w_ptr + 24)
            )
        );

        src_ptr += 8;
        w_ptr += 32;
        r -= 8;
    }
    if (r >= 4) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
        _mm256_storeu_ps(w_ptr + 8,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
                _mm256_loadu_ps(w_ptr + 8)
            )
        );

        src_ptr += 4;
        w_ptr += 16;
        r -= 4;
    }
    if (r >= 2) {
        _mm256_storeu_ps(w_ptr,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );

        src_ptr += 2;
        w_ptr += 8;
        r -= 2;
    }
    if (r >= 1) {
        _mm256_maskstore_ps(w_ptr,
            mask,
            _mm256_fmadd_ps(
                x,
                _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
                _mm256_loadu_ps(w_ptr)
            )
        );
    }
}

__forceinline void kernelfma_n32x_s(
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

__forceinline void kernelfma_aligned_s(
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

__forceinline void kernelfma_unaligned_s(
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