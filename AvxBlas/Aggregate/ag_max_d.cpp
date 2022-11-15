#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_max_d.hpp"
#include "../Inline/inline_fill_d.hpp"
#include "../Inline/inline_copy_d.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"
#include <memory.h>
#include <math.h>

using namespace System;

#pragma unmanaged

int ag_max_stride1_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 1) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const uint maskn = samples & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    for (uint i = 0; i < n; i++) {
        __m256d x;
        __m256d s = minf;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                x = _mm256_load_pd(x_ptr);

                s = _mm256_max_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                x = _mm256_loadu_pd(x_ptr);

                s = _mm256_max_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r > 0) {
            x = _mm256_maskload_pd(x_ptr, mask);
            x = _mm256_or_pd(x, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s = _mm256_max_pd(x, s);

            x_ptr += r;
        }

        double y = _mm256_max4to1_pd(s);

        *y_ptr = y;

        y_ptr += 1;
    }

    return SUCCESS;
}

int ag_max_stride2_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const uint maskn = (2 * samples) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);

    for (uint i = 0; i < n; i++) {
        __m256d x;
        __m256d s = minf;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                x = _mm256_load_pd(x_ptr);

                s = _mm256_max_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                x = _mm256_loadu_pd(x_ptr);

                s = _mm256_max_pd(x, s);

                x_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        if (r > 0) {
            x = _mm256_maskload_pd(x_ptr, mask);
            x = _mm256_or_pd(x, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s = _mm256_max_pd(x, s);

            x_ptr += r * 2;
        }

        __m128d y = _mm256_max4to2_pd(s);

        _mm_store_pd(y_ptr, y);

        y_ptr += 2;
    }

    return SUCCESS;
}

int ag_max_stride3_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    const __m256i mask = _mm256_setmask_pd((3 * samples) & AVX2_DOUBLE_REMAIN_MASK);
    const __m256i mask3 = _mm256_setmask_pd(3);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2;
        __m256d s0 = minf, s1 = minf, s2 = minf;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_load_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_loadu_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r > AVX2_DOUBLE_STRIDE * 2 / 3) {
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);
        }
        else if (r > AVX2_DOUBLE_STRIDE / 3) {
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
            x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
        }
        else if (r > 0) {
            _mm256_maskload_x1_pd(x_ptr, x0, mask);
            x0 = _mm256_or_pd(x0, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
        }

        __m256d y = _mm256_max12to3_pd(s0, s1, s2);

        _mm256_maskstore_pd(y_ptr, mask3, y);

        x_ptr += 3 * r;
        y_ptr += 3;
    }

    return SUCCESS;
}

int ag_max_stride4_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    for (uint i = 0; i < n; i++) {
        __m256d x;
        __m256d s = minf;

        for (uint j = 0; j < samples; j++) {
            x = _mm256_load_pd(x_ptr);

            s = _mm256_max_pd(x, s);

            x_ptr += AVX2_DOUBLE_STRIDE;
        }

        _mm256_stream_pd(y_ptr, s);

        y_ptr += AVX2_DOUBLE_STRIDE;
    }

    return SUCCESS;
}

int ag_max_stride5_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 5) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const uint maskn = (5 * samples) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);
    const __m256i mask1 = _mm256_setmask_pd(1);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2, x3, x4;
        __m256d s0 = minf, s1 = minf, s2 = minf, s3 = minf, s4 = minf;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_load_x5_pd(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);
                s3 = _mm256_max_pd(x3, s3);
                s4 = _mm256_max_pd(x4, s4);

                x_ptr += AVX2_DOUBLE_STRIDE * 5;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_loadu_x5_pd(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);
                s3 = _mm256_max_pd(x3, s3);
                s4 = _mm256_max_pd(x4, s4);

                x_ptr += AVX2_DOUBLE_STRIDE * 5;
                r -= AVX2_DOUBLE_STRIDE;
            }
        }
        if (r > AVX2_DOUBLE_STRIDE * 3 / 5) {
            _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);
            x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);
            s3 = _mm256_max_pd(x3, s3);
        }
        else if (r > AVX2_DOUBLE_STRIDE * 2 / 5) {
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);
        }
        else if (r > 0) {
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
            x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
        }

        __m256dx2 y = _mm256_max20to5_pd(s0, s1, s2, s3, s4);

        _mm256_maskstore_x2_pd(y_ptr, y.imm0, y.imm1, mask1);

        x_ptr += 5 * r;
        y_ptr += 5;
    }

    return SUCCESS;
}

int ag_max_stride6_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != 6) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const uint maskn = (6 * samples) & AVX2_DOUBLE_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_pd(maskn);
    const __m256i mask2 = _mm256_setmask_pd(2);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2;
        __m256d s0 = minf, s1 = minf, s2 = minf;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                _mm256_load_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_DOUBLE_STRIDE / 2) {
                _mm256_loadu_x3_pd(x_ptr, x0, x1, x2);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);

                x_ptr += AVX2_DOUBLE_STRIDE * 3;
                r -= AVX2_DOUBLE_STRIDE / 2;
            }
        }
        if (r > 0) {
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask);
            x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
        }

        __m256dx2 y = _mm256_max12to6_pd(s0, s1, s2);

        _mm256_maskstore_x2_pd(y_ptr, y.imm0, y.imm1, mask2);

        x_ptr += 6 * r;
        y_ptr += 6;
    }

    return SUCCESS;
}

int ag_max_stride7_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride != 7) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const __m256i mask3 = _mm256_setmask_pd(3);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1;
        __m256d s0 = minf, s1 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x2_pd(x_ptr, x0, x1, mask3);
            x1 = _mm256_or_pd(x1, _mm256_andnot_pd(_mm256_castsi256_pd(mask3), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);

            x_ptr += 7;
        }

        _mm256_maskstore_x2_pd(y_ptr, s0, s1, mask3);

        y_ptr += 7;
    }

    return SUCCESS;
}

int ag_max_stride8_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1;
        __m256d s0 = minf, s1 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x2_pd(x_ptr, x0, x1);

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
        }

        _mm256_stream_x2_pd(y_ptr, s0, s1);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
    }

    return SUCCESS;
}

int ag_max_stride9to11_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 2 || stride >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2;
        __m256d s0 = minf, s1 = minf, s2 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x3_pd(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_pd(x2, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);

            x_ptr += stride;
        }

        _mm256_maskstore_x3_pd(y_ptr, s0, s1, s2, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_max_stride12_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2;
        __m256d s0 = minf, s1 = minf, s2 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x3_pd(x_ptr, x0, x1, x2);

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);

            x_ptr += AVX2_DOUBLE_STRIDE * 3;
        }

        _mm256_stream_x3_pd(y_ptr, s0, s1, s2);

        y_ptr += AVX2_DOUBLE_STRIDE * 3;
    }

    return SUCCESS;
}

int ag_max_stride13to15_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_DOUBLE_STRIDE * 3 || stride >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);
    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2, x3;
        __m256d s0 = minf, s1 = minf, s2 = minf, s3 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x4_pd(x_ptr, x0, x1, x2, x3, mask);
            x3 = _mm256_or_pd(x3, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);
            s3 = _mm256_max_pd(x3, s3);

            x_ptr += stride;
        }

        _mm256_maskstore_x4_pd(y_ptr, s0, s1, s2, s3, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_max_stride16_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_DOUBLE_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    for (uint i = 0; i < n; i++) {
        __m256d x0, x1, x2, x3;
        __m256d s0 = minf, s1 = minf, s2 = minf, s3 = minf;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

            s0 = _mm256_max_pd(x0, s0);
            s1 = _mm256_max_pd(x1, s1);
            s2 = _mm256_max_pd(x2, s2);
            s3 = _mm256_max_pd(x3, s3);

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
        }

        _mm256_stream_x4_pd(y_ptr, s0, s1, s2, s3);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
    }

    return SUCCESS;
}

int ag_max_strideleq8_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == 1) {
        return ag_max_stride1_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return ag_max_stride2_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return ag_max_stride3_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return ag_max_stride4_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 5) {
        return ag_max_stride5_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 6) {
        return ag_max_stride6_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 7) {
        return ag_max_stride7_d(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 8) {
        return ag_max_stride8_d(n, samples, stride, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_max_aligned_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride == AVX2_DOUBLE_STRIDE) {
        return ag_max_stride4_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 2) {
        return ag_max_stride8_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 3) {
        return ag_max_stride12_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_DOUBLE_STRIDE * 4) {
        return ag_max_stride16_d(n, samples, stride, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((stride & AVX2_DOUBLE_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* s_ptr = (double*)_aligned_malloc((size_t)stride * sizeof(double), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        fill_aligned_d(stride, -HUGE_VAL, s_ptr);

        for (uint j = 0; j < samples; j++) {
            double* sc_ptr = s_ptr;

            __m256d x0, x1, x2, x3;
            __m256d s0, s1, s2, s3;

            uint r = stride;

            while (r >= AVX2_DOUBLE_STRIDE * 4) {
                _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
                _mm256_load_x4_pd(sc_ptr, s0, s1, s2, s3);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);
                s3 = _mm256_max_pd(x3, s3);

                _mm256_store_x4_pd(sc_ptr, s0, s1, s2, s3);

                x_ptr += AVX2_DOUBLE_STRIDE * 4;
                sc_ptr += AVX2_DOUBLE_STRIDE * 4;
                r -= AVX2_DOUBLE_STRIDE * 4;
            }
            if (r >= AVX2_DOUBLE_STRIDE * 2) {
                _mm256_load_x2_pd(x_ptr, x0, x1);
                _mm256_load_x2_pd(sc_ptr, s0, s1);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);

                _mm256_store_x2_pd(sc_ptr, s0, s1);

                x_ptr += AVX2_DOUBLE_STRIDE * 2;
                sc_ptr += AVX2_DOUBLE_STRIDE * 2;
                r -= AVX2_DOUBLE_STRIDE * 2;
            }
            if (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_load_x1_pd(x_ptr, x0);
                _mm256_load_x1_pd(sc_ptr, s0);

                s0 = _mm256_max_pd(x0, s0);

                _mm256_store_x1_pd(sc_ptr, s0);

                x_ptr += AVX2_DOUBLE_STRIDE;
            }
        }

        copy_aligned_d(stride, s_ptr, y_ptr);
        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_max_unaligned_d(
    const uint n, const uint samples, const uint stride,
    indoubles x_ptr, outdoubles y_ptr) {

    if (stride <= AVX2_DOUBLE_STRIDE * 2) {
        return ag_max_strideleq8_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 3) {
        return ag_max_stride9to11_d(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_DOUBLE_STRIDE * 4) {
        return ag_max_stride13to15_d(n, samples, stride, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint ssize = (stride + AVX2_DOUBLE_REMAIN_MASK) & AVX2_DOUBLE_BATCH_MASK;

    const __m256i mask = _mm256_setmask_pd(stride & AVX2_DOUBLE_REMAIN_MASK);
    const __m256d minf = _mm256_set1_pd(-HUGE_VAL);

    double* s_ptr = (double*)_aligned_malloc(ssize * sizeof(double), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        fill_aligned_d(ssize, -HUGE_VAL, s_ptr);

        for (uint j = 0; j < samples; j++) {
            double* sc_ptr = s_ptr;

            __m256d x0, x1, x2, x3;
            __m256d s0, s1, s2, s3;

            uint r = stride;

            while (r >= AVX2_DOUBLE_STRIDE * 4) {
                _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
                _mm256_load_x4_pd(sc_ptr, s0, s1, s2, s3);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);
                s2 = _mm256_max_pd(x2, s2);
                s3 = _mm256_max_pd(x3, s3);

                _mm256_store_x4_pd(sc_ptr, s0, s1, s2, s3);

                x_ptr += AVX2_DOUBLE_STRIDE * 4;
                sc_ptr += AVX2_DOUBLE_STRIDE * 4;
                r -= AVX2_DOUBLE_STRIDE * 4;
            }
            if (r >= AVX2_DOUBLE_STRIDE * 2) {
                _mm256_loadu_x2_pd(x_ptr, x0, x1);
                _mm256_load_x2_pd(sc_ptr, s0, s1);

                s0 = _mm256_max_pd(x0, s0);
                s1 = _mm256_max_pd(x1, s1);

                _mm256_store_x2_pd(sc_ptr, s0, s1);

                x_ptr += AVX2_DOUBLE_STRIDE * 2;
                sc_ptr += AVX2_DOUBLE_STRIDE * 2;
                r -= AVX2_DOUBLE_STRIDE * 2;
            }
            if (r >= AVX2_DOUBLE_STRIDE) {
                _mm256_loadu_x1_pd(x_ptr, x0);
                _mm256_load_x1_pd(sc_ptr, s0);

                s0 = _mm256_max_pd(x0, s0);

                _mm256_store_x1_pd(sc_ptr, s0);

                x_ptr += AVX2_DOUBLE_STRIDE;
                sc_ptr += AVX2_DOUBLE_STRIDE;
                r -= AVX2_DOUBLE_STRIDE;
            }
            if (r > 0) {
                _mm256_loadu_x1_pd(x_ptr, x0);
                _mm256_load_x1_pd(sc_ptr, s0);
                x0 = _mm256_or_pd(x0, _mm256_andnot_pd(_mm256_castsi256_pd(mask), minf));

                s0 = _mm256_max_pd(x0, s0);

                _mm256_maskstore_x1_pd(sc_ptr, s0, mask);

                x_ptr += r;
            }
        }

        copy_srcaligned_d(stride, s_ptr, y_ptr, mask);
        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Aggregate::Max(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<double>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    if (samples == 0) {
        Initialize::Clear(n * stride, NAN, y);
        return;
    }
    if (samples == 1) {
        Elementwise::Copy(n * stride, x, y);
        return;
    }

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_DOUBLE_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_max_aligned_d(n, samples, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_max_unaligned_d(n, samples, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
