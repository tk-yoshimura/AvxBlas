#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_prod_s.hpp"
#include "../Inline/inline_fill_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include <memory.h>
#include <math.h>

using namespace System;

#pragma unmanaged

int ag_prod_stride1_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 1) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = samples & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);

    for (uint i = 0; i < n; i++) {
        __m256 x;
        __m256 s = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                x = _mm256_load_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                x = _mm256_loadu_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r > 0) {
            x = _mm256_maskload_ps(x_ptr, mask);
            x = _mm256_or_ps(x, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s = _mm256_mul_ps(x, s);

            x_ptr += r;
        }

        float y = _mm256_prod8to1_ps(s);

        *y_ptr = y;

        y_ptr += 1;
    }

    return SUCCESS;
}

int ag_prod_stride2_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = (2 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m128i mask2 = _mm_setmask_ps(2);

    for (uint i = 0; i < n; i++) {
        __m256 x;
        __m256 s = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                x = _mm256_load_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                x = _mm256_loadu_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        if (r > 0) {
            x = _mm256_maskload_ps(x_ptr, mask);
            x = _mm256_or_ps(x, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s = _mm256_mul_ps(x, s);

            x_ptr += r * 2;
        }

        __m128 y = _mm256_prod8to2_ps(s);

        _mm_maskstore_ps(y_ptr, mask2, y);

        y_ptr += 2;
    }

    return SUCCESS;
}

int ag_prod_stride3_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = (3 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m128i mask3 = _mm_setmask_ps(3);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2;
        __m256 s0 = ones, s1 = ones, s2 = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r > AVX2_FLOAT_STRIDE * 2 / 3) {
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
        }
        else if (r > AVX2_FLOAT_STRIDE / 3) {
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
            x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
        }
        else if (r > 0) {
            _mm256_maskload_x1_ps(x_ptr, x0, mask);
            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
        }

        __m128 y = _mm256_prod24to3_ps(s0, s1, s2);

        _mm_maskstore_ps(y_ptr, mask3, y);

        x_ptr += 3 * r;
        y_ptr += 3;
    }

    return SUCCESS;
}

int ag_prod_stride4_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = (4 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask4 = _mm256_setmask_ps(4);

    for (uint i = 0; i < n; i++) {
        __m256 x;
        __m256 s = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 4) {
                x = _mm256_load_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 4;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 4) {
                x = _mm256_loadu_ps(x_ptr);

                s = _mm256_mul_ps(x, s);

                x_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE / 4;
            }
        }
        if (r > 0) {
            x = _mm256_maskload_ps(x_ptr, mask4);
            x = _mm256_or_ps(x, _mm256_andnot_ps(_mm256_castsi256_ps(mask4), ones));

            s = _mm256_mul_ps(x, s);

            x_ptr += AVX2_FLOAT_STRIDE / 2;
        }

        __m128 y = _mm256_prod8to4_ps(s);

        _mm_stream_ps(y_ptr, y);

        y_ptr += AVX2_FLOAT_STRIDE / 2;
    }

    return SUCCESS;
}

int ag_prod_stride5_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 5) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = (5 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask5 = _mm256_setmask_ps(5);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2, x3, x4;
        __m256 s0 = ones, s1 = ones, s2 = ones, s3 = ones, s4 = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x5_ps(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);
                s3 = _mm256_mul_ps(x3, s3);
                s4 = _mm256_mul_ps(x4, s4);

                x_ptr += AVX2_FLOAT_STRIDE * 5;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x5_ps(x_ptr, x0, x1, x2, x3, x4);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);
                s3 = _mm256_mul_ps(x3, s3);
                s4 = _mm256_mul_ps(x4, s4);

                x_ptr += AVX2_FLOAT_STRIDE * 5;
                r -= AVX2_FLOAT_STRIDE;
            }
        }
        if (r > AVX2_FLOAT_STRIDE * 4 / 5) {
            _mm256_maskload_x5_ps(x_ptr, x0, x1, x2, x3, x4, mask);
            x4 = _mm256_or_ps(x4, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
            s3 = _mm256_mul_ps(x3, s3);
            s4 = _mm256_mul_ps(x4, s4);
        }
        else if (r > AVX2_FLOAT_STRIDE * 3 / 5) {
            _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);
            x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
            s3 = _mm256_mul_ps(x3, s3);
        }
        else if (r > AVX2_FLOAT_STRIDE * 2 / 5) {
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
        }
        else if (r > AVX2_FLOAT_STRIDE / 5) {
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
            x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
        }
        else if (r > 0) {
            _mm256_maskload_x1_ps(x_ptr, x0, mask);
            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
        }

        __m256 y = _mm256_prod40to5_ps(s0, s1, s2, s3, s4);

        _mm256_maskstore_ps(y_ptr, mask5, y);

        x_ptr += 5 * r;
        y_ptr += 5;
    }

    return SUCCESS;
}

int ag_prod_stride6_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != 6) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const uint maskn = (6 * samples) & AVX2_FLOAT_REMAIN_MASK;
    const __m256i mask = _mm256_setmask_ps(maskn);
    const __m256i mask6 = _mm256_setmask_ps(6);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2;
        __m256 s0 = ones, s1 = ones, s2 = ones;

        uint r = samples;

        if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        else {
            while (r >= AVX2_FLOAT_STRIDE / 2) {
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);

                x_ptr += AVX2_FLOAT_STRIDE * 3;
                r -= AVX2_FLOAT_STRIDE / 2;
            }
        }
        if (r > AVX2_FLOAT_STRIDE * 2 / 6) {
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
        }
        else if (r > AVX2_FLOAT_STRIDE / 6) {
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
            x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
        }
        else if (r > 0) {
            _mm256_maskload_x1_ps(x_ptr, x0, mask);
            x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
        }

        __m256 y = _mm256_prod24to6_ps(s0, s1, s2);

        _mm256_maskstore_ps(y_ptr, mask6, y);

        x_ptr += 6 * r;
        y_ptr += 6;
    }

    return SUCCESS;
}

int ag_prod_stride7_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride != 7) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const __m256i mask = _mm256_setmask_ps(7);

    for (uint i = 0; i < n; i++) {
        __m256 x;
        __m256 s = ones;

        for (uint j = 0; j < samples; j++) {
            x = _mm256_maskload_ps(x_ptr, mask);
            x = _mm256_or_ps(x, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s = _mm256_mul_ps(x, s);

            x_ptr += 7;
        }

        _mm256_maskstore_ps(y_ptr, mask, s);

        y_ptr += 7;
    }

    return SUCCESS;
}

int ag_prod_stride8_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);

    for (uint i = 0; i < n; i++) {
        __m256 x;
        __m256 s = ones;

        for (uint j = 0; j < samples; j++) {
            x = _mm256_load_ps(x_ptr);

            s = _mm256_mul_ps(x, s);

            x_ptr += AVX2_FLOAT_STRIDE;
        }

        _mm256_stream_ps(y_ptr, s);

        y_ptr += AVX2_FLOAT_STRIDE;
    }

    return SUCCESS;
}

int ag_prod_stride9to15_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE || stride >= AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1;
        __m256 s0 = ones, s1 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x2_ps(x_ptr, x0, x1, mask);
            x1 = _mm256_or_ps(x1, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);

            x_ptr += stride;
        }

        _mm256_maskstore_x2_ps(y_ptr, s0, s1, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_prod_stride16_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 2) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1;
        __m256 s0 = ones, s1 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x2_ps(x_ptr, x0, x1);

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);

            x_ptr += AVX2_FLOAT_STRIDE * 2;
        }

        _mm256_stream_x2_ps(y_ptr, s0, s1);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
    }

    return SUCCESS;
}

int ag_prod_stride17to23_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 2 || stride >= AVX2_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2;
        __m256 s0 = ones, s1 = ones, s2 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x3_ps(x_ptr, x0, x1, x2, mask);
            x2 = _mm256_or_ps(x2, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);

            x_ptr += stride;
        }

        _mm256_maskstore_x3_ps(y_ptr, s0, s1, s2, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_prod_stride24_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 3) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2;
        __m256 s0 = ones, s1 = ones, s2 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x3_ps(x_ptr, x0, x1, x2);

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);

            x_ptr += AVX2_FLOAT_STRIDE * 3;
        }

        _mm256_stream_x3_ps(y_ptr, s0, s1, s2);

        y_ptr += AVX2_FLOAT_STRIDE * 3;
    }

    return SUCCESS;
}

int ag_prod_stride25to31_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (stride <= AVX2_FLOAT_STRIDE * 3 || stride >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);
    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2, x3;
        __m256 buf0 = ones, buf1 = ones, buf2 = ones, buf3 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_maskload_x4_ps(x_ptr, x0, x1, x2, x3, mask);
            x3 = _mm256_or_ps(x3, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

            buf0 = _mm256_mul_ps(x0, buf0);
            buf1 = _mm256_mul_ps(x1, buf1);
            buf2 = _mm256_mul_ps(x2, buf2);
            buf3 = _mm256_mul_ps(x3, buf3);

            x_ptr += stride;
        }

        _mm256_maskstore_x4_ps(y_ptr, buf0, buf1, buf2, buf3, mask);

        y_ptr += stride;
    }

    return SUCCESS;
}

int ag_prod_stride32_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((stride != AVX2_FLOAT_STRIDE * 4) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 ones = _mm256_set1_ps(1);

    for (uint i = 0; i < n; i++) {
        __m256 x0, x1, x2, x3;
        __m256 s0 = ones, s1 = ones, s2 = ones, s3 = ones;

        for (uint j = 0; j < samples; j++) {
            _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

            s0 = _mm256_mul_ps(x0, s0);
            s1 = _mm256_mul_ps(x1, s1);
            s2 = _mm256_mul_ps(x2, s2);
            s3 = _mm256_mul_ps(x3, s3);

            x_ptr += AVX2_FLOAT_STRIDE * 4;
        }

        _mm256_stream_x4_ps(y_ptr, s0, s1, s2, s3);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
    }

    return SUCCESS;
}

int ag_prod_strideleq8_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == 1) {
        return ag_prod_stride1_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 2) {
        return ag_prod_stride2_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 3) {
        return ag_prod_stride3_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 4) {
        return ag_prod_stride4_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 5) {
        return ag_prod_stride5_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 6) {
        return ag_prod_stride6_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == 7) {
        return ag_prod_stride7_s(n, samples, stride, x_ptr, y_ptr);
    }
    else if (stride == AVX2_FLOAT_STRIDE) {
        return ag_prod_stride8_s(n, samples, stride, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

int ag_prod_aligned_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride == AVX2_FLOAT_STRIDE) {
        return ag_prod_stride8_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 2) {
        return ag_prod_stride16_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 3) {
        return ag_prod_stride24_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride == AVX2_FLOAT_STRIDE * 4) {
        return ag_prod_stride32_s(n, samples, stride, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if (((stride & AVX2_FLOAT_REMAIN_MASK) != 0) || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* s_ptr = (float*)_aligned_malloc((size_t)stride * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        fill_aligned_s(stride, 1, s_ptr);

        for (uint j = 0; j < samples; j++) {
            float* sc_ptr = s_ptr;

            __m256 x0, x1, x2, x3;
            __m256 s0, s1, s2, s3;

            uint r = stride;

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);
                s3 = _mm256_mul_ps(x3, s3);

                _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                sc_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_load_x2_ps(x_ptr, x0, x1);
                _mm256_load_x2_ps(sc_ptr, s0, s1);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);

                _mm256_store_x2_ps(sc_ptr, s0, s1);

                x_ptr += AVX2_FLOAT_STRIDE * 2;
                sc_ptr += AVX2_FLOAT_STRIDE * 2;
                r -= AVX2_FLOAT_STRIDE * 2;
            }
            if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(x_ptr, x0);
                _mm256_load_x1_ps(sc_ptr, s0);

                s0 = _mm256_mul_ps(x0, s0);

                _mm256_store_x1_ps(sc_ptr, s0);

                x_ptr += AVX2_FLOAT_STRIDE;
            }
        }

        copy_aligned_s(stride, s_ptr, y_ptr);
        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int ag_prod_unaligned_s(
    const uint n, const uint samples, const uint stride,
    infloats x_ptr, outfloats y_ptr) {

    if (stride <= AVX2_FLOAT_STRIDE) {
        return ag_prod_strideleq8_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 2) {
        return ag_prod_stride9to15_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 3) {
        return ag_prod_stride17to23_s(n, samples, stride, x_ptr, y_ptr);
    }
    if (stride <= AVX2_FLOAT_STRIDE * 4) {
        return ag_prod_stride25to31_s(n, samples, stride, x_ptr, y_ptr);
    }

#ifdef _DEBUG
    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint ssize = (stride + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    const __m256i mask = _mm256_setmask_ps(stride & AVX2_FLOAT_REMAIN_MASK);
    const __m256 ones = _mm256_set1_ps(1);

    float* s_ptr = (float*)_aligned_malloc(ssize * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        fill_aligned_s(ssize, 1, s_ptr);

        for (uint j = 0; j < samples; j++) {
            float* sc_ptr = s_ptr;

            __m256 x0, x1, x2, x3;
            __m256 s0, s1, s2, s3;

            uint r = stride;

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);
                s2 = _mm256_mul_ps(x2, s2);
                s3 = _mm256_mul_ps(x3, s3);

                _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                sc_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_loadu_x2_ps(x_ptr, x0, x1);
                _mm256_load_x2_ps(sc_ptr, s0, s1);

                s0 = _mm256_mul_ps(x0, s0);
                s1 = _mm256_mul_ps(x1, s1);

                _mm256_store_x2_ps(sc_ptr, s0, s1);

                x_ptr += AVX2_FLOAT_STRIDE * 2;
                sc_ptr += AVX2_FLOAT_STRIDE * 2;
                r -= AVX2_FLOAT_STRIDE * 2;
            }
            if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x1_ps(x_ptr, x0);
                _mm256_load_x1_ps(sc_ptr, s0);

                s0 = _mm256_mul_ps(x0, s0);

                _mm256_store_x1_ps(sc_ptr, s0);

                x_ptr += AVX2_FLOAT_STRIDE;
                sc_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
            if (r > 0) {
                _mm256_loadu_x1_ps(x_ptr, x0);
                _mm256_load_x1_ps(sc_ptr, s0);
                x0 = _mm256_or_ps(x0, _mm256_andnot_ps(_mm256_castsi256_ps(mask), ones));

                s0 = _mm256_mul_ps(x0, s0);

                _mm256_maskstore_x1_ps(sc_ptr, s0, mask);

                x_ptr += r;
            }
        }

        copy_srcaligned_s(stride, s_ptr, y_ptr, mask);
        y_ptr += stride;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Aggregate::Prod(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || samples <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, samples, stride);

    Util::CheckLength(n * samples * stride, x);
    Util::CheckLength(n * stride, y);

    if (samples == 1) {
        Elementwise::Copy(n * stride, x, y);
        return;
    }

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = ag_prod_aligned_s(n, samples, stride, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = ag_prod_unaligned_s(n, samples, stride, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
