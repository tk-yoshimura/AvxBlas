#pragma once
#pragma unmanaged

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#include "../utils.h"
#include "../constants.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

__forceinline void maxpooliter_n32x_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);
        y3 = _mm256_max_ps(x3, y3);

        _mm256_store_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void maxpooliter_aligned_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);
        y3 = _mm256_max_ps(x3, y3);

        _mm256_store_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if(r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(y_ptr, y0, y1, y2);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);

        _mm256_store_x3_ps(y_ptr, y0, y1, y2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);
        _mm256_load_x2_ps(y_ptr, y0, y1);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);

        _mm256_store_x2_ps(y_ptr, y0, y1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);
        _mm256_load_x1_ps(y_ptr, y0);

        y0 = _mm256_max_ps(x0, y0);

        _mm256_store_x1_ps(y_ptr, y0);
    }
}

__forceinline void maxpooliter_unaligned_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);
        y3 = _mm256_max_ps(x3, y3);

        _mm256_storeu_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);
        y3 = _mm256_max_ps(x3, y3);

        _mm256_maskstore_x4_ps(y_ptr, y0, y1, y2, y3, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
        _mm256_loadu_x3_ps(y_ptr, y0, y1, y2);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);
        y2 = _mm256_max_ps(x2, y2);

        _mm256_maskstore_x3_ps(y_ptr, y0, y1, y2, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x2_ps(x_ptr, x0, x1);
        _mm256_loadu_x2_ps(y_ptr, y0, y1);

        y0 = _mm256_max_ps(x0, y0);
        y1 = _mm256_max_ps(x1, y1);

        _mm256_maskstore_x2_ps(y_ptr, y0, y1, mask);
    }
    else if (r > 0) {
        _mm256_loadu_x1_ps(x_ptr, x0);
        _mm256_loadu_x1_ps(y_ptr, y0);

        y0 = _mm256_max_ps(x0, y0);

        _mm256_maskstore_x1_ps(y_ptr, y0, mask);
    }
}

__forceinline void avgpooliter_n32x_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);
        y3 = _mm256_add_ps(x3, y3);

        _mm256_store_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void avgpooliter_aligned_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);
        y3 = _mm256_add_ps(x3, y3);

        _mm256_store_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(y_ptr, y0, y1, y2);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);

        _mm256_store_x3_ps(y_ptr, y0, y1, y2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);
        _mm256_load_x2_ps(y_ptr, y0, y1);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);

        _mm256_store_x2_ps(y_ptr, y0, y1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);
        _mm256_load_x1_ps(y_ptr, y0);

        y0 = _mm256_add_ps(x0, y0);

        _mm256_store_x1_ps(y_ptr, y0);
    }
}

__forceinline void avgpooliter_unaligned_s(
    const uint n, 
    infloats x_ptr, outfloats y_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);
        y3 = _mm256_add_ps(x3, y3);

        _mm256_storeu_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);
        y3 = _mm256_add_ps(x3, y3);

        _mm256_maskstore_x4_ps(y_ptr, y0, y1, y2, y3, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
        _mm256_loadu_x3_ps(y_ptr, y0, y1, y2);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);
        y2 = _mm256_add_ps(x2, y2);

        _mm256_maskstore_x3_ps(y_ptr, y0, y1, y2, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x2_ps(x_ptr, x0, x1);
        _mm256_loadu_x2_ps(y_ptr, y0, y1);

        y0 = _mm256_add_ps(x0, y0);
        y1 = _mm256_add_ps(x1, y1);

        _mm256_maskstore_x2_ps(y_ptr, y0, y1, mask);
    }
    else if (r > 0) {
        _mm256_loadu_x1_ps(x_ptr, x0);
        _mm256_loadu_x1_ps(y_ptr, y0);

        y0 = _mm256_add_ps(x0, y0);

        _mm256_maskstore_x1_ps(y_ptr, y0, mask);
    }
}

__forceinline void maxunpooliter_n32x_s(
    const uint n, 
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_load_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
        dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

        _mm256_stream_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void maxunpooliter_aligned_s(
    const uint n, 
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_load_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
        dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

        _mm256_stream_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(y_ptr, y0, y1, y2);
        _mm256_load_x3_ps(dy_ptr, dy0, dy1, dy2);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));

        _mm256_stream_x3_ps(dx_ptr, dx0, dx1, dx2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);
        _mm256_load_x2_ps(y_ptr, y0, y1);
        _mm256_load_x2_ps(dy_ptr, dy0, dy1);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

        _mm256_stream_x2_ps(dx_ptr, dx0, dx1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);
        _mm256_load_x1_ps(y_ptr, y0);
        _mm256_load_x1_ps(dy_ptr, dy0);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

        _mm256_stream_x1_ps(dx_ptr, dx0);
    }
}

__forceinline void maxunpooliter_unaligned_s(
    const uint n, 
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_loadu_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
        dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

        _mm256_storeu_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_loadu_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
        dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

        _mm256_maskstore_x4_ps(dx_ptr, dx0, dx1, dx2, dx3, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
        _mm256_loadu_x3_ps(y_ptr, y0, y1, y2);
        _mm256_loadu_x3_ps(dy_ptr, dy0, dy1, dy2);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
        dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));

        _mm256_maskstore_x3_ps(dx_ptr, dx0, dx1, dx2, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x2_ps(x_ptr, x0, x1);
        _mm256_loadu_x2_ps(y_ptr, y0, y1);
        _mm256_loadu_x2_ps(dy_ptr, dy0, dy1);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
        dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

        _mm256_maskstore_x2_ps(dx_ptr, dx0, dx1, mask);
    }
    else if (r > 0) {
        _mm256_loadu_x1_ps(x_ptr, x0);
        _mm256_loadu_x1_ps(y_ptr, y0);
        _mm256_loadu_x1_ps(dy_ptr, dy0);

        dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

        _mm256_maskstore_x1_ps(dx_ptr, dx0, mask);
    }
}

__forceinline void maxunpooliter_add_n32x_s(
    const uint n,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_load_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);
        _mm256_load_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
        dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

        _mm256_store_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void maxunpooliter_add_aligned_s(
    const uint n,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_load_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);
        _mm256_load_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
        dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

        _mm256_store_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_load_x3_ps(x_ptr, x0, x1, x2);
        _mm256_load_x3_ps(y_ptr, y0, y1, y2);
        _mm256_load_x3_ps(dy_ptr, dy0, dy1, dy2);
        _mm256_load_x3_ps(dx_ptr, dx0, dx1, dx2);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));

        _mm256_store_x3_ps(dx_ptr, dx0, dx1, dx2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);
        _mm256_load_x2_ps(y_ptr, y0, y1);
        _mm256_load_x2_ps(dy_ptr, dy0, dy1);
        _mm256_load_x2_ps(dx_ptr, dx0, dx1);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));

        _mm256_store_x2_ps(dx_ptr, dx0, dx1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);
        _mm256_load_x1_ps(y_ptr, y0);
        _mm256_load_x1_ps(dy_ptr, dy0);
        _mm256_load_x1_ps(dx_ptr, dx0);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));

        _mm256_store_x1_ps(dx_ptr, dx0);
    }
}

__forceinline void maxunpooliter_add_unaligned_s(
    const uint n,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_loadu_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);
        _mm256_loadu_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
        dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

        _mm256_storeu_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        dx_ptr += AVX2_FLOAT_STRIDE * 4;
        dy_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);
        _mm256_loadu_x4_ps(dy_ptr, dy0, dy1, dy2, dy3);
        _mm256_loadu_x4_ps(dx_ptr, dx0, dx1, dx2, dx3);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
        dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

        _mm256_maskstore_x4_ps(dx_ptr, dx0, dx1, dx2, dx3, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
        _mm256_loadu_x3_ps(y_ptr, y0, y1, y2);
        _mm256_loadu_x3_ps(dy_ptr, dy0, dy1, dy2);
        _mm256_loadu_x3_ps(dx_ptr, dx0, dx1, dx2);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
        dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));

        _mm256_maskstore_x3_ps(dx_ptr, dx0, dx1, dx2, mask);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x2_ps(x_ptr, x0, x1);
        _mm256_loadu_x2_ps(y_ptr, y0, y1);
        _mm256_loadu_x2_ps(dy_ptr, dy0, dy1);
        _mm256_loadu_x2_ps(dx_ptr, dx0, dx1);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
        dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));

        _mm256_maskstore_x2_ps(dx_ptr, dx0, dx1, mask);
    }
    else if (r > 0) {
        _mm256_loadu_x1_ps(x_ptr, x0);
        _mm256_loadu_x1_ps(y_ptr, y0);
        _mm256_loadu_x1_ps(dy_ptr, dy0);
        _mm256_loadu_x1_ps(dx_ptr, dx0);

        dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));

        _mm256_maskstore_x1_ps(dx_ptr, dx0, mask);
    }
}