#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"

using namespace System;

#pragma unmanaged

int pool2d_maxpool_n32x_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* s_ptr = (float*)_aligned_malloc((size_t)c * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    __m256 x0, x1, x2, x3;
    __m256 s0, s1, s2, s3;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                const uint ekw = min(iw - isx, kw), ekh = min(ih - isy, kh);

                copy_n32x_s(c, x_ptr + c * (isx + iw * isy), s_ptr);

                for (uint kx = 1 % ekw, ky = 1 / ekw; ky < ekh; kx++, ky += kx / ekw, kx %= ekw) {
                    const uint ix = isx + kx, iy = isy + ky;

                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                    float* sc_ptr = s_ptr;

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE * 4) {
                        _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                        _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                        s0 = _mm256_max_ps(x0, s0);
                        s1 = _mm256_max_ps(x1, s1);
                        s2 = _mm256_max_ps(x2, s2);
                        s3 = _mm256_max_ps(x3, s3);

                        _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                        xc_ptr += AVX2_FLOAT_STRIDE * 4;
                        sc_ptr += AVX2_FLOAT_STRIDE * 4;
                        r -= AVX2_FLOAT_STRIDE * 4;
                    }
                }

                copy_n32x_s(c, s_ptr, y_ptr + c * (ox + ow * oy));
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int pool2d_maxpool_aligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* s_ptr = (float*)_aligned_malloc((size_t)c * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    __m256 x0, x1, x2, x3;
    __m256 s0, s1, s2, s3;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                const uint ekw = min(iw - isx, kw), ekh = min(ih - isy, kh);

                copy_aligned_s(c, x_ptr + c * (isx + iw * isy), s_ptr);

                for (uint kx = 1 % ekw, ky = 1 / ekw; ky < ekh; kx++, ky += kx / ekw, kx %= ekw) {
                    const uint ix = isx + kx, iy = isy + ky;

                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                    float* sc_ptr = s_ptr;

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE * 4) {
                        _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                        _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                        s0 = _mm256_max_ps(x0, s0);
                        s1 = _mm256_max_ps(x1, s1);
                        s2 = _mm256_max_ps(x2, s2);
                        s3 = _mm256_max_ps(x3, s3);

                        _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                        xc_ptr += AVX2_FLOAT_STRIDE * 4;
                        sc_ptr += AVX2_FLOAT_STRIDE * 4;
                        r -= AVX2_FLOAT_STRIDE * 4;
                    }
                    if (r >= AVX2_FLOAT_STRIDE * 2) {
                        _mm256_load_x2_ps(xc_ptr, x0, x1);
                        _mm256_load_x2_ps(sc_ptr, s0, s1);

                        s0 = _mm256_max_ps(x0, s0);
                        s1 = _mm256_max_ps(x1, s1);

                        _mm256_store_x2_ps(sc_ptr, s0, s1);

                        xc_ptr += AVX2_FLOAT_STRIDE * 2;
                        sc_ptr += AVX2_FLOAT_STRIDE * 2;
                        r -= AVX2_FLOAT_STRIDE * 2;
                    }
                    if (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_load_x1_ps(xc_ptr, x0);
                        _mm256_load_x1_ps(sc_ptr, s0);

                        s0 = _mm256_max_ps(x0, s0);

                        _mm256_store_x1_ps(sc_ptr, s0);
                    }
                }

                copy_aligned_s(c, s_ptr, y_ptr + c * (ox + ow * oy));
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

int pool2d_maxpool_unaligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* s_ptr = (float*)_aligned_malloc(((size_t)c + AVX2_FLOAT_STRIDE) * sizeof(float), AVX2_ALIGNMENT);
    if (s_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3;
    __m256 s0, s1, s2, s3;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                const uint ekw = min(iw - isx, kw), ekh = min(ih - isy, kh);

                copy_dstaligned_s(c, x_ptr + c * (isx + iw * isy), s_ptr, mask);

                for (uint kx = 1 % ekw, ky = 1 / ekw; ky < ekh; kx++, ky += kx / ekw, kx %= ekw) {
                    const uint ix = isx + kx, iy = isy + ky;

                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                    float* sc_ptr = s_ptr;

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE * 4) {
                        _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);
                        _mm256_load_x4_ps(sc_ptr, s0, s1, s2, s3);

                        s0 = _mm256_max_ps(x0, s0);
                        s1 = _mm256_max_ps(x1, s1);
                        s2 = _mm256_max_ps(x2, s2);
                        s3 = _mm256_max_ps(x3, s3);

                        _mm256_store_x4_ps(sc_ptr, s0, s1, s2, s3);

                        xc_ptr += AVX2_FLOAT_STRIDE * 4;
                        sc_ptr += AVX2_FLOAT_STRIDE * 4;
                        r -= AVX2_FLOAT_STRIDE * 4;
                    }
                    if (r >= AVX2_FLOAT_STRIDE * 2) {
                        _mm256_loadu_x2_ps(xc_ptr, x0, x1);
                        _mm256_load_x2_ps(sc_ptr, s0, s1);

                        s0 = _mm256_max_ps(x0, s0);
                        s1 = _mm256_max_ps(x1, s1);

                        _mm256_store_x2_ps(sc_ptr, s0, s1);

                        xc_ptr += AVX2_FLOAT_STRIDE * 2;
                        sc_ptr += AVX2_FLOAT_STRIDE * 2;
                        r -= AVX2_FLOAT_STRIDE * 2;
                    }
                    if (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_loadu_x1_ps(xc_ptr, x0);
                        _mm256_load_x1_ps(sc_ptr, s0);

                        s0 = _mm256_max_ps(x0, s0);

                        _mm256_store_x1_ps(sc_ptr, s0);

                        xc_ptr += AVX2_FLOAT_STRIDE;
                        sc_ptr += AVX2_FLOAT_STRIDE;
                        r -= AVX2_FLOAT_STRIDE;
                    }
                    if (r > 0) {
                        _mm256_loadu_x1_ps(xc_ptr, x0);
                        _mm256_load_x1_ps(sc_ptr, s0);

                        s0 = _mm256_max_ps(x0, s0);

                        _mm256_store_x1_ps(sc_ptr, s0);
                    }
                }

                copy_srcaligned_s(c, s_ptr, y_ptr + c * (ox + ow * oy), mask);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    _aligned_free(s_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool2D::MaxPooling(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
    UInt32 sx, UInt32 sy, UInt32 kw, UInt32 kh,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (sx <= 0 || sx > MAX_POOL_STRIDE || sy <= 0 || sy > MAX_POOL_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidPoolStride);
    }
    if ((kw <= 1 && kh <= 1) || kw > MAX_KERNEL_SIZE || kh > MAX_KERNEL_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE || ih > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0) {
        return;
    }

    UInt32 ow = (iw - 1) / sx + 1;
    UInt32 oh = (ih - 1) / sy + 1;

    Util::CheckProdOverflow(n, c, iw, ih);
    Util::CheckProdOverflow(n, c, ow, oh);

    Util::CheckLength(n * c * iw * ih, x);
    Util::CheckLength(n * c * ow * oh, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        ret = pool2d_maxpool_n32x_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pool2d_maxpool_aligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pool2d_maxpool_unaligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}