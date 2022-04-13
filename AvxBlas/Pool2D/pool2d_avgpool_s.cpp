#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_pooliter_s.hpp"

using namespace System;

#pragma unmanaged

int pool2d_avgpool_n32x_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                copy_n32x_s(c, x_ptr + c * (isx + iw * isy), y_ptr + c * (ox + ow * oy));

                for (uint kx = 1 % kw, ky = 1 / kw; ky < kh; kx++, ky += kx / kw, kx %= kw) {
                    const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1);

                    avgpooliter_n32x_s(c, x_ptr + c * (ix + iw * iy), y_ptr + c * (ox + ow * oy));
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int pool2d_avgpool_aligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                copy_aligned_s(c, x_ptr + c * (isx + iw * isy), y_ptr + c * (ox + ow * oy));

                for (uint kx = 1 % kw, ky = 1 / kw; ky < kh; kx++, ky += kx / kw, kx %= kw) {
                    const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1);

                    avgpooliter_aligned_s(c, x_ptr + c * (ix + iw * iy), y_ptr + c * (ox + ow * oy));
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int pool2d_avgpool_unaligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
            for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                copy_unaligned_s(c, x_ptr + c * (isx + iw * isy), y_ptr + c * (ox + ow * oy), mask);

                for (uint kx = 1 % kw, ky = 1 / kw; ky < kh; kx++, ky += kx / kw, kx %= kw) {
                    const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1);

                    avgpooliter_unaligned_s(c, x_ptr + c * (ix + iw * iy), y_ptr + c * (ox + ow * oy), mask);
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool2D::AveragePooling(
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
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidStride);
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

        ret = pool2d_avgpool_n32x_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pool2d_avgpool_aligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pool2d_avgpool_unaligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh, y, 1.0f / (kw * kh), y);
}