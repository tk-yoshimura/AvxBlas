#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_pooliter_s.hpp"

using namespace System;

#pragma unmanaged

int pool1d_maxpool_n32x_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            copy_n32x_s(c, x_ptr + c * isx, y_ptr + c * ox);

            for (uint kx = 1, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                maxpooliter_n32x_s(c, x_ptr + c * ix, y_ptr + c * ox);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxpool_aligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            copy_aligned_s(c, x_ptr + c * isx, y_ptr + c * ox);

            for (uint kx = 1, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                maxpooliter_aligned_s(c, x_ptr + c * ix, y_ptr + c * ox);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxpool_unaligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            copy_unaligned_s(c, x_ptr + c * isx, y_ptr + c * ox, mask);

            for (uint kx = 1, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                maxpooliter_unaligned_s(c, x_ptr + c * ix, y_ptr + c * ox, mask);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool1D::MaxPooling(
    UInt32 n, UInt32 c, UInt32 iw,
    UInt32 sx, UInt32 kw,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (sx <= 0 || sx > MAX_POOL_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidPoolStride);
    }
    if (kw <= 1 || kw > MAX_KERNEL_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || c <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = (iw - 1) / sx + 1;

    Util::CheckProdOverflow(n, c, iw);
    Util::CheckProdOverflow(n, c, ow);

    Util::CheckLength(n * c * iw, x);
    Util::CheckLength(n * c * ow, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        ret = pool1d_maxpool_n32x_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pool1d_maxpool_aligned_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pool1d_maxpool_unaligned_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}