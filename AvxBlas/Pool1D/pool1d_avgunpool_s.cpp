#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_pooliter_s.hpp"

using namespace System;

#pragma unmanaged

int pool1d_avgunpool_n32x_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx != kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_n32x_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_n32x_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx >= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                avgpooliter_n32x_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_n32x_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx <= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_n32x_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_aligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx != kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_aligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_aligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx >= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                avgpooliter_aligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_aligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx <= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_aligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}


int pool1d_avgunpool_unaligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx != kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_unaligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix, mask);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_unaligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx >= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                avgpooliter_unaligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix, mask);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_avgunpool_unaligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx <= kw)
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                copy_unaligned_s(c, dy_ptr + c * ox, dx_ptr + c * ix, mask);
            }
        }

        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool1D::AverageUnpooling(
    UInt32 n, UInt32 c, UInt32 iw,
    UInt32 sx, UInt32 kw,
    Array<float>^ dy, Array<float>^ dx) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (sx <= 0 || sx > MAX_POOL_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidStride);
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

    Util::CheckLength(n * c * iw, dx);
    Util::CheckLength(n * c * ow, dy);

    Util::CheckDuplicateArray(dy, dx);

    const float* dy_ptr = (const float*)(dy->Ptr.ToPointer());
    float* dx_ptr = (float*)(dx->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx = kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_n32x_seqk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx < kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_n32x_sltk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx > kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_n32x_sgtk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx = kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_aligned_seqk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx < kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_aligned_sltk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx > kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_aligned_sgtk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
    }
    else {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx = kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_unaligned_seqk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx < kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_unaligned_sltk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx > kx");
#endif // _DEBUG

            ret = pool1d_avgunpool_unaligned_sgtk_s(n, c, iw, ow, sx, kw, dy_ptr, dx_ptr);
        }
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * iw, dx, 1.0f / kw, dx);
}