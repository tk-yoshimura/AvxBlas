#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_pooliter_s.hpp"

using namespace System;

#pragma unmanaged

int pool3d_avgpool_n32x_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    copy_n32x_s(c, x_ptr + c * (isx + iw * (isy + ih * isz)), y_ptr + c * (ox + ow * (oy + oh * oz)));

                    for (uint kx = 1 % kw, ky = (1 / kw) % kh, kz = 1 / (kw * kh); kz < kd; kx++, ky += kx / kw, kz += ky / kh, kx %= kw, ky %= kh) {
                        const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1), iz = min(isz + kz, id - 1);

                        avgpooliter_n32x_s(c, x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)));
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_avgpool_aligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    copy_aligned_s(c, x_ptr + c * (isx + iw * (isy + ih * isz)), y_ptr + c * (ox + ow * (oy + oh * oz)));

                    for (uint kx = 1 % kw, ky = (1 / kw) % kh, kz = 1 / (kw * kh); kz < kd; kx++, ky += kx / kw, kz += ky / kh, kx %= kw, ky %= kh) {
                        const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1), iz = min(isz + kz, id - 1);

                        avgpooliter_aligned_s(c, x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)));
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_avgpool_unaligned_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    copy_unaligned_s(c, x_ptr + c * (isx + iw * (isy + ih * isz)), y_ptr + c * (ox + ow * (oy + oh * oz)), mask);

                    for (uint kx = 1 % kw, ky = (1 / kw) % kh, kz = 1 / (kw * kh); kz < kd; kx++, ky += kx / kw, kz += ky / kh, kx %= kw, ky %= kh) {
                        const uint ix = min(isx + kx, iw - 1), iy = min(isy + ky, ih - 1), iz = min(isz + kz, id - 1);

                        avgpooliter_unaligned_s(c, x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)), mask);
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool3D::AveragePooling(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
    UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (sx <= 0 || sx > MAX_POOL_STRIDE || sy <= 0 || sy > MAX_POOL_STRIDE || sz <= 0 || sz > MAX_POOL_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidPoolStride);
    }
    if ((kw <= 1 && kh <= 1 && kd <= 1) || kw > MAX_KERNEL_SIZE || kh > MAX_KERNEL_SIZE || kd > MAX_KERNEL_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE || ih > MAX_MAP_SIZE || id > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0 || id <= 0) {
        return;
    }

    UInt32 ow = (iw - 1) / sx + 1;
    UInt32 oh = (ih - 1) / sy + 1;
    UInt32 od = (id - 1) / sz + 1;

    Util::CheckProdOverflow(n, c, iw, ih, id);
    Util::CheckProdOverflow(n, c, ow, oh, od);

    Util::CheckLength(n * c * iw * ih * id, x);
    Util::CheckLength(n * c * ow * oh * od, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        ret = pool3d_avgpool_n32x_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pool3d_avgpool_aligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pool3d_avgpool_unaligned_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr);
    }

    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh * od, y, 1.0f / (kw * kh * kd), y);
}