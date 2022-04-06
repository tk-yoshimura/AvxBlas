#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"
#include "../Inline/inline_pooliter_s.hpp"

using namespace System;

#pragma unmanaged

int pool3d_maxunpool_n32x_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx != kw || sy != kh || sz != kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_n32x_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_n32x_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx >= kw && sy >= kh && sz >= kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_add_n32x_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_n32x_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx < kw || sy < kh || sz < kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_n32x_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_aligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx != kw || sy != kh || sz != kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_aligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_aligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx >= kw && sy >= kh && sz >= kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_add_aligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_aligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx < kw || sy < kh || sz < kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_aligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz))
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}


int pool3d_maxunpool_unaligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx != kw || sy != kh || sz != kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_unaligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz)),
                                    mask
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_unaligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx >= kw && sy >= kh && sz >= kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_add_unaligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz)),
                                    mask
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int pool3d_maxunpool_unaligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    const uint ih, const uint oh, const uint sy, const uint kh,
    const uint id, const uint od, const uint sz, const uint kd,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx < kw || sy < kh || sz < kd)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    zeroset_s(n * c * iw * ih * id, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint oz = 0, isz = 0; oz < od; oz++, isz += sz) {
            for (uint oy = 0, isy = 0; oy < oh; oy++, isy += sy) {
                for (uint ox = 0, isx = 0; ox < ow; ox++, isx += sx) {
                    for (uint kz = 0, iz = isz; kz < kd && iz < id; kz++, iz++) {
                        for (uint ky = 0, iy = isy; ky < kh && iy < ih; ky++, iy++) {
                            for (uint kx = 0, ix = isx; kx < kw && ix < iw; kx++, ix++) {
                                maxunpooliter_unaligned_s(c,
                                    x_ptr + c * (ix + iw * (iy + ih * iz)), y_ptr + c * (ox + ow * (oy + oh * oz)),
                                    dy_ptr + c * (ox + ow * (oy + oh * oz)), dx_ptr + c * (ix + iw * (iy + ih * iz)),
                                    mask
                                );
                            }
                        }
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
        dx_ptr += c * iw * ih * id;
        dy_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool3D::MaxUnpooling(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
    UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
    Array<float>^ x, Array<float>^ y, Array<float>^ dy, Array<float>^ dx) {

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
    Util::CheckLength(n * c * iw * ih * id, dx);
    Util::CheckLength(n * c * ow * oh * od, dy);

    Util::CheckDuplicateArray(x, y, dy, dx);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* y_ptr = (const float*)(y->Ptr.ToPointer());
    const float* dy_ptr = (const float*)(dy->Ptr.ToPointer());
    float* dx_ptr = (float*)(dx->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
        if (sx == kw && sy == kh && sz == kd) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx = kx and sy = ky and sz = kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_n32x_seqk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw || sy < kh || sz < kd) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx < kx or sy < kh or sz < kd");
#endif // _DEBUG

            ret = pool3d_maxunpool_n32x_sltk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx >= kw && sy >= kh && sz >= kd) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx >= kx and sy >= ky and sz >= kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_n32x_sgtk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        if (sx == kw && sy == kh && sz == kd) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx = kx and sy = ky and sz = kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_aligned_seqk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw || sy < kh || sz < kd) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx < kx or sy < kh or sz < kd");
#endif // _DEBUG

            ret = pool3d_maxunpool_aligned_sltk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx >= kw && sy >= kh && sz >= kd) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx >= kx and sy >= ky and sz >= kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_aligned_sgtk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }
    else {
        if (sx == kw && sy == kh && sz == kd) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx = kx and sy = ky and sz = kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_unaligned_seqk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw || sy < kh || sz < kd) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx < kx or sy < kh or sz < kd");
#endif // _DEBUG

            ret = pool3d_maxunpool_unaligned_sltk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx >= kw && sy >= kh && sz >= kd) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx >= kx and sy >= ky and sz >= kz");
#endif // _DEBUG

            ret = pool3d_maxunpool_unaligned_sgtk_s(n, c, iw, ow, sx, kw, ih, oh, sy, kh, id, od, sz, kd, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }

    Util::AssertReturnCode(ret);
}