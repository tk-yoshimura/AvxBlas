#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_copy_s.hpp"

#pragma unmanaged

int pixelshuffle3d_channeltospace_aligned(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if ((cs & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            copy_aligned_s(cs, xc_ptr, y_ptr);
                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_unaligned(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if ((cs & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(cs & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            copy_unaligned_s(cs, xc_ptr, y_ptr, mask);
                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_cs2to3(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if (cs != 2 && cs != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    const __m128i mask = _mm_setmask_ps(cs);

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            x = _mm_loadu_ps(xc_ptr);
                            _mm_maskstore_ps(y_ptr, mask, x);

                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_cs4(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if (cs != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            x = _mm_load_ps(xc_ptr);
                            _mm_stream_ps(y_ptr, x);

                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_cs5to7(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if (cs <= AVX1_FLOAT_STRIDE || cs >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(cs & AVX2_FLOAT_REMAIN_MASK);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            _mm256_loadu_x1_ps(xc_ptr, x);
                            _mm256_maskstore_x1_ps(y_ptr, x, mask);

                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_cs8(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

#ifdef _DEBUG
    if (cs != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0; iz < id; iz++) {
            for (uint cz = 0; cz < s; cz++) {
                for (uint iy = 0; iy < ih; iy++) {
                    for (uint cy = 0; cy < s; cy++) {
                        const float* xc_ptr = x_ptr + cs * (cy + s * cz) + ic * iw * (iy + ih * iz);

                        for (uint ix = 0; ix < iw; ix++) {
                            _mm256_load_x1_ps(xc_ptr, x);
                            _mm256_stream_x1_ps(y_ptr, x);

                            xc_ptr += ic;
                            y_ptr += cs;
                        }
                    }
                }
            }
        }

        x_ptr += ic * iw * ih * id;
    }

    return SUCCESS;
}

int pixelshuffle3d_channeltospace_csleq8(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = oc * s;

    if (cs <= 1) {
        return FAILURE_BADPARAM;
    }
    if (cs < AVX1_FLOAT_STRIDE) {
        return pixelshuffle3d_channeltospace_cs2to3(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }
    if (cs == AVX1_FLOAT_STRIDE) {
        return pixelshuffle3d_channeltospace_cs4(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }
    if (cs < AVX2_FLOAT_STRIDE) {
        return pixelshuffle3d_channeltospace_cs5to7(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }
    if (cs == AVX2_FLOAT_STRIDE) {
        return pixelshuffle3d_channeltospace_cs8(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::PixelShuffle3D::ChannelToSpace(
    UInt32 n, UInt32 ic, UInt32 iw, UInt32 ih, UInt32 id, UInt32 s,
    Array<float>^ xc, Array<float>^ ys) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (s <= 0 || s > MAX_PIXELSHUFFLE_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidStride);
    }
    if ((ic % (s * s * s)) != 0 || ic > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || ic <= 0 || iw <= 0 || ih <= 0 || id <= 0) {
        return;
    }
    if (iw * s > MAX_MAP_SIZE || ih * s > MAX_MAP_SIZE || id * s > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(iw, s);
    Util::CheckProdOverflow(ih, s);
    Util::CheckProdOverflow(id, s);

    UInt32 ow = iw * s, oh = ih * s, od = id * s, oc = ic / (s * s * s);

    Util::CheckProdOverflow(n, ic, iw, ih, id);
    Util::CheckProdOverflow(n, oc, ow, oh, od);

    Util::CheckLength(n * ic * iw * ih * id, xc);
    Util::CheckLength(n * oc * ow * oh * od, ys);

    Util::CheckDuplicateArray(xc, ys);

    if (s == 1) {
        Elementwise::Copy(n * ic * iw * ih * id, xc, ys);
        return;
    }

    const float* x_ptr = (const float*)(xc->Ptr.ToPointer());
    float* y_ptr = (float*)(ys->Ptr.ToPointer());

    int ret = UNEXECUTED;

    const uint cs = oc * s;

    if (cs <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = pixelshuffle3d_channeltospace_csleq8(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }
    else if ((cs & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pixelshuffle3d_channeltospace_aligned(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pixelshuffle3d_channeltospace_unaligned(n, ic, oc, iw, ow, ih, oh, id, od, s, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}