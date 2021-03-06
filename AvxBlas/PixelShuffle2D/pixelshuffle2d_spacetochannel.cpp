#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_copy_s.hpp"

#pragma unmanaged

int pixelshuffle2d_spacetochannel_aligned(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if ((cs & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    copy_aligned_s(cs, x_ptr, yc_ptr);
                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_unaligned(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if ((cs & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(cs & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    copy_unaligned_s(cs, x_ptr, yc_ptr, mask);
                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_cs2to3(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if (cs != 2 && cs != 3) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    const __m128i mask = _mm_setmask_ps(cs);

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    x = _mm_loadu_ps(x_ptr);
                    _mm_maskstore_ps(yc_ptr, mask, x);

                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_cs4(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if (cs != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    x = _mm_load_ps(x_ptr);
                    _mm_stream_ps(yc_ptr, x);

                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_cs5to7(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if (cs <= AVX1_FLOAT_STRIDE || cs >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(cs & AVX2_FLOAT_REMAIN_MASK);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    _mm256_loadu_x1_ps(x_ptr, x);
                    _mm256_maskstore_x1_ps(yc_ptr, x, mask);

                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_cs8(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

#ifdef _DEBUG
    if (cs != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint oy = 0; oy < oh; oy++) {
            for (uint cy = 0; cy < s; cy++) {
                float* yc_ptr = y_ptr + cs * cy + oc * ow * oy;

                for (uint ox = 0; ox < ow; ox++) {
                    _mm256_load_x1_ps(x_ptr, x);
                    _mm256_stream_x1_ps(yc_ptr, x);

                    x_ptr += cs;
                    yc_ptr += oc;
                }
            }
        }

        y_ptr += oc * ow * oh;
    }

    return SUCCESS;
}

int pixelshuffle2d_spacetochannel_csleq8(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint s,
    infloats x_ptr, outfloats y_ptr) {

    const uint cs = ic * s;

    if (cs <= 1) {
        return FAILURE_BADPARAM;
    }
    if (cs < AVX1_FLOAT_STRIDE) {
        return pixelshuffle2d_spacetochannel_cs2to3(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }
    if (cs == AVX1_FLOAT_STRIDE) {
        return pixelshuffle2d_spacetochannel_cs4(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }
    if (cs < AVX2_FLOAT_STRIDE) {
        return pixelshuffle2d_spacetochannel_cs5to7(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }
    if (cs == AVX2_FLOAT_STRIDE) {
        return pixelshuffle2d_spacetochannel_cs8(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::PixelShuffle2D::SpaceToChannel(
    UInt32 n, UInt32 ic, UInt32 iw, UInt32 ih, UInt32 s,
    Array<float>^ xs, Array<float>^ yc) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (s <= 0 || s > MAX_PIXELSHUFFLE_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidStride);
    }
    if (ic > MAX_CHANNELS || ic * s * s > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || ic <= 0 || iw <= 0 || ih <= 0) {
        return;
    }
    if ((iw % s) != 0 || iw > MAX_MAP_SIZE || (ih % s) != 0 || ih > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(ic, s, s);

    UInt32 ow = iw / s, oh = ih / s, oc = ic * s * s;

    Util::CheckProdOverflow(n, ic, iw, ih);
    Util::CheckProdOverflow(n, oc, ow, oh);

    Util::CheckLength(n * ic * iw * ih, xs);
    Util::CheckLength(n * oc * ow * oh, yc);

    Util::CheckDuplicateArray(xs, yc);

    if (s == 1) {
        Elementwise::Copy(n * ic * iw * ih, xs, yc);
        return;
    }

    const float* x_ptr = (const float*)(xs->Ptr.ToPointer());
    float* y_ptr = (float*)(yc->Ptr.ToPointer());

    int ret = UNEXECUTED;

    const uint cs = ic * s;

    if (cs <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = pixelshuffle2d_spacetochannel_csleq8(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }
    else if ((cs & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = pixelshuffle2d_spacetochannel_aligned(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = pixelshuffle2d_spacetochannel_unaligned(n, ic, oc, iw, ow, ih, oh, s, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}