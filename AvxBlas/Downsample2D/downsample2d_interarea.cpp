#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_sum_s.hpp"

#pragma unmanaged

__forceinline __m128 _mm_interarea2d_ps(__m128 xlu, __m128 xru, __m128 xld, __m128 xrd) {
    __m128 y = _mm_add_ps(
        _mm_add_ps(xlu, xru),
        _mm_add_ps(xld, xrd)
    );

    return y;
}

__forceinline __m256 _mm256_interarea2d_ps(__m256 xlu, __m256 xru, __m256 xld, __m256 xrd) {
    __m256 y = _mm256_add_ps(
        _mm256_add_ps(xlu, xru),
        _mm256_add_ps(xld, xrd)
    );

    return y;
}

int downsample2d_interarea_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(xlu_ptr, xlu);
                    _mm256_load_x1_ps(xru_ptr, xru);
                    _mm256_load_x1_ps(xld_ptr, xld);
                    _mm256_load_x1_ps(xrd_ptr, xrd);

                    __m256 y = _mm256_interarea2d_ps(xlu, xru, xld, xrd);

                    _mm256_stream_x1_ps(yc_ptr, y);

                    xlu_ptr += AVX2_FLOAT_STRIDE;
                    xru_ptr += AVX2_FLOAT_STRIDE;
                    xld_ptr += AVX2_FLOAT_STRIDE;
                    xrd_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
                    _mm256_loadu_x1_ps(xru_ptr, xru);
                    _mm256_loadu_x1_ps(xld_ptr, xld);
                    _mm256_loadu_x1_ps(xrd_ptr, xrd);

                    __m256 y = _mm256_interarea2d_ps(xlu, xru, xld, xrd);

                    _mm256_storeu_x1_ps(yc_ptr, y);

                    xlu_ptr += AVX2_FLOAT_STRIDE;
                    xru_ptr += AVX2_FLOAT_STRIDE;
                    xld_ptr += AVX2_FLOAT_STRIDE;
                    xrd_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
                if (r > 0) {
                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
                    _mm256_loadu_x1_ps(xru_ptr, xru);
                    _mm256_loadu_x1_ps(xld_ptr, xld);
                    _mm256_loadu_x1_ps(xrd_ptr, xrd);

                    __m256 y = _mm256_interarea2d_ps(xlu, xru, xld, xrd);

                    _mm256_maskstore_x1_ps(yc_ptr, y, mask);
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(ow & AVX2_FLOAT_REMAIN_MASK);

    __m256 y;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += AVX2_FLOAT_STRIDE * 2, ox += AVX2_FLOAT_STRIDE) {

                const float* xlu_ptr = x_ptr + (ix + iw * iy);
                const float* xld_ptr = xlu_ptr + iw;

                float* yc_ptr = y_ptr + (ox + ow * oy);

                if (ix + AVX2_FLOAT_STRIDE < iw) {
                    __m256 xu0 = _mm256_loadu_ps(xlu_ptr);
                    __m256 xu1 = _mm256_loadu_ps(xlu_ptr + AVX2_FLOAT_STRIDE);
                    __m256 xd0 = _mm256_loadu_ps(xld_ptr);
                    __m256 xd1 = _mm256_loadu_ps(xld_ptr + AVX2_FLOAT_STRIDE);

                    y = _mm256_add_ps(
                        _mm256_hadd2_x2_ps(xu0, xu1),
                        _mm256_hadd2_x2_ps(xd0, xd1)
                    );
                }
                else {
                    __m256 xu0 = _mm256_loadu_ps(xlu_ptr);
                    __m256 xd0 = _mm256_loadu_ps(xld_ptr);

                    y = _mm256_add_ps(
                        _mm256_hadd2_x2_ps(xu0, xu0),
                        _mm256_hadd2_x2_ps(xd0, xd0)
                    );
                }

                if (ox + AVX2_FLOAT_STRIDE <= ow) {
                    _mm256_storeu_x1_ps(yc_ptr, y);
                }
                else {
                    _mm256_maskstore_x1_ps(yc_ptr, y, mask);
                }
            }
        }

        x_ptr += iw * ih;
        y_ptr += ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m128 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                xlu = _mm_loadu_ps(xlu_ptr);
                xru = _mm_loadu_ps(xru_ptr);
                xld = _mm_loadu_ps(xld_ptr);
                xrd = _mm_loadu_ps(xrd_ptr);

                __m128 y = _mm_interarea2d_ps(xlu, xru, xld, xrd);

                _mm_maskstore_ps(yc_ptr, mask, y);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_c4(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                xlu = _mm_load_ps(xlu_ptr);
                xru = _mm_load_ps(xru_ptr);
                xld = _mm_load_ps(xld_ptr);
                xrd = _mm_load_ps(xrd_ptr);

                __m128 y = _mm_interarea2d_ps(xlu, xru, xld, xrd);

                _mm_stream_ps(yc_ptr, y);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                _mm256_loadu_x1_ps(xlu_ptr, xlu);
                _mm256_loadu_x1_ps(xru_ptr, xru);
                _mm256_loadu_x1_ps(xld_ptr, xld);
                _mm256_loadu_x1_ps(xrd_ptr, xrd);

                __m256 y = _mm256_interarea2d_ps(xlu, xru, xld, xrd);

                _mm256_maskstore_x1_ps(yc_ptr, y, mask);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_c8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xru, xld, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
            for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                const float* xlu_ptr = x_ptr + c * (ix + iw * iy);
                const float* xru_ptr = xlu_ptr + c;
                const float* xld_ptr = xlu_ptr + c * iw;
                const float* xrd_ptr = xlu_ptr + c * (iw + 1u);

                float* yc_ptr = y_ptr + c * (ox + ow * oy);

                _mm256_load_x1_ps(xlu_ptr, xlu);
                _mm256_load_x1_ps(xru_ptr, xru);
                _mm256_load_x1_ps(xld_ptr, xld);
                _mm256_load_x1_ps(xrd_ptr, xrd);

                __m256 y = _mm256_interarea2d_ps(xlu, xru, xld, xrd);

                _mm256_stream_x1_ps(yc_ptr, y);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int downsample2d_interarea_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return downsample2d_interarea_c1(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return downsample2d_interarea_c2to3(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return downsample2d_interarea_c4(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return downsample2d_interarea_c5to7(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return downsample2d_interarea_c8(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Downsample2D::InterareaX2(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0) {
        return;
    }
    if ((iw & 1u) != 0 || iw > MAX_MAP_SIZE || (ih & 1u) != 0 || ih > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    UInt32 ow = iw / 2, oh = ih / 2;

    Util::CheckProdOverflow(n, c, iw, ih);
    Util::CheckProdOverflow(n, c, ow, oh);

    Util::CheckLength(n * c * iw * ih, x);
    Util::CheckLength(n * c * ow * oh, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = downsample2d_interarea_cleq8(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = downsample2d_interarea_aligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = downsample2d_interarea_unaligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh, y, 1.0f / 4.0f, y);
}
