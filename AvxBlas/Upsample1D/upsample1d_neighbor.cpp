#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_transpose_s.hpp"

#pragma unmanaged

int upsample1d_neighbor_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            uint r = c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(x_ptr, x);
                _mm256_stream_x1_ps(yl_ptr, x);
                _mm256_stream_x1_ps(yr_ptr, x);

                x_ptr += AVX2_FLOAT_STRIDE;
                yl_ptr += AVX2_FLOAT_STRIDE;
                yr_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            uint r = c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x1_ps(x_ptr, x);
                _mm256_storeu_x1_ps(yl_ptr, x);
                _mm256_storeu_x1_ps(yr_ptr, x);

                x_ptr += AVX2_FLOAT_STRIDE;
                yl_ptr += AVX2_FLOAT_STRIDE;
                yr_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
            if (r > 0) {
                _mm256_loadu_x1_ps(x_ptr, x);
                _mm256_maskstore_x1_ps(yl_ptr, x, mask);
                _mm256_maskstore_x1_ps(yr_ptr, x, mask);

                x_ptr += r;
            }
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps((iw * 2u) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix += AVX2_FLOAT_STRIDE, ox += AVX2_FLOAT_STRIDE * 2) {

            const float* xc_ptr = x_ptr + ix;

            float* yc_ptr = y_ptr + ox;

            __m256 x = _mm256_loadu_ps(xc_ptr);

            __m256x2 yt = _mm256_transpose8x2_ps(x, x);

            if (ix + AVX2_FLOAT_STRIDE <= iw) {
                _mm256_storeu_x2_ps(yc_ptr, yt.imm0, yt.imm1);
            }
            else if (ix + AVX2_FLOAT_STRIDE / 2 < iw) {
                _mm256_maskstore_x2_ps(yc_ptr, yt.imm0, yt.imm1, mask);
            }
            else if (ix + AVX2_FLOAT_STRIDE / 2 == iw) {
                _mm256_storeu_x1_ps(yc_ptr, yt.imm0);
            }
            else {
                _mm256_maskstore_x1_ps(yc_ptr, yt.imm0, mask);
            }
        }

        x_ptr += iw;
        y_ptr += ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c);

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            x = _mm_loadu_ps(x_ptr);

            _mm_maskstore_ps(yl_ptr, mask, x);
            _mm_maskstore_ps(yr_ptr, mask, x);

            x_ptr += c;
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_c4(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            x = _mm_load_ps(x_ptr);

            _mm_stream_ps(yl_ptr, x);
            _mm_stream_ps(yr_ptr, x);

            x_ptr += c;
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            _mm256_loadu_x1_ps(x_ptr, x);

            _mm256_maskstore_x1_ps(yl_ptr, x, mask);
            _mm256_maskstore_x1_ps(yr_ptr, x, mask);

            x_ptr += c;
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_c8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            _mm256_load_x1_ps(x_ptr, x);

            _mm256_stream_x1_ps(yl_ptr, x);
            _mm256_stream_x1_ps(yr_ptr, x);

            x_ptr += c;
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return upsample1d_neighbor_c1(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return upsample1d_neighbor_c2to3(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return upsample1d_neighbor_c4(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return upsample1d_neighbor_c5to7(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return upsample1d_neighbor_c8(n, c, iw, ow, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample1D::NeighborX2(
    UInt32 n, UInt32 c, UInt32 iw,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0) {
        return;
    }

    Util::CheckProdOverflow(iw, 2u);
    UInt32 ow = iw * 2;

    if (ow > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(n, c, iw);
    Util::CheckProdOverflow(n, c, ow);

    Util::CheckLength(n * c * iw, x);
    Util::CheckLength(n * c * ow, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = upsample1d_neighbor_cleq8(n, c, iw, ow, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample1d_neighbor_aligned(n, c, iw, ow, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample1d_neighbor_unaligned(n, c, iw, ow, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
