#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

int upsample1d_neighbor_c32x(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            uint r = c;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = y_ptr + c * (ox + 1);

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_ps(yl_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_ps(yr_ptr, x0, x1, x2, x3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            uint r = c;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = y_ptr + c * (ox + 1);

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_ps(yl_ptr, x0, x1, x2, x3);
                _mm256_stream_x4_ps(yr_ptr, x0, x1, x2, x3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 3) {
                _mm256_load_x3_ps(x_ptr, x0, x1, x2);
                _mm256_stream_x3_ps(yl_ptr, x0, x1, x2);
                _mm256_stream_x3_ps(yr_ptr, x0, x1, x2);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_load_x2_ps(x_ptr, x0, x1);
                _mm256_stream_x2_ps(yl_ptr, x0, x1);
                _mm256_stream_x2_ps(yr_ptr, x0, x1);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(x_ptr, x0);
                _mm256_stream_x1_ps(yl_ptr, x0);
                _mm256_stream_x1_ps(yr_ptr, x0);
            }

            x_ptr += r;
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

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            uint r = c;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = y_ptr + c * (ox + 1);

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_storeu_x4_ps(yl_ptr, x0, x1, x2, x3);
                _mm256_storeu_x4_ps(yr_ptr, x0, x1, x2, x3);

                x_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 3) {
                _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                _mm256_maskstore_x4_ps(yl_ptr, x0, x1, x2, x3, mask);
                _mm256_maskstore_x4_ps(yr_ptr, x0, x1, x2, x3, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
                _mm256_maskstore_x3_ps(yl_ptr, x0, x1, x2, mask);
                _mm256_maskstore_x3_ps(yr_ptr, x0, x1, x2, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x2_ps(x_ptr, x0, x1);
                _mm256_maskstore_x2_ps(yl_ptr, x0, x1, mask);
                _mm256_maskstore_x2_ps(yr_ptr, x0, x1, mask);
            }
            else if (r > 0) {
                _mm256_loadu_x1_ps(x_ptr, x0);
                _mm256_maskstore_x1_ps(yl_ptr, x0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, x0, mask);
            }

            x_ptr += r;
        }

        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_neighbor_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c > AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (c == AVX2_FLOAT_STRIDE) {
        __m256 x0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                _mm256_load_x1_ps(x_ptr, x0);

                _mm256_stream_x1_ps(yl_ptr, x0);
                _mm256_stream_x1_ps(yr_ptr, x0);
            
                x_ptr += c;
            }

            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c > AVX1_FLOAT_STRIDE && c < AVX2_FLOAT_STRIDE) {
        const __m256i mask = _mm256_setmask_ps(c);

        __m256 x0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                _mm256_loadu_x1_ps(x_ptr, x0);

                _mm256_maskstore_x1_ps(yl_ptr, x0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, x0, mask);

                x_ptr += c;
            }

            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c == AVX1_FLOAT_STRIDE) {
        __m128 x0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                x0 = _mm_load_ps(x_ptr);

                _mm_stream_ps(yl_ptr, x0);
                _mm_stream_ps(yr_ptr, x0);

                x_ptr += c;
            }

            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c > 1 && c < AVX1_FLOAT_STRIDE) {
        const __m128i mask = _mm_setmask_ps(c);

        __m128 x0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                x0 = _mm_loadu_ps(x_ptr);

                _mm_maskstore_ps(yl_ptr, mask, x0);
                _mm_maskstore_ps(yr_ptr, mask, x0);

                x_ptr += c;
            }

            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c == 1) {
        float x0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                float* yl_ptr = y_ptr + ox;
                float* yr_ptr = y_ptr + (ox + 1);

                x0 = *x_ptr;

                *yl_ptr = x0;
                *yr_ptr = x0;

                x_ptr++;
            }

            y_ptr += ow;
        }

        return SUCCESS;
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

    if (iw > MAX_MAP_SIZE || ow > MAX_MAP_SIZE) {
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
    else if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type c32x");
#endif // _DEBUG

        ret = upsample1d_neighbor_c32x(n, c, iw, ow, x_ptr, y_ptr);
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
