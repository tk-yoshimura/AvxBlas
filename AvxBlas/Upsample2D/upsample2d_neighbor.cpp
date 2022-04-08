#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

int upsample2d_neighbor_c32x(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                uint r = c;

                float* ylu_ptr = y_ptr + c * ox;
                float* yru_ptr = y_ptr + c * (ox + 1);
                float* yld_ptr = y_ptr + c * (ox + ow);
                float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(ylu_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yru_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yld_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yrd_ptr, x0, x1, x2, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    ylu_ptr += AVX2_FLOAT_STRIDE * 4;
                    yru_ptr += AVX2_FLOAT_STRIDE * 4;
                    yld_ptr += AVX2_FLOAT_STRIDE * 4;
                    yrd_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
            }
        }

        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_neighbor_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                uint r = c;

                float* ylu_ptr = y_ptr + c * ox;
                float* yru_ptr = y_ptr + c * (ox + 1);
                float* yld_ptr = y_ptr + c * (ox + ow);
                float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(ylu_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yru_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yld_ptr, x0, x1, x2, x3);
                    _mm256_stream_x4_ps(yrd_ptr, x0, x1, x2, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    ylu_ptr += AVX2_FLOAT_STRIDE * 4;
                    yru_ptr += AVX2_FLOAT_STRIDE * 4;
                    yld_ptr += AVX2_FLOAT_STRIDE * 4;
                    yrd_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 3) {
                    _mm256_load_x3_ps(x_ptr, x0, x1, x2);
                    _mm256_stream_x3_ps(ylu_ptr, x0, x1, x2);
                    _mm256_stream_x3_ps(yru_ptr, x0, x1, x2);
                    _mm256_stream_x3_ps(yld_ptr, x0, x1, x2);
                    _mm256_stream_x3_ps(yrd_ptr, x0, x1, x2);
                }
                else if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_load_x2_ps(x_ptr, x0, x1);
                    _mm256_stream_x2_ps(ylu_ptr, x0, x1);
                    _mm256_stream_x2_ps(yru_ptr, x0, x1);
                    _mm256_stream_x2_ps(yld_ptr, x0, x1);
                    _mm256_stream_x2_ps(yrd_ptr, x0, x1);
                }
                else if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(x_ptr, x0);
                    _mm256_stream_x1_ps(ylu_ptr, x0);
                    _mm256_stream_x1_ps(yru_ptr, x0);
                    _mm256_stream_x1_ps(yld_ptr, x0);
                    _mm256_stream_x1_ps(yrd_ptr, x0);
                }

                x_ptr += r;
            }
        }

        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_neighbor_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                uint r = c;

                float* ylu_ptr = y_ptr + c * ox;
                float* yru_ptr = y_ptr + c * (ox + 1);
                float* yld_ptr = y_ptr + c * (ox + ow);
                float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_ps(ylu_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_ps(yru_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_ps(yld_ptr, x0, x1, x2, x3);
                    _mm256_storeu_x4_ps(yrd_ptr, x0, x1, x2, x3);

                    x_ptr += AVX2_FLOAT_STRIDE * 4;
                    ylu_ptr += AVX2_FLOAT_STRIDE * 4;
                    yru_ptr += AVX2_FLOAT_STRIDE * 4;
                    yld_ptr += AVX2_FLOAT_STRIDE * 4;
                    yrd_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 3) {
                    _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);
                    _mm256_maskstore_x4_ps(ylu_ptr, x0, x1, x2, x3, mask);
                    _mm256_maskstore_x4_ps(yru_ptr, x0, x1, x2, x3, mask);
                    _mm256_maskstore_x4_ps(yld_ptr, x0, x1, x2, x3, mask);
                    _mm256_maskstore_x4_ps(yrd_ptr, x0, x1, x2, x3, mask);
                }
                else if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_loadu_x3_ps(x_ptr, x0, x1, x2);
                    _mm256_maskstore_x3_ps(ylu_ptr, x0, x1, x2, mask);
                    _mm256_maskstore_x3_ps(yru_ptr, x0, x1, x2, mask);
                    _mm256_maskstore_x3_ps(yld_ptr, x0, x1, x2, mask);
                    _mm256_maskstore_x3_ps(yrd_ptr, x0, x1, x2, mask);
                }
                else if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x2_ps(x_ptr, x0, x1);
                    _mm256_maskstore_x2_ps(ylu_ptr, x0, x1, mask);
                    _mm256_maskstore_x2_ps(yru_ptr, x0, x1, mask);
                    _mm256_maskstore_x2_ps(yld_ptr, x0, x1, mask);
                    _mm256_maskstore_x2_ps(yrd_ptr, x0, x1, mask);
                }
                else if (r > 0) {
                    _mm256_loadu_x1_ps(x_ptr, x0);
                    _mm256_maskstore_x1_ps(ylu_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yru_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yld_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yrd_ptr, x0, mask);
                }

                x_ptr += r;
            }
        }

        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_neighbor_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c > AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (c == AVX2_FLOAT_STRIDE) {
        __m256 x0;

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    float* ylu_ptr = y_ptr + c * ox;
                    float* yru_ptr = y_ptr + c * (ox + 1);
                    float* yld_ptr = y_ptr + c * (ox + ow);
                    float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                    _mm256_load_x1_ps(x_ptr, x0);

                    _mm256_stream_x1_ps(ylu_ptr, x0);
                    _mm256_stream_x1_ps(yru_ptr, x0);
                    _mm256_stream_x1_ps(yld_ptr, x0);
                    _mm256_stream_x1_ps(yrd_ptr, x0);

                    x_ptr += c;
                }
            }

            y_ptr += c * ow * oh;
        }

        return SUCCESS;
    }

    if (c > AVX1_FLOAT_STRIDE && c < AVX2_FLOAT_STRIDE) {
        const __m256i mask = _mm256_setmask_ps(c);

        __m256 x0;

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    float* ylu_ptr = y_ptr + c * ox;
                    float* yru_ptr = y_ptr + c * (ox + 1);
                    float* yld_ptr = y_ptr + c * (ox + ow);
                    float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                    _mm256_loadu_x1_ps(x_ptr, x0);

                    _mm256_maskstore_x1_ps(ylu_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yru_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yld_ptr, x0, mask);
                    _mm256_maskstore_x1_ps(yrd_ptr, x0, mask);

                    x_ptr += c;
                }
            }

            y_ptr += c * ow * oh;
        }

        return SUCCESS;
    }

    if (c == AVX1_FLOAT_STRIDE) {
        __m128 x0;

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    float* ylu_ptr = y_ptr + c * ox;
                    float* yru_ptr = y_ptr + c * (ox + 1);
                    float* yld_ptr = y_ptr + c * (ox + ow);
                    float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                    x0 = _mm_load_ps(x_ptr);

                    _mm_stream_ps(ylu_ptr, x0);
                    _mm_stream_ps(yru_ptr, x0);
                    _mm_stream_ps(yld_ptr, x0);
                    _mm_stream_ps(yrd_ptr, x0);

                    x_ptr += c;
                }
            }

            y_ptr += c * ow * oh;
        }

        return SUCCESS;
    }

    if (c > 1 && c < AVX1_FLOAT_STRIDE) {
        const __m128i mask = _mm_setmask_ps(c);

        __m128 x0;

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    float* ylu_ptr = y_ptr + c * ox;
                    float* yru_ptr = y_ptr + c * (ox + 1);
                    float* yld_ptr = y_ptr + c * (ox + ow);
                    float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                    x0 = _mm_loadu_ps(x_ptr);

                    _mm_maskstore_ps(ylu_ptr, mask, x0);
                    _mm_maskstore_ps(yru_ptr, mask, x0);
                    _mm_maskstore_ps(yld_ptr, mask, x0);
                    _mm_maskstore_ps(yrd_ptr, mask, x0);

                    x_ptr += c;
                }
            }

            y_ptr += c * ow * oh;
        }

        return SUCCESS;
    }

    if (c == 1) {
        float x0;

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    float* ylu_ptr = y_ptr + c * ox;
                    float* yru_ptr = y_ptr + c * (ox + 1);
                    float* yld_ptr = y_ptr + c * (ox + ow);
                    float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                    x0 = *x_ptr;

                    *ylu_ptr = x0;
                    *yru_ptr = x0;
                    *yld_ptr = x0;
                    *yrd_ptr = x0;

                    x_ptr++;
                }
            }

            y_ptr += ow * oh;
        }

        return SUCCESS;
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample2D::NeighborX2(
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

    Util::CheckProdOverflow(iw, 2u);
    Util::CheckProdOverflow(ih, 2u);
    UInt32 ow = iw * 2, oh = ih * 2;

    if (iw > MAX_MAP_SIZE || ow > MAX_MAP_SIZE || ih > MAX_MAP_SIZE || oh > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

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

        ret = upsample2d_neighbor_cleq8(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type c32x");
#endif // _DEBUG

        ret = upsample2d_neighbor_c32x(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample2d_neighbor_aligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample2d_neighbor_unaligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
