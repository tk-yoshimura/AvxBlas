#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_transpose_s.hpp"

#pragma unmanaged

int upsample3d_neighbor_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_load_x1_ps(x_ptr, x);
                        _mm256_stream_x1_ps(yluf_ptr, x);
                        _mm256_stream_x1_ps(yruf_ptr, x);
                        _mm256_stream_x1_ps(yldf_ptr, x);
                        _mm256_stream_x1_ps(yrdf_ptr, x);
                        _mm256_stream_x1_ps(ylub_ptr, x);
                        _mm256_stream_x1_ps(yrub_ptr, x);
                        _mm256_stream_x1_ps(yldb_ptr, x);
                        _mm256_stream_x1_ps(yrdb_ptr, x);

                        x_ptr += AVX2_FLOAT_STRIDE;
                        yluf_ptr += AVX2_FLOAT_STRIDE;
                        yruf_ptr += AVX2_FLOAT_STRIDE;
                        yldf_ptr += AVX2_FLOAT_STRIDE;
                        yrdf_ptr += AVX2_FLOAT_STRIDE;
                        ylub_ptr += AVX2_FLOAT_STRIDE;
                        yrub_ptr += AVX2_FLOAT_STRIDE;
                        yldb_ptr += AVX2_FLOAT_STRIDE;
                        yrdb_ptr += AVX2_FLOAT_STRIDE;
                        r -= AVX2_FLOAT_STRIDE;
                    }
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_loadu_x1_ps(x_ptr, x);
                        _mm256_storeu_x1_ps(yluf_ptr, x);
                        _mm256_storeu_x1_ps(yruf_ptr, x);
                        _mm256_storeu_x1_ps(yldf_ptr, x);
                        _mm256_storeu_x1_ps(yrdf_ptr, x);
                        _mm256_storeu_x1_ps(ylub_ptr, x);
                        _mm256_storeu_x1_ps(yrub_ptr, x);
                        _mm256_storeu_x1_ps(yldb_ptr, x);
                        _mm256_storeu_x1_ps(yrdb_ptr, x);

                        x_ptr += AVX2_FLOAT_STRIDE;
                        yluf_ptr += AVX2_FLOAT_STRIDE;
                        yruf_ptr += AVX2_FLOAT_STRIDE;
                        yldf_ptr += AVX2_FLOAT_STRIDE;
                        yrdf_ptr += AVX2_FLOAT_STRIDE;
                        ylub_ptr += AVX2_FLOAT_STRIDE;
                        yrub_ptr += AVX2_FLOAT_STRIDE;
                        yldb_ptr += AVX2_FLOAT_STRIDE;
                        yrdb_ptr += AVX2_FLOAT_STRIDE;
                        r -= AVX2_FLOAT_STRIDE;
                    }
                    if (r > 0) {
                        _mm256_loadu_x1_ps(x_ptr, x);
                        _mm256_maskstore_x1_ps(yluf_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yruf_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yldf_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yrdf_ptr, x, mask);
                        _mm256_maskstore_x1_ps(ylub_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yrub_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yldb_ptr, x, mask);
                        _mm256_maskstore_x1_ps(yrdb_ptr, x, mask);

                        x_ptr += r;
                    }
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_c1(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps((iw * 2u) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix += AVX2_FLOAT_STRIDE, ox += AVX2_FLOAT_STRIDE * 2) {

                    const float* xc_ptr = x_ptr + (ix + iw * (iy + ih * iz));

                    float* yuf_ptr = y_ptr + (ox + ow * (oy + oh * oz));
                    float* ydf_ptr = yuf_ptr + ow;
                    float* yub_ptr = yuf_ptr + ow * oh;
                    float* ydb_ptr = ydf_ptr + ow * oh;

                    __m256 x = _mm256_loadu_ps(xc_ptr);

                    __m256x2 yt = _mm256_transpose8x2_ps(x, x);

                    if (ix + AVX2_FLOAT_STRIDE <= iw) {
                        _mm256_storeu_x2_ps(yuf_ptr, yt.imm0, yt.imm1);
                        _mm256_storeu_x2_ps(ydf_ptr, yt.imm0, yt.imm1);
                        _mm256_storeu_x2_ps(yub_ptr, yt.imm0, yt.imm1);
                        _mm256_storeu_x2_ps(ydb_ptr, yt.imm0, yt.imm1);
                    }
                    else if (ix + AVX2_FLOAT_STRIDE / 2 < iw) {
                        _mm256_maskstore_x2_ps(yuf_ptr, yt.imm0, yt.imm1, mask);
                        _mm256_maskstore_x2_ps(ydf_ptr, yt.imm0, yt.imm1, mask);
                        _mm256_maskstore_x2_ps(yub_ptr, yt.imm0, yt.imm1, mask);
                        _mm256_maskstore_x2_ps(ydb_ptr, yt.imm0, yt.imm1, mask);
                    }
                    else if (ix + AVX2_FLOAT_STRIDE / 2 == iw) {
                        _mm256_storeu_x1_ps(yuf_ptr, yt.imm0);
                        _mm256_storeu_x1_ps(ydf_ptr, yt.imm0);
                        _mm256_storeu_x1_ps(yub_ptr, yt.imm0);
                        _mm256_storeu_x1_ps(ydb_ptr, yt.imm0);
                    }
                    else {
                        _mm256_maskstore_x1_ps(yuf_ptr, yt.imm0, mask);
                        _mm256_maskstore_x1_ps(ydf_ptr, yt.imm0, mask);
                        _mm256_maskstore_x1_ps(yub_ptr, yt.imm0, mask);
                        _mm256_maskstore_x1_ps(ydb_ptr, yt.imm0, mask);
                    }
                }
            }
        }

        x_ptr += iw * ih * id;
        y_ptr += ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c);

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    x = _mm_loadu_ps(x_ptr);

                    _mm_maskstore_ps(yluf_ptr, mask, x);
                    _mm_maskstore_ps(yruf_ptr, mask, x);
                    _mm_maskstore_ps(yldf_ptr, mask, x);
                    _mm_maskstore_ps(yrdf_ptr, mask, x);
                    _mm_maskstore_ps(ylub_ptr, mask, x);
                    _mm_maskstore_ps(yrub_ptr, mask, x);
                    _mm_maskstore_ps(yldb_ptr, mask, x);
                    _mm_maskstore_ps(yrdb_ptr, mask, x);

                    x_ptr += c;
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_c4(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    x = _mm_load_ps(x_ptr);

                    _mm_stream_ps(yluf_ptr, x);
                    _mm_stream_ps(yruf_ptr, x);
                    _mm_stream_ps(yldf_ptr, x);
                    _mm_stream_ps(yrdf_ptr, x);
                    _mm_stream_ps(ylub_ptr, x);
                    _mm_stream_ps(yrub_ptr, x);
                    _mm_stream_ps(yldb_ptr, x);
                    _mm_stream_ps(yrdb_ptr, x);

                    x_ptr += c;
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    _mm256_loadu_x1_ps(x_ptr, x);

                    _mm256_maskstore_x1_ps(yluf_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yruf_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yldf_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yrdf_ptr, x, mask);
                    _mm256_maskstore_x1_ps(ylub_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yrub_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yldb_ptr, x, mask);
                    _mm256_maskstore_x1_ps(yrdb_ptr, x, mask);

                    x_ptr += c;
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_c8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * ow * oh;
                    float* yrub_ptr = yruf_ptr + c * ow * oh;
                    float* yldb_ptr = yldf_ptr + c * ow * oh;
                    float* yrdb_ptr = yrdf_ptr + c * ow * oh;

                    _mm256_load_x1_ps(x_ptr, x);

                    _mm256_stream_x1_ps(yluf_ptr, x);
                    _mm256_stream_x1_ps(yruf_ptr, x);
                    _mm256_stream_x1_ps(yldf_ptr, x);
                    _mm256_stream_x1_ps(yrdf_ptr, x);
                    _mm256_stream_x1_ps(ylub_ptr, x);
                    _mm256_stream_x1_ps(yrub_ptr, x);
                    _mm256_stream_x1_ps(yldb_ptr, x);
                    _mm256_stream_x1_ps(yrdb_ptr, x);

                    x_ptr += c;
                }
            }
        }

        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_neighbor_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return upsample3d_neighbor_c1(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return upsample3d_neighbor_c2to3(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return upsample3d_neighbor_c4(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return upsample3d_neighbor_c5to7(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return upsample3d_neighbor_c8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample3D::NeighborX2(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0 || id <= 0) {
        return;
    }

    Util::CheckProdOverflow(iw, 2u);
    Util::CheckProdOverflow(ih, 2u);
    Util::CheckProdOverflow(id, 2u);
    UInt32 ow = iw * 2, oh = ih * 2, od = id * 2;

    if (ow > MAX_MAP_SIZE || oh > MAX_MAP_SIZE || od > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(n, c, iw, ih, id);
    Util::CheckProdOverflow(n, c, ow, oh, od);

    Util::CheckLength(n * c * iw * ih * id, x);
    Util::CheckLength(n * c * ow * oh * od, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = upsample3d_neighbor_cleq8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample3d_neighbor_aligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample3d_neighbor_unaligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }

    Util::AssertReturnCode(ret);
}
