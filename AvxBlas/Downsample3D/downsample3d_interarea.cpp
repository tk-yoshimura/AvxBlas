#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_sum_s.hpp"

#pragma unmanaged

__forceinline __m128 _mm_interarea3d_ps(
    __m128 xluf, __m128 xruf, __m128 xldf, __m128 xrdf,
    __m128 xlub, __m128 xrub, __m128 xldb, __m128 xrdb) {

    __m128 y = _mm_add_ps(
        _mm_add_ps(
            _mm_add_ps(xluf, xruf),
            _mm_add_ps(xldf, xrdf)
        ),
        _mm_add_ps(
            _mm_add_ps(xlub, xrub),
            _mm_add_ps(xldb, xrdb)
        )
    );

    return y;
}

__forceinline __m256 _mm256_interarea3d_ps(
    __m256 xluf, __m256 xruf, __m256 xldf, __m256 xrdf,
    __m256 xlub, __m256 xrub, __m256 xldb, __m256 xrdb) {

    __m256 y = _mm256_add_ps(
        _mm256_add_ps(
            _mm256_add_ps(xluf, xruf),
            _mm256_add_ps(xldf, xrdf)
        ),
        _mm256_add_ps(
            _mm256_add_ps(xlub, xrub),
            _mm256_add_ps(xldb, xrdb)
        )
    );

    return y;
}

int downsample3d_interarea_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_load_x1_ps(xluf_ptr, xluf);
                        _mm256_load_x1_ps(xruf_ptr, xruf);
                        _mm256_load_x1_ps(xldf_ptr, xldf);
                        _mm256_load_x1_ps(xrdf_ptr, xrdf);
                        _mm256_load_x1_ps(xlub_ptr, xlub);
                        _mm256_load_x1_ps(xrub_ptr, xrub);
                        _mm256_load_x1_ps(xldb_ptr, xldb);
                        _mm256_load_x1_ps(xrdb_ptr, xrdb);

                        __m256 y = _mm256_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                        _mm256_stream_x1_ps(yc_ptr, y);

                        xluf_ptr += AVX2_FLOAT_STRIDE;
                        xruf_ptr += AVX2_FLOAT_STRIDE;
                        xldf_ptr += AVX2_FLOAT_STRIDE;
                        xrdf_ptr += AVX2_FLOAT_STRIDE;
                        xlub_ptr += AVX2_FLOAT_STRIDE;
                        xrub_ptr += AVX2_FLOAT_STRIDE;
                        xldb_ptr += AVX2_FLOAT_STRIDE;
                        xrdb_ptr += AVX2_FLOAT_STRIDE;
                        yc_ptr += AVX2_FLOAT_STRIDE;
                        r -= AVX2_FLOAT_STRIDE;
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    uint r = c;

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_loadu_x1_ps(xluf_ptr, xluf);
                        _mm256_loadu_x1_ps(xruf_ptr, xruf);
                        _mm256_loadu_x1_ps(xldf_ptr, xldf);
                        _mm256_loadu_x1_ps(xrdf_ptr, xrdf);
                        _mm256_loadu_x1_ps(xlub_ptr, xlub);
                        _mm256_loadu_x1_ps(xrub_ptr, xrub);
                        _mm256_loadu_x1_ps(xldb_ptr, xldb);
                        _mm256_loadu_x1_ps(xrdb_ptr, xrdb);

                        __m256 y = _mm256_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                        _mm256_storeu_x1_ps(yc_ptr, y);

                        xluf_ptr += AVX2_FLOAT_STRIDE;
                        xruf_ptr += AVX2_FLOAT_STRIDE;
                        xldf_ptr += AVX2_FLOAT_STRIDE;
                        xrdf_ptr += AVX2_FLOAT_STRIDE;
                        xlub_ptr += AVX2_FLOAT_STRIDE;
                        xrub_ptr += AVX2_FLOAT_STRIDE;
                        xldb_ptr += AVX2_FLOAT_STRIDE;
                        xrdb_ptr += AVX2_FLOAT_STRIDE;
                        yc_ptr += AVX2_FLOAT_STRIDE;
                        r -= AVX2_FLOAT_STRIDE;
                    }
                    if (r > 0) {
                        _mm256_loadu_x1_ps(xluf_ptr, xluf);
                        _mm256_loadu_x1_ps(xruf_ptr, xruf);
                        _mm256_loadu_x1_ps(xldf_ptr, xldf);
                        _mm256_loadu_x1_ps(xrdf_ptr, xrdf);
                        _mm256_loadu_x1_ps(xlub_ptr, xlub);
                        _mm256_loadu_x1_ps(xrub_ptr, xrub);
                        _mm256_loadu_x1_ps(xldb_ptr, xldb);
                        _mm256_loadu_x1_ps(xrdb_ptr, xrdb);

                        __m256 y = _mm256_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                        _mm256_maskstore_x1_ps(yc_ptr, y, mask);
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(ow & AVX2_FLOAT_REMAIN_MASK);

    __m256 y;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += AVX2_FLOAT_STRIDE * 2, ox += AVX2_FLOAT_STRIDE) {

                    const float* xluf_ptr = x_ptr + (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + 1;
                    const float* xldf_ptr = xluf_ptr + iw;
                    const float* xrdf_ptr = xluf_ptr + (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + (iw * ih);
                    const float* xrub_ptr = xruf_ptr + (iw * ih);
                    const float* xldb_ptr = xldf_ptr + (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    if (ix + AVX2_FLOAT_STRIDE < iw) {
                        __m256 xuf0 = _mm256_loadu_ps(xluf_ptr);
                        __m256 xuf1 = _mm256_loadu_ps(xluf_ptr + AVX2_FLOAT_STRIDE);
                        __m256 xdf0 = _mm256_loadu_ps(xldf_ptr);
                        __m256 xdf1 = _mm256_loadu_ps(xldf_ptr + AVX2_FLOAT_STRIDE);
                        __m256 xub0 = _mm256_loadu_ps(xlub_ptr);
                        __m256 xub1 = _mm256_loadu_ps(xlub_ptr + AVX2_FLOAT_STRIDE);
                        __m256 xdb0 = _mm256_loadu_ps(xldb_ptr);
                        __m256 xdb1 = _mm256_loadu_ps(xldb_ptr + AVX2_FLOAT_STRIDE);

                        y = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_hadd2_x2_ps(xuf0, xuf1),
                                _mm256_hadd2_x2_ps(xdf0, xdf1)
                            ),
                            _mm256_add_ps(
                                _mm256_hadd2_x2_ps(xub0, xub1),
                                _mm256_hadd2_x2_ps(xdb0, xdb1)
                            )
                        );
                    }
                    else {
                        __m256 xuf0 = _mm256_loadu_ps(xluf_ptr);
                        __m256 xdf0 = _mm256_loadu_ps(xldf_ptr);
                        __m256 xub0 = _mm256_loadu_ps(xlub_ptr);
                        __m256 xdb0 = _mm256_loadu_ps(xldb_ptr);

                        y = _mm256_add_ps(
                            _mm256_add_ps(
                                _mm256_hadd2_x2_ps(xuf0, xuf0),
                                _mm256_hadd2_x2_ps(xdf0, xdf0)
                            ),
                            _mm256_add_ps(
                                _mm256_hadd2_x2_ps(xub0, xub0),
                                _mm256_hadd2_x2_ps(xdb0, xdb0)
                            )
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
        }

        x_ptr += iw * ih * id;
        y_ptr += ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m128 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    xluf = _mm_loadu_ps(xluf_ptr);
                    xruf = _mm_loadu_ps(xruf_ptr);
                    xldf = _mm_loadu_ps(xldf_ptr);
                    xrdf = _mm_loadu_ps(xrdf_ptr);
                    xlub = _mm_loadu_ps(xlub_ptr);
                    xrub = _mm_loadu_ps(xrub_ptr);
                    xldb = _mm_loadu_ps(xldb_ptr);
                    xrdb = _mm_loadu_ps(xrdb_ptr);

                    __m128 y = _mm_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                    _mm_maskstore_ps(yc_ptr, mask, y);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_c4(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    xluf = _mm_load_ps(xluf_ptr);
                    xruf = _mm_load_ps(xruf_ptr);
                    xldf = _mm_load_ps(xldf_ptr);
                    xrdf = _mm_load_ps(xrdf_ptr);
                    xlub = _mm_load_ps(xlub_ptr);
                    xrub = _mm_load_ps(xrub_ptr);
                    xldb = _mm_load_ps(xldb_ptr);
                    xrdb = _mm_load_ps(xrdb_ptr);

                    __m128 y = _mm_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                    _mm_stream_ps(yc_ptr, y);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    uint r = c;

                    _mm256_loadu_x1_ps(xluf_ptr, xluf);
                    _mm256_loadu_x1_ps(xruf_ptr, xruf);
                    _mm256_loadu_x1_ps(xldf_ptr, xldf);
                    _mm256_loadu_x1_ps(xrdf_ptr, xrdf);
                    _mm256_loadu_x1_ps(xlub_ptr, xlub);
                    _mm256_loadu_x1_ps(xrub_ptr, xrub);
                    _mm256_loadu_x1_ps(xldb_ptr, xldb);
                    _mm256_loadu_x1_ps(xrdb_ptr, xrdb);

                    __m256 y = _mm256_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                    _mm256_maskstore_x1_ps(yc_ptr, y, mask);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_c8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; oz < od; iz += 2, oz++) {
            for (uint iy = 0, oy = 0; oy < oh; iy += 2, oy++) {
                for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

                    const float* xluf_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xruf_ptr = xluf_ptr + c;
                    const float* xldf_ptr = xluf_ptr + c * iw;
                    const float* xrdf_ptr = xluf_ptr + c * (iw + 1u);
                    const float* xlub_ptr = xluf_ptr + c * (iw * ih);
                    const float* xrub_ptr = xruf_ptr + c * (iw * ih);
                    const float* xldb_ptr = xldf_ptr + c * (iw * ih);
                    const float* xrdb_ptr = xrdf_ptr + c * (iw * ih);

                    float* yc_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));

                    uint r = c;

                    _mm256_load_x1_ps(xluf_ptr, xluf);
                    _mm256_load_x1_ps(xruf_ptr, xruf);
                    _mm256_load_x1_ps(xldf_ptr, xldf);
                    _mm256_load_x1_ps(xrdf_ptr, xrdf);
                    _mm256_load_x1_ps(xlub_ptr, xlub);
                    _mm256_load_x1_ps(xrub_ptr, xrub);
                    _mm256_load_x1_ps(xldb_ptr, xldb);
                    _mm256_load_x1_ps(xrdb_ptr, xrdb);

                    __m256 y = _mm256_interarea3d_ps(xluf, xruf, xldf, xrdf, xlub, xrub, xldb, xrdb);

                    _mm256_stream_x1_ps(yc_ptr, y);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int downsample3d_interarea_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    const uint ih, const uint oh,
    const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return downsample3d_interarea_c1(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return downsample3d_interarea_c2to3(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return downsample3d_interarea_c4(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return downsample3d_interarea_c5to7(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return downsample3d_interarea_c8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Downsample3D::InterareaX2(
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
    if ((iw & 1u) != 0 || iw > MAX_MAP_SIZE || (ih & 1u) != 0 || ih > MAX_MAP_SIZE || (id & 1u) != 0 || id > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    UInt32 ow = iw / 2, oh = ih / 2, od = id / 2;

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

        ret = downsample3d_interarea_cleq8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = downsample3d_interarea_aligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = downsample3d_interarea_unaligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh * od, y, 1.0f / 8.0f, y);
}
