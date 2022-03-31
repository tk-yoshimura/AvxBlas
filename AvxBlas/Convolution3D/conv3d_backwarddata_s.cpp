#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_matmul_s.hpp"
#include "../Inline/inline_imcol_s.hpp"

using namespace System;

#pragma unmanaged

void conv3d_transpose_kernel_s(
    const uint ic, const uint oc, const uint kw, const uint kh, const uint kd,
    infloats w_ptr, outfloats wt_ptr) {

    uint src_index = 0;

    for (uint j = 0; j < oc; j++) {
        for (uint kz = 0, rkz = kd - 1; kz < kd; kz++, rkz--) {
            for (uint ky = 0, rky = kh - 1; ky < kh; ky++, rky--) {
                for (uint kx = 0, rkx = kw - 1; kx < kw; kx++, rkx--) {
                    uint dst_index = j + oc * (rkx + kw * (rky + kh * rkz));

                    for (uint i = 0; i < ic; i++) {
                        wt_ptr[dst_index] = w_ptr[src_index];

                        src_index++;
                        dst_index += kw * kh * kd * oc;
                    }
                }
            }
        }
    }
}

#pragma region padnone

int conv3d_backwarddata_padnone_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmul_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padnone_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmul_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padnone_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw * kh * kd, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padnone

#pragma region padzero

int conv3d_backwarddata_padzero_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmul_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padzero_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmul_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padzero_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw * kh * kd, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padzero

#pragma region padedge

int conv3d_backwarddata_padedge_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmul_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (y + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * z)));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ih * z));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ih * z);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ih * z));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ((ih - 1) + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * z)));
                }
            }
        }
        for (uint z = 0; z < kd / 2; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * y));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * y);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * y));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * x);
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (iw - 1));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (ih - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (ih - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (ih - 1)));
                }
            }
        }
        for (uint z = id + kd / 2; z < id + kd - 1; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (y + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * (id - 1))));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ih * (id - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ih * (id - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ih * (id - 1)));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ((ih - 1) + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_n32x_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_n32x_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * (id - 1))));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padedge_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmul_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (y + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * z)));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ih * z));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ih * z);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ih * z));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ((ih - 1) + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * z)));
                }
            }
        }
        for (uint z = 0; z < kd / 2; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * y));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * y);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * y));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * x);
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (iw - 1));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (ih - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (ih - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (ih - 1)));
                }
            }
        }
        for (uint z = id + kd / 2; z < id + kd - 1; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * (y + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * (y + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * (id - 1))));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ih * (id - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ih * (id - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ih * (id - 1)));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * iw * ((ih - 1) + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_aligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr);

                    matmuladd_aligned_s(oc * kw * kh * kd, ic, col_ptr, w_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * (id - 1))));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv3d_backwarddata_padedge_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw * kh * kd, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < id; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * (y + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * (y + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * z)));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * ih * z));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * ih * z);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * ih * z));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * z)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * ((ih - 1) + ih * z));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd / 2, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * z)));
                }
            }
        }
        for (uint z = 0; z < kd / 2; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * y));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * y);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * y));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * x);
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr);
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (iw - 1));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * (ih - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * (ih - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * (ih - 1)));
                }
            }
        }
        for (uint z = id + kd / 2; z < id + kd - 1; z++) {
            for (uint y = 0; y < ih; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * (y + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * (y + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh / 2, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * (y + ih * (id - 1))));
                }
            }
            for (uint y = 0; y < kh / 2; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * ih * (id - 1)));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * ih * (id - 1));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * ih * (id - 1)));
                }
            }
            for (uint y = ih + kh / 2; y < ih + kh - 1; y++) {
                for (uint x = 0; x < iw; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (x + iw * ((ih - 1) + ih * (id - 1))));
                }
                for (uint x = 0; x < kw / 2; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * iw * ((ih - 1) + ih * (id - 1)));
                }
                for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
                    imcol3d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, kh, oh, y, kh - 1, kd, od, z, kd - 1, y_ptr, col_ptr, mask);

                    matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * ((iw - 1) + iw * ((ih - 1) + ih * (id - 1))));
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padedge

#pragma managed

void AvxBlas::Convolution3D::BackwardData(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 id, UInt32 kw, UInt32 kh, UInt32 kd,
    PadMode padmode, Array<float>^ dy, Array<float>^ w, Array<float>^ dx) {

    if (!Enum::IsDefined(PadMode::typeid, padmode)) {
        throw gcnew System::ArgumentException(ErrorMessage::UndefinedEnum);
    }
    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (ic > MAX_CHANNELS || oc > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if ((kw & 1) == 0 || kw > MAX_KERNEL_SIZE || ((iw < kw) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if ((kh & 1) == 0 || kh > MAX_KERNEL_SIZE || ((ih < kh) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if ((kd & 1) == 0 || kd > MAX_KERNEL_SIZE || ((id < kd) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (kw == 1 && kh == 1 && kd == 1 && padmode != PadMode::None) {
        throw gcnew System::ArgumentException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE || ih > MAX_MAP_SIZE || id > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0 || ih <= 0 || id <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;
    UInt32 oh = padmode == PadMode::None ? (ih - kh + 1) : ih;
    UInt32 od = padmode == PadMode::None ? (id - kd + 1) : id;

    Util::CheckProdOverflow(n, ic, iw, ih, id);
    Util::CheckProdOverflow(n, oc, ow, oh, od);
    Util::CheckProdOverflow(ic, oc, kw, kh, kd);

    Util::CheckLength(n * ic * iw * ih * id, dx);
    Util::CheckLength(n * oc * ow * oh * od, dy);
    Util::CheckLength(ic * oc * kw * kh * kd, w);

    Util::CheckDuplicateArray(dx, w, dy);

    if (kw == 1 && kh == 1 && kd == 1) {
        Dense::BackwardData(n * iw * ih * id, ic, oc, dy, w, dx);
        return;
    }

    Array<float>^ transpose_w = gcnew Array<float>(w->Length, false);
    conv3d_transpose_kernel_s(ic, oc, kw, kh, kd, (const float*)(w->Ptr.ToPointer()), (float*)(transpose_w->Ptr.ToPointer()));

    const float* y_ptr = (const float*)(dy->Ptr.ToPointer());
    const float* w_ptr = (const float*)(transpose_w->Ptr.ToPointer());
    float* x_ptr = (float*)(dx->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((oc % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwarddata_padnone_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwarddata_padzero_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwarddata_padedge_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
    }
    else if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwarddata_padnone_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwarddata_padzero_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwarddata_padedge_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwarddata_padnone_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwarddata_padzero_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwarddata_padedge_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, y_ptr, w_ptr, x_ptr);
        }
    }

    transpose_w->~Array();

    Util::AssertReturnCode(ret);
}