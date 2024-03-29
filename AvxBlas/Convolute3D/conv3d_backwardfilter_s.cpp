#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_kernelfma_ss.hpp"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_matmul_s.hpp"
#include "../Inline/inline_imcol_s.hpp"

using namespace System;

#pragma unmanaged

#pragma region padnone

int conv3d_backwardfilter_padnone_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_n32x_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_n32x_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padnone_n32x_s(ic, kw, iw, x, kh, ih, y, kd, id, z, x_ptr, col_ptr);

                    kernelfma_n32x_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padnone_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_aligned_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padnone_aligned_s(ic, kw, iw, x, kh, ih, y, kd, id, z, x_ptr, col_ptr);

                    kernelfma_aligned_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padnone_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    zeroset_aligned_s(col_size * oc, we_ptr);
    zeroset_aligned_s(col_size * oc, wc_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padnone_unaligned_s(ic, kw, iw, x, kh, ih, y, kd, id, z, x_ptr, col_ptr, mask);

                    kernelfma_aligned_ss(col_size, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), we_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    unalign_kernel_s(oc, col_size, ic * kw * kh * kd, we_ptr, w_ptr);

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

#pragma endregion padnone

#pragma region padzero

int conv3d_backwardfilter_padzero_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_n32x_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_n32x_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padzero_n32x_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr);

                    kernelfma_n32x_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padzero_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_aligned_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padzero_aligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr);

                    kernelfma_aligned_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padzero_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    zeroset_aligned_s(col_size * oc, we_ptr);
    zeroset_aligned_s(col_size * oc, wc_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padzero_unaligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr, mask);

                    kernelfma_aligned_ss(col_size, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), we_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    unalign_kernel_s(oc, col_size, ic * kw * kh * kd, we_ptr, w_ptr);

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

#pragma endregion padzero

#pragma region padedge

int conv3d_backwardfilter_padedge_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_n32x_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_n32x_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padedge_n32x_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr);

                    kernelfma_n32x_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padedge_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * kd * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(ic * kw * kh * kd * oc, w_ptr);
    zeroset_aligned_s(ic * kw * kh * kd * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padedge_aligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr);

                    kernelfma_aligned_ss(ic * kw * kh * kd, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), w_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    _aligned_free(col_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

int conv3d_backwardfilter_padedge_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    const uint id, const uint od, const uint kd,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh * kd + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    float* wc_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr || wc_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);
        if (wc_ptr != nullptr) _aligned_free(wc_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    zeroset_aligned_s(col_size * oc, we_ptr);
    zeroset_aligned_s(col_size * oc, wc_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint z = 0; z < od; z++) {
            for (uint y = 0; y < oh; y++) {
                for (uint x = 0; x < ow; x++) {
                    imcol3d_padedge_unaligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, kd, id, z, kd / 2, x_ptr, col_ptr, mask);

                    kernelfma_aligned_ss(col_size, oc, col_ptr, y_ptr + oc * (x + ow * (y + oh * z)), we_ptr, wc_ptr);
                }
            }
        }

        x_ptr += ic * iw * ih * id;
        y_ptr += oc * ow * oh * od;
    }

    unalign_kernel_s(oc, col_size, ic * kw * kh * kd, we_ptr, w_ptr);

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);
    _aligned_free(wc_ptr);

    return SUCCESS;
}

#pragma endregion padedge

#pragma managed

void AvxBlas::Convolute3D::BackwardFilter(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 id, UInt32 kw, UInt32 kh, UInt32 kd,
    PadMode padmode, Array<float>^ x, Array<float>^ dy, Array<float>^ dw) {

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

    Util::CheckLength(n * ic * iw * ih * id, x);
    Util::CheckLength(n * oc * ow * oh * od, dy);
    Util::CheckLength(ic * oc * kw * kh * kd, dw);

    Util::CheckDuplicateArray(x, dy, dw);

    if (kw == 1 && kh == 1 && kd == 1) {
        Dense::BackwardFilter(n * iw * ih * id, ic, oc, x, dy, dw);
        return;
    }

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* y_ptr = (const float*)(dy->Ptr.ToPointer());
    float* w_ptr = (float*)(dw->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((ic % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwardfilter_padnone_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwardfilter_padzero_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwardfilter_padedge_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
    }
    else if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwardfilter_padnone_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwardfilter_padzero_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwardfilter_padedge_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv3d_backwardfilter_padnone_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv3d_backwardfilter_padzero_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv3d_backwardfilter_padedge_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, id, od, kd, x_ptr, y_ptr, w_ptr);
        }
    }

    Util::AssertReturnCode(ret);
}