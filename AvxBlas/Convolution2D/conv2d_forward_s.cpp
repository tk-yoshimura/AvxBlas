#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_matmul_s.hpp"
#include "../Inline/inline_imcol_s.hpp"

using namespace System;

#pragma unmanaged

#pragma region padnone

int conv2d_forward_padnone_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padnone_n32x_s(ic, kw, iw, x, kh, ih, y, x_ptr, col_ptr);

                matmul_n32x_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padnone_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padnone_aligned_s(ic, kw, iw, x, kh, ih, y, x_ptr, col_ptr);

                matmul_aligned_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padnone_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw * kh, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padnone_unaligned_s(ic, kw, iw, x, kh, ih, y, x_ptr, col_ptr, mask);

                matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padnone

#pragma region padzero

int conv2d_forward_padzero_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padzero_n32x_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr);

                matmul_n32x_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padzero_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padzero_aligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr);

                matmul_aligned_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padzero_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw * kh, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padzero_unaligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr, mask);

                matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padzero

#pragma region padedge

int conv2d_forward_padedge_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padedge_n32x_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr);

                matmul_n32x_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padedge_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * kh * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padedge_aligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr);

                matmul_aligned_s(ic * kw * kh, oc, col_ptr, w_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv2d_forward_padedge_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    const uint ih, const uint oh, const uint kh,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (ic * kw * kh + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw * kh, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint y = 0; y < oh; y++) {
            for (uint x = 0; x < ow; x++) {
                imcol2d_padedge_unaligned_s(ic, kw, iw, x, kw / 2, kh, ih, y, kh / 2, x_ptr, col_ptr, mask);

                matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + oc * (x + ow * y));
            }
        }

        x_ptr += ic * iw * ih;
        y_ptr += oc * ow * oh;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padedge

#pragma managed

void AvxBlas::Convolution2D::Forward(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 kw, UInt32 kh,
    PadMode padmode, Array<float>^ x, Array<float>^ w, Array<float>^ y) {

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
    if (kw == 1 && kh == 1 && padmode != PadMode::None) {
        throw gcnew System::ArgumentException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE || ih > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0 || ih <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;
    UInt32 oh = padmode == PadMode::None ? (ih - kh + 1) : ih;

    Util::CheckProdOverflow(n, ic, iw, ih);
    Util::CheckProdOverflow(n, oc, ow, oh);
    Util::CheckProdOverflow(ic, oc, kw, kh);

    Util::CheckLength(n * ic * iw * ih, x);
    Util::CheckLength(n * oc * ow * oh, y);
    Util::CheckLength(ic * oc * kw * kh, w);

    Util::CheckDuplicateArray(x, w, y);

    if (kw == 1 && kh == 1) {
        Dense::Forward(n * iw * ih, ic, oc, x, w, y);
        return;
    }

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* w_ptr = (const float*)(w->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((ic % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv2d_forward_padnone_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv2d_forward_padzero_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv2d_forward_padedge_n32x_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
    }
    else if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv2d_forward_padnone_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv2d_forward_padzero_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv2d_forward_padedge_aligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv2d_forward_padnone_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv2d_forward_padzero_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv2d_forward_padedge_unaligned_s(n, ic, oc, iw, ow, kw, ih, oh, kh, x_ptr, w_ptr, y_ptr);
        }
    }

    Util::AssertReturnCode(ret);
}