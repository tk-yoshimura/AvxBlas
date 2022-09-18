#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_matmul_s.hpp"
#include "../Inline/inline_imcol_s.hpp"

using namespace System;

#pragma unmanaged

void conv1d_transpose_kernel_s(
    const uint ic, const uint oc, const uint kw,
    infloats w_ptr, outfloats wt_ptr) {

    uint src_index = 0;

    for (uint j = 0; j < oc; j++) {
        for (uint kx = 0, rkx = kw - 1; kx < kw; kx++, rkx--) {
            uint dst_index = j + oc * rkx;

            for (uint i = 0; i < ic; i++) {
                wt_ptr[dst_index] = w_ptr[src_index];

                src_index++;
                dst_index += kw * oc;
            }
        }
    }
}

#pragma region padnone

int conv1d_backwarddata_padnone_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_n32x_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmul_n32x_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padnone_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_aligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmul_aligned_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padnone_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padnone

#pragma region padzero

int conv1d_backwarddata_padzero_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_n32x_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr);

            matmul_n32x_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padzero_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_aligned_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr);

            matmul_aligned_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padzero_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * x);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padzero

#pragma region padedge

int conv1d_backwarddata_padedge_n32x_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_n32x_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr);

            matmul_n32x_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }
        for (uint x = 0; x < kw / 2; x++) {
            imcol1d_padzero_n32x_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmuladd_n32x_s(oc * kw, ic, col_ptr, w_ptr, x_ptr);
        }
        for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
            imcol1d_padzero_n32x_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmuladd_n32x_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * (iw - 1));
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padedge_aligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)oc * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_aligned_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr);

            matmul_aligned_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * x);
        }
        for (uint x = 0; x < kw / 2; x++) {
            imcol1d_padzero_aligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmuladd_aligned_s(oc * kw, ic, col_ptr, w_ptr, x_ptr);
        }
        for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
            imcol1d_padzero_aligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr);

            matmuladd_aligned_s(oc * kw, ic, col_ptr, w_ptr, x_ptr + ic * (iw - 1));
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_backwarddata_padedge_unaligned_s(
    const uint n, const uint ic, const uint oc,
    const uint iw, const uint ow, const uint kw,
    infloats y_ptr, infloats w_ptr, outfloats x_ptr) {

#ifdef _DEBUG
    if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint col_size = (oc * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * ic * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(ic, oc * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((oc * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        for (uint x = 0; x < iw; x++) {
            imcol1d_padzero_unaligned_s(oc, kw, ow, x, kw / 2, y_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * x);
        }
        for (uint x = 0; x < kw / 2; x++) {
            imcol1d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr, mask);

            matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr);
        }
        for (uint x = iw + kw / 2; x < iw + kw - 1; x++) {
            imcol1d_padzero_unaligned_s(oc, kw, ow, x, kw - 1, y_ptr, col_ptr, mask);

            matmuladd_aligned_s(col_size, ic, col_ptr, we_ptr, x_ptr + ic * (iw - 1));
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);
    _aligned_free(we_ptr);

    return SUCCESS;
}

#pragma endregion padedge

#pragma managed

void AvxBlas::Convolute1D::BackwardData(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
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
    if (kw == 1 && padmode != PadMode::None) {
        throw gcnew System::ArgumentException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;

    Util::CheckProdOverflow(n, ic, iw);
    Util::CheckProdOverflow(n, oc, ow);
    Util::CheckProdOverflow(ic, oc, kw);

    Util::CheckLength(n * ic * iw, dx);
    Util::CheckLength(n * oc * ow, dy);
    Util::CheckLength(ic * oc * kw, w);

    Util::CheckDuplicateArray(dx, w, dy);

    if (kw == 1) {
        Dense::BackwardData(n * iw, ic, oc, dy, w, dx);
        return;
    }

    Array<float>^ transpose_w = gcnew Array<float>(w->Length, false);
    conv1d_transpose_kernel_s(ic, oc, kw, (const float*)(w->Ptr.ToPointer()), (float*)(transpose_w->Ptr.ToPointer()));

    const float* y_ptr = (const float*)(dy->Ptr.ToPointer());
    const float* w_ptr = (const float*)(transpose_w->Ptr.ToPointer());
    float* x_ptr = (float*)(dx->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((oc % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv1d_backwarddata_padnone_n32x_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv1d_backwarddata_padzero_n32x_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv1d_backwarddata_padedge_n32x_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
    }
    else if ((oc & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv1d_backwarddata_padnone_aligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv1d_backwarddata_padzero_aligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv1d_backwarddata_padedge_aligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            ret = conv1d_backwarddata_padnone_unaligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Zero) {
            ret = conv1d_backwarddata_padzero_unaligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
        else if (padmode == PadMode::Edge) {
            ret = conv1d_backwarddata_padedge_unaligned_s(n, ic, oc, iw, ow, kw, y_ptr, w_ptr, x_ptr);
        }
    }

    transpose_w->~Array();

    Util::AssertReturnCode(ret);
}