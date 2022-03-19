#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_matmul_s.hpp"
#include "../Inline/inline_imcol_s.hpp"

using namespace System;

#pragma unmanaged

#pragma region padnone

int conv1d_forward_padnone_n32x_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            matmul_n32x_s(ic * kw, oc, x_ptr + x * ic, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    return SUCCESS;
}

int conv1d_forward_padnone_aligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            matmul_aligned_s(ic * kw, oc, x_ptr + x * ic, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    return SUCCESS;
}

int conv1d_forward_padnone_unaligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int col_size = (ic * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padnone_unaligned_s(ic, kw, iw, x, x_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + x * oc);
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

int conv1d_forward_padzero_n32x_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padzero_n32x_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr);

            matmul_n32x_s(ic * kw, oc, col_ptr, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_forward_padzero_aligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padzero_aligned_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr);

            matmul_aligned_s(ic * kw, oc, col_ptr, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_forward_padzero_unaligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int col_size = (ic * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padzero_unaligned_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + x * oc);
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

int conv1d_forward_padedge_n32x_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padedge_n32x_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr);

            matmul_n32x_s(ic * kw, oc, col_ptr, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_forward_padedge_aligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float* col_ptr = (float*)_aligned_malloc((size_t)ic * kw * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padedge_aligned_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr);

            matmul_aligned_s(ic * kw, oc, col_ptr, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    _aligned_free(col_ptr);

    return SUCCESS;
}

int conv1d_forward_padedge_unaligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const unsigned int col_size = (ic * kw + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK;

    float* col_ptr = (float*)_aligned_malloc((size_t)col_size * sizeof(float), AVX2_ALIGNMENT);
    float* we_ptr = (float*)_aligned_malloc((size_t)col_size * oc * sizeof(float), AVX2_ALIGNMENT);
    if (col_ptr == nullptr || we_ptr == nullptr) {
        if (col_ptr != nullptr) _aligned_free(col_ptr);
        if (we_ptr != nullptr) _aligned_free(we_ptr);

        return FAILURE_BADALLOC;
    }
    zeroset_aligned_s(col_size, col_ptr);
    align_kernel_s(oc, ic * kw, col_size, w_ptr, we_ptr);

    const __m256i mask = _mm256_setmask_ps((ic * kw) & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            imcol1d_padedge_unaligned_s(ic, kw, iw, x, kw / 2, x_ptr, col_ptr, mask);

            matmul_aligned_s(col_size, oc, col_ptr, we_ptr, y_ptr + x * oc);
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

void AvxBlas::Convolution1D::Forward(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
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
    if (kw == 1 && padmode != PadMode::None) {
        throw gcnew System::ArgumentException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_DATA_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;

    Util::CheckProdOverflow(n, ic, iw);
    Util::CheckProdOverflow(n, oc, ow);
    Util::CheckProdOverflow(ic, oc, kw);

    Util::CheckLength(n * ic * iw, x);
    Util::CheckLength(n * oc * ow, y);
    Util::CheckLength(ic * oc * kw, w);

    Util::CheckDuplicateArray(x, w, y);

    if (kw == 1) {
        Dense::Forward(n * iw, ic, oc, x, w, y);
        return;
    }

    Array<float>^ transpose_w = gcnew Array<float>(w->Length, false);
    Transform::Transpose(1, kw, oc, ic, w, transpose_w);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* w_ptr = (const float*)(transpose_w->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    if ((ic % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type n32x");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            conv1d_forward_padnone_n32x_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Zero) {
            conv1d_forward_padzero_n32x_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Edge) {
            conv1d_forward_padedge_n32x_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
    }
    else if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            conv1d_forward_padnone_aligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Zero) {
            conv1d_forward_padzero_aligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
        else if (padmode == PadMode::Edge) {
            conv1d_forward_padedge_aligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr);
        }
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        if (padmode == PadMode::None) {
            if (conv1d_forward_padnone_unaligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
        }
        else if (padmode == PadMode::Zero) {
            if (conv1d_forward_padzero_unaligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
        }
        else if (padmode == PadMode::Edge) {
            if (conv1d_forward_padedge_unaligned_s(n, ic, oc, iw, ow, kw, x_ptr, w_ptr, y_ptr) == FAILURE_BADALLOC) {
                throw gcnew System::OutOfMemoryException();
            }
        }
    }

    transpose_w->~Array();
}