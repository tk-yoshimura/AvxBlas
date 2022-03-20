#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline//inline_kernelfma_s.hpp"

using namespace System;

#pragma unmanaged

int dense_backwardfilter_n1_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if (ic != 1 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint maskn = oc & AVX2_FLOAT_REMAIN_MASK;

    if (maskn > 0) {
        const __m256i mask = _mm256_setmask_ps(maskn);

        for (uint i = 0; i < n; i++) {
            kernelfma_n1_unaligned_s(ic, oc, x_ptr, y_ptr, w_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n1_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    return SUCCESS;
}

int dense_backwardfilter_n2_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if (ic != 2 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint maskn = (oc % (AVX2_FLOAT_STRIDE / 2)) * ic;

    if (maskn > 0) {
        const __m256i mask = _mm256_setmask_ps(maskn);

        for (uint i = 0; i < n; i++) {
            kernelfma_n2_unaligned_s(ic, oc, x_ptr, y_ptr, w_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n2_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    return SUCCESS;
}

int dense_backwardfilter_n3_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if (ic != 3 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (oc % 8 != 0) {
        uint maskn = 2 - (oc & 1);
        const __m256i mask = _mm256_setmask_ps(maskn * ic);

        for (uint i = 0; i < n; i++) {
            kernelfma_n3_unaligned_s(ic, oc, x_ptr, y_ptr, w_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n3_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    return SUCCESS;
}

int dense_backwardfilter_n4_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if (ic != 4 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (oc % 2 != 0) {
        const __m256i mask = _mm256_setmask_ps((oc % 2) * ic);

        for (uint i = 0; i < n; i++) {
            kernelfma_n4_unaligned_s(ic, oc, x_ptr, y_ptr, w_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n4_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    return SUCCESS;
}

int dense_backwardfilter_nleq4_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if (ic > AVX2_FLOAT_STRIDE / 2 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (ic == 1) {
        return dense_backwardfilter_n1_s(n, 1, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 2) {
        return dense_backwardfilter_n2_s(n, 2, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 3) {
        return dense_backwardfilter_n3_s(n, 3, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 4) {
        return dense_backwardfilter_n4_s(n, 4, oc, x_ptr, y_ptr, w_ptr);
    }

    return FAILURE_BADPARAM;
}

int dense_backwardfilter_n32x_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        kernelfma_n32x_s(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_aligned_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (uint i = 0; i < n; i++) {
        kernelfma_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_unaligned_s(
    const uint n, const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(ic & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        kernelfma_unaligned_s(ic, oc, x_ptr, y_ptr, w_ptr, mask);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Dense::BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ x, Array<float>^ dy, Array<float>^ dw) {
    if (n <= 0 || ic <= 0 || oc <= 0) {
        return;
    }

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (ic > MAX_CHANNELS || oc > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }

    Util::CheckProdOverflow(n, ic);
    Util::CheckProdOverflow(n, oc);
    Util::CheckProdOverflow(ic, oc);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* y_ptr = (const float*)(dy->Ptr.ToPointer());
    float* w_ptr = (float*)(dw->Ptr.ToPointer());

    zeroset_s(ic * oc, w_ptr);

    int ret = UNEXECUTED;

    if ((ic % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned x32");
#endif // _DEBUG

        ret = dense_backwardfilter_n32x_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = dense_backwardfilter_aligned_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if (ic <= AVX2_FLOAT_STRIDE / 2) {
#ifdef _DEBUG
        Console::WriteLine("type leq4");
#endif // _DEBUG

        ret = dense_backwardfilter_nleq4_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = dense_backwardfilter_unaligned_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }

    Util::AssertReturnCode(ret);
}