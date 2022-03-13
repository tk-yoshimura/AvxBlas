#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline//inline_kernelfma_s.hpp"

using namespace System;

#pragma unmanaged

int dense_backwardfilter_n32x_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        kernelfma_n32x_s(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_aligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        kernelfma_aligned_s(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_unaligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(ic & AVX2_FLOAT_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
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

    Util::CheckProdOverflow(n, ic);
    Util::CheckProdOverflow(n, oc);
    Util::CheckProdOverflow(ic, oc);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(dy->Ptr.ToPointer());
    float* w_ptr = (float*)(dw->Ptr.ToPointer());

    zeroset_s(ic * oc, w_ptr);

    if ((ic % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned x32");
#endif // _DEBUG

        dense_backwardfilter_n32x_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        dense_backwardfilter_aligned_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        dense_backwardfilter_unaligned_s(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
}