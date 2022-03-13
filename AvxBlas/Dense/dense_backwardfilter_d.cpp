#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline//inline_kernelfma_d.hpp"

using namespace System;

#pragma unmanaged

int dense_backwardfilter_n16x_d(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        kernelfma_n16x_d(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_aligned_d(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        kernelfma_aligned_d(ic, oc, x_ptr, y_ptr, w_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

int dense_backwardfilter_unaligned_d(
    const unsigned int n, const unsigned int ic, const unsigned int oc,
    const double* __restrict x_ptr, const double* __restrict y_ptr, double* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(ic & AVX2_DOUBLE_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        kernelfma_unaligned_d(ic, oc, x_ptr, y_ptr, w_ptr, mask);

        x_ptr += ic;
        y_ptr += oc;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Dense::BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ x, Array<double>^ dy, Array<double>^ dw) {
    if (n <= 0 || ic <= 0 || oc <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, ic);
    Util::CheckProdOverflow(n, oc);
    Util::CheckProdOverflow(ic, oc);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(dy->Ptr.ToPointer());
    double* w_ptr = (double*)(dw->Ptr.ToPointer());

    zeroset_d(ic * oc, w_ptr);

    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned x16");
#endif // _DEBUG

        dense_backwardfilter_n16x_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        dense_backwardfilter_aligned_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        dense_backwardfilter_unaligned_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
}