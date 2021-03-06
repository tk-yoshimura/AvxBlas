#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_kernelfma_dd.hpp"

using namespace System;

#pragma unmanaged

int dense_backwardfilter_n1_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 1 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc(((size_t)ic * oc + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    uint maskn = oc & AVX2_DOUBLE_REMAIN_MASK;

    if (maskn > 0) {
        const __m256i mask = _mm256_setmask_pd(maskn);

        for (uint i = 0; i < n; i++) {
            kernelfma_n1_unaligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n1_aligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_n2_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 2 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc(((size_t)ic * oc + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    uint maskn = (oc % (AVX2_DOUBLE_STRIDE / 2)) * ic;

    if (maskn > 0) {
        const __m256i mask = _mm256_setmask_pd(maskn);

        for (uint i = 0; i < n; i++) {
            kernelfma_n2_unaligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n2_aligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_n3_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 3 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc(((size_t)ic * oc + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    if (oc % AVX2_DOUBLE_STRIDE != 0) {
        const __m256i mask = _mm256_setmask_pd((ic * oc) & AVX2_DOUBLE_REMAIN_MASK);

        for (uint i = 0; i < n; i++) {
            kernelfma_n3_unaligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr, mask);

            x_ptr += ic;
            y_ptr += oc;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            kernelfma_n3_aligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

            x_ptr += ic;
            y_ptr += oc;
        }
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_n4_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 4 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc((size_t)ic * oc * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        kernelfma_n4_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_nleq4_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic > AVX2_DOUBLE_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (ic == 1) {
        return dense_backwardfilter_n1_d(n, 1, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 2) {
        return dense_backwardfilter_n2_d(n, 2, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 3) {
        return dense_backwardfilter_n3_d(n, 3, oc, x_ptr, y_ptr, w_ptr);
    }
    if (ic == 4) {
        return dense_backwardfilter_n4_d(n, 4, oc, x_ptr, y_ptr, w_ptr);
    }

    return FAILURE_BADPARAM;
}

int dense_backwardfilter_n16x_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc((size_t)ic * oc * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        kernelfma_n16x_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_aligned_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc((size_t)ic * oc * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    for (uint i = 0; i < n; i++) {
        kernelfma_aligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr);

        x_ptr += ic;
        y_ptr += oc;
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

int dense_backwardfilter_unaligned_d(
    const uint n, const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    double* wc_ptr = (double*)_aligned_malloc(((size_t)ic * oc + AVX2_DOUBLE_STRIDE) * sizeof(double), AVX2_ALIGNMENT);
    if (wc_ptr == nullptr) {
        return FAILURE_BADALLOC;
    }
    zeroset_d(ic * oc, w_ptr);
    zeroset_d(ic * oc, wc_ptr);

    const __m256i mask = _mm256_setmask_pd(ic & AVX2_DOUBLE_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        kernelfma_unaligned_dd(ic, oc, x_ptr, y_ptr, w_ptr, wc_ptr, mask);

        x_ptr += ic;
        y_ptr += oc;
    }

    _aligned_free(wc_ptr);

    return SUCCESS;
}

#pragma managed

void AvxBlas::Dense::BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ x, Array<double>^ dy, Array<double>^ dw) {
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

    Util::CheckDuplicateArray(x, dy, dw);

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    const double* y_ptr = (const double*)(dy->Ptr.ToPointer());
    double* w_ptr = (double*)(dw->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned x16");
#endif // _DEBUG

        ret = dense_backwardfilter_n16x_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = dense_backwardfilter_aligned_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else if (ic <= AVX2_DOUBLE_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq4");
#endif // _DEBUG

        ret = dense_backwardfilter_nleq4_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = dense_backwardfilter_unaligned_d(n, ic, oc, x_ptr, y_ptr, w_ptr);
    }

    Util::AssertReturnCode(ret);
}