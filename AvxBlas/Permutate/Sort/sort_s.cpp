#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "sort.h"

using namespace System;

#pragma unmanaged

int sortasc_s(const uint n, const uint s, float* y_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_s(n * s, y_ptr);

    if (!iscontains_nan) {
        return sortasc_ignnan_s(n, s, y_ptr);
    }
    return (minimize_nan)
        ? sortasc_minnan_s(n, s, y_ptr)
        : sortasc_maxnan_s(n, s, y_ptr);
}

int sortdsc_s(const uint n, const uint s, float* y_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_s(n * s, y_ptr);

    if (!iscontains_nan) {
        return sortdsc_ignnan_s(n, s, y_ptr);
    }
    return (minimize_nan)
        ? sortdsc_minnan_s(n, s, y_ptr)
        : sortdsc_maxnan_s(n, s, y_ptr);
}

#pragma managed

void AvxBlas::Permutate::Sort(UInt32 n, UInt32 s, Array<float>^ y, SortOrder order, SortNanMode nan_mode) {
    if (n <= 0 || s <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, s);
    Util::CheckLength(n * s, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (order == SortOrder::Ascending) {
        ret = sortasc_s(n, s, y_ptr, nan_mode == SortNanMode::MinimizeNaN);
    }
    else {
        ret = sortdsc_s(n, s, y_ptr, nan_mode == SortNanMode::MinimizeNaN);
    }

    Util::AssertReturnCode(ret);
}