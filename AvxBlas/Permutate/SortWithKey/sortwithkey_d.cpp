#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "sortwithkey.h"

using namespace System;

#pragma unmanaged

int sortwithkeyasc_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_d(n * s, k_ptr);

    if (!iscontains_nan) {
        return sortwithkeyasc_ignnan_d(n, s, v_ptr, k_ptr);
    }
    return (minimize_nan)
        ? sortwithkeyasc_minnan_d(n, s, v_ptr, k_ptr)
        : sortwithkeyasc_maxnan_d(n, s, v_ptr, k_ptr);
}

int sortwithkeydsc_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_d(n * s, k_ptr);

    if (!iscontains_nan) {
        return sortwithkeydsc_ignnan_d(n, s, v_ptr, k_ptr);
    }
    return (minimize_nan)
        ? sortwithkeydsc_minnan_d(n, s, v_ptr, k_ptr)
        : sortwithkeydsc_maxnan_d(n, s, v_ptr, k_ptr);
}

#pragma managed

void AvxBlas::Permutate::SortWithKey(UInt32 n, UInt32 s, Array<double>^ k, Array<Int32>^ v, SortOrder order, SortNanMode nan_mode) {
    if (n <= 0 || s <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, s);
    Util::CheckLength(n * s, k);
    Util::CheckLength(n * s, v);

    Array<Int64>^ vl = gcnew Array<Int64>(n * s, false);

    Cast::AsType(n * s, v, vl);

    double* k_ptr = (double*)(k->Ptr.ToPointer());
    ulong* v_ptr = (ulong*)(vl->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (order == SortOrder::Ascending) {
        ret = sortwithkeyasc_d(n, s, v_ptr, k_ptr, nan_mode == SortNanMode::MinimizeNaN);
    }
    else {
        ret = sortwithkeydsc_d(n, s, v_ptr, k_ptr, nan_mode == SortNanMode::MinimizeNaN);
    }

#if _DEBUG
    vl->CheckOverflow();
#endif

    Cast::AsType(n * s, vl, v);

    vl->~Array();

    Util::AssertReturnCode(ret);
}