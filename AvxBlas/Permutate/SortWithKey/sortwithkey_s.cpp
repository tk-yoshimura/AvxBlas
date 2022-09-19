#include "../../avxblas.h"
#include "../../constants.h"
#include "../../utils.h"
#include "sortwithkey.h"

using namespace System;

#pragma unmanaged

int sortwithkeyasc_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_s(n * s, k_ptr);

    if (!iscontains_nan) {
        return sortwithkeyasc_ignnan_s(n, s, v_ptr, k_ptr);
    }
    return (minimize_nan) 
        ? sortwithkeyasc_minnan_s(n, s, v_ptr, k_ptr) 
        : sortwithkeyasc_maxnan_s(n, s, v_ptr, k_ptr);
}

int sortwithkeydsc_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr, bool minimize_nan) {
    bool iscontains_nan = contains_nan_s(n * s, k_ptr);

    if (!iscontains_nan) {
        return sortwithkeydsc_ignnan_s(n, s, v_ptr, k_ptr);
    }
    return (minimize_nan)
        ? sortwithkeydsc_minnan_s(n, s, v_ptr, k_ptr)
        : sortwithkeydsc_maxnan_s(n, s, v_ptr, k_ptr);
}

#pragma managed

void AvxBlas::Permutate::SortWithKey(UInt32 n, UInt32 s, Array<float>^ k, Array<Int32>^ v, SortOrder order, SortNaNMode nan_mode) {
    if (n <= 0 || s <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, s);
    Util::CheckLength(n * s, k);
    Util::CheckLength(n * s, v);

    float* k_ptr = (float*)(k->Ptr.ToPointer());
    uint* v_ptr = (uint*)(v->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (order == SortOrder::Ascending) {
        ret = sortwithkeyasc_s(n, s, v_ptr, k_ptr, nan_mode == SortNaNMode::MinimizeNaN);
    }
    else {
        ret = sortwithkeydsc_s(n, s, v_ptr, k_ptr, nan_mode == SortNaNMode::MinimizeNaN);
    }

    Util::AssertReturnCode(ret);
}