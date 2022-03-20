#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dotmul_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void matmul_n16x_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles w_ptr, outdoubles y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        double y = dotmul_n16x_d(ic, x_ptr, w_ptr);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline void matmul_aligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles w_ptr, outdoubles y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        double y = dotmul_aligned_d(ic, x_ptr, w_ptr);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline double matmul_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles w_ptr, outdoubles y_ptr, const __m256i mask) {

    for (unsigned int i = 0; i < oc; i++) {
        double y = dotmul_unaligned_d(ic, x_ptr, w_ptr, mask);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}