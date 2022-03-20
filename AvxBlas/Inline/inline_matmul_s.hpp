#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dotmul_s.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void matmul_n32x_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_n32x_s(ic, x_ptr, w_ptr);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline void matmul_aligned_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_aligned_s(ic, x_ptr, w_ptr);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline float matmul_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr, const __m256i mask) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_unaligned_s(ic, x_ptr, w_ptr, mask);

        *y_ptr = y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline void matmuladd_n32x_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_n32x_s(ic, x_ptr, w_ptr);

        *y_ptr += y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline void matmuladd_aligned_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_aligned_s(ic, x_ptr, w_ptr);

        *y_ptr += y;

        y_ptr++;
        w_ptr += ic;
    }
}

__forceinline float matmuladd_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    infloats x_ptr, infloats w_ptr, outfloats y_ptr, const __m256i mask) {

    for (unsigned int i = 0; i < oc; i++) {
        float y = dotmul_unaligned_s(ic, x_ptr, w_ptr, mask);

        *y_ptr += y;

        y_ptr++;
        w_ptr += ic;
    }
}