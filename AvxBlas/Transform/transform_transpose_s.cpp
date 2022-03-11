#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int transpose_s(
    const unsigned int n, const unsigned int r, const unsigned int s, const unsigned int stride, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {

    for (unsigned int th = 0; th < n; th++) {

        for (unsigned int k = 0; k < r; k++) {
            unsigned int offset = stride * k;

            for (unsigned int j = 0; j < s; j++) {
                for (unsigned int i = 0; i < stride; i++) {
                    y_ptr[offset + i] = x_ptr[i];
                }

                x_ptr += stride;
                offset += stride * r;
            }
        }

        y_ptr += r * s * stride;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Transform::Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || r <= 0 || s <= 0 || stride <= 0) {
        return;
    }

    Util::CheckProdOverflow(n, r, s, stride);
    Util::CheckLength(n * r * s * stride, x, y);
    Util::CheckDuplicateArray(x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    transpose_s(n, r, s, stride, x_ptr, y_ptr);
}