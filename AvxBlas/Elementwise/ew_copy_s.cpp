#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_copy_s(
    unsigned int n, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    if (x_ptr == y_ptr) {
        return SUCCESS;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG
    
    while (n >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0 = _mm256_load_ps(x_ptr);
        __m256 x1 = _mm256_load_ps(x_ptr + 8);
        __m256 x2 = _mm256_load_ps(x_ptr + 16);
        __m256 x3 = _mm256_load_ps(x_ptr + 24);

        _mm256_stream_ps(y_ptr, x0);
        _mm256_stream_ps(y_ptr + 8, x1);
        _mm256_stream_ps(y_ptr + 16, x2);
        _mm256_stream_ps(y_ptr + 24, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        n -= AVX2_FLOAT_STRIDE * 4;
    }
    if (n >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0 = _mm256_load_ps(x_ptr);
        __m256 x1 = _mm256_load_ps(x_ptr + 8);

        _mm256_stream_ps(y_ptr, x0);
        _mm256_stream_ps(y_ptr + 8, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        n -= AVX2_FLOAT_STRIDE * 2;
    }
    if (n >= AVX2_FLOAT_STRIDE) {
        __m256 x0 = _mm256_load_ps(x_ptr);

        _mm256_stream_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        n -= AVX2_FLOAT_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n);

        __m256 x = _mm256_maskload_ps(x_ptr, mask);

        _mm256_maskstore_ps(y_ptr, mask, x);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Copy(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_copy_s(n, x_ptr, y_ptr);
}