#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_sum.cpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

float dotmul_n1_ps(const float* x_ptr, const float* y_ptr) {
    float ret = x_ptr[0] * y_ptr[0];

    return ret;
}

float dotmul_n2_ps(const float* x_ptr, const float* y_ptr) {
    float ret = x_ptr[0] * y_ptr[0] + x_ptr[1] * y_ptr[1];

    return ret;
}

float dotmul_n3_ps(const float* x_ptr, const float* y_ptr) {
    float ret = x_ptr[0] * y_ptr[0] + x_ptr[1] * y_ptr[1] + x_ptr[2] * y_ptr[2];

    return ret;
}

float dotmul_n4_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m128 s0 = _mm_mul_ps(_mm_load_ps(x_ptr), _mm_load_ps(y_ptr));
    float ret = _mm_sum4to1_ps(s0);

    return ret;
}

float dotmul_n5to7_ps(unsigned int n, const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (n >= AVX2_FLOAT_STRIDE || n <= AVX2_FLOAT_STRIDE / 2) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(n);

    __m256 s0 = _mm256_mul_ps(_mm256_maskload_ps(x_ptr, mask), _mm256_maskload_ps(y_ptr, mask));
    float ret = _mm256_sum8to1_ps(s0);

    return ret;
}

float dotmul_n8_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
    float ret = _mm256_sum8to1_ps(s0);

    return ret;
}

float dotmul_n16_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
    __m256 s1 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8));
    float ret = _mm256_sum16to1_ps(s0, s1);

    return ret;
}

float dotmul_n24_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
    __m256 s1 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8));
    __m256 s2 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 16), _mm256_load_ps(y_ptr + 16));
    float ret = _mm256_sum16to1_ps(s0, _mm256_add_ps(s1, s2));

    return ret;
}

float dotmul_n32_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
    __m256 s1 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8));
    __m256 s2 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 16), _mm256_load_ps(y_ptr + 16));
    __m256 s3 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 24), _mm256_load_ps(y_ptr + 24));
    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

float dotmul_n64_ps(const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_mul_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr));
    __m256 s1 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8));
    __m256 s2 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 16), _mm256_load_ps(y_ptr + 16));
    __m256 s3 = _mm256_mul_ps(_mm256_load_ps(x_ptr + 24), _mm256_load_ps(y_ptr + 24));

    s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 32), _mm256_load_ps(y_ptr + 32), s0);
    s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 40), _mm256_load_ps(y_ptr + 40), s1);
    s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 48), _mm256_load_ps(y_ptr + 48), s2);
    s3 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 56), _mm256_load_ps(y_ptr + 56), s3);

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

float dotmul_alignment_ps(unsigned int n, const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    while (n >= AVX2_FLOAT_STRIDE * 4) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 16), _mm256_load_ps(y_ptr + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 24), _mm256_load_ps(y_ptr + 24), s3);

        n -= AVX2_FLOAT_STRIDE * 4;
        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
    }
    if (n == AVX2_FLOAT_STRIDE * 3) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 16), _mm256_load_ps(y_ptr + 16), s2);
    }
    else if (n == AVX2_FLOAT_STRIDE * 2) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr + 8), _mm256_load_ps(y_ptr + 8), s1);
    }
    else if (n == AVX2_FLOAT_STRIDE) {
        s0 = _mm256_fmadd_ps(_mm256_load_ps(x_ptr), _mm256_load_ps(y_ptr), s0);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}

float dotmul_disorder_ps(unsigned int n, const float* x_ptr, const float* y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_set_mask(n & AVX2_FLOAT_REMAIN_MASK);

    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();

    while (n >= AVX2_FLOAT_STRIDE * 4) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 8), _mm256_loadu_ps(y_ptr + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 16), _mm256_loadu_ps(y_ptr + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 24), _mm256_loadu_ps(y_ptr + 24), s3);

        n -= AVX2_FLOAT_STRIDE * 4;
        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
    }
    if (n >= AVX2_FLOAT_STRIDE * 3) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 8), _mm256_loadu_ps(y_ptr + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 16), _mm256_loadu_ps(y_ptr + 16), s2);

        n -= AVX2_FLOAT_STRIDE * 3;
        x_ptr += AVX2_FLOAT_STRIDE * 3;
        y_ptr += AVX2_FLOAT_STRIDE * 3;
    }
    else if (n >= AVX2_FLOAT_STRIDE * 2) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + 8), _mm256_loadu_ps(y_ptr + 8), s1);

        n -= AVX2_FLOAT_STRIDE * 2;
        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
    }
    else if (n >= AVX2_FLOAT_STRIDE) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr), _mm256_loadu_ps(y_ptr), s0);

        n -= AVX2_FLOAT_STRIDE;
        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
    }
    {
        s3 = _mm256_fmadd_ps(_mm256_maskload_ps(x_ptr, mask), _mm256_maskload_ps(y_ptr, mask), s3);
    }

    float ret = _mm256_sum32to1_ps(s0, s1, s2, s3);

    return ret;
}