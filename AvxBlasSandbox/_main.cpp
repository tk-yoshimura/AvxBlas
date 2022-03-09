#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

void add_stride8_test(const unsigned int N, float* x1, float* x2, float* y){
    int s = add_stride8_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; i++) {
        int ret = add_stride8_s(N, x1, x2, y);

        ret += s;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void add_stride16_test(const unsigned int N, float* x1, float* x2, float* y) {
    int s = add_stride16_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; i++) {
        int ret = add_stride16_s(N, x1, x2, y);

        ret += s;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void add_stride32_test(const unsigned int N, float* x1, float* x2, float* y) {
    int s = add_stride32_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000; i++) {
        int ret = add_stride32_s(N, x1, x2, y);

        ret += s;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}


int main(){
    const unsigned int N = 1 << 24;

    float* x1 = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);
    float* x2 = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);
    float* y  = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);

    add_stride8_test(N, x1, x2, y);

    std::cout << y[N - 1] << std::endl;

    add_stride16_test(N, x1, x2, y);

    std::cout << y[N - 1] << std::endl;

    add_stride32_test(N, x1, x2, y);

    std::cout << y[N - 1] << std::endl;

    _aligned_free(x1);
    _aligned_free(x2);
    _aligned_free(y);

    getchar();
}
