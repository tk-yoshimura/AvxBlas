#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

void add_stride8_test(const unsigned int N, float* x1, float* x2, float* y){
    int s = add_stride8_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        int ret = add_stride8_s(N, x1, x2, y);

        s += ret;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void add_stride16_test(const unsigned int N, float* x1, float* x2, float* y) {
    int s = add_stride16_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        int ret = add_stride16_s(N, x1, x2, y);

        s += ret;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void add_stride32_test(const unsigned int N, float* x1, float* x2, float* y) {
    int s = add_stride32_s(N, x1, x2, y);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        int ret = add_stride32_s(N, x1, x2, y);

        s += ret;
    }

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void dotmul_stride8_test(const unsigned int N, float* x1, float* x2) {
    double s = dotmul_stride8_s(N, x1, x2);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        float ret = dotmul_stride8_s(N, x1, x2);

        s += ret;
    }

    s /= 1025;

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << "dotmul_stride8" << std::endl;
    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void dotmul_stride16_test(const unsigned int N, float* x1, float* x2) {
    double s = dotmul_stride16_s(N, x1, x2);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        float ret = dotmul_stride16_s(N, x1, x2);

        s += ret;
    }

    s /= 1025;

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << "dotmul_stride16" << std::endl;
    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}

void dotmul_stride32_test(const unsigned int N, float* x1, float* x2) {
    double s = dotmul_stride32_s(N, x1, x2);

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1024; i++) {
        float ret = dotmul_stride32_s(N, x1, x2);

        s += ret;
    }

    s /= 1025;

    auto dur = std::chrono::system_clock::now() - start;
    auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();

    std::cout << "dotmul_stride32" << std::endl;
    std::cout << microsec << " micro sec \n";
    std::cout << "ret: " << s << "\n";
}


int main(){
    const unsigned int N = 1 << 24;

    float* x1 = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);
    float* x2 = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);
    //float* y  = (float*)_aligned_malloc(N * sizeof(float), AVX2_ALIGNMENT);

    if (x1 == nullptr || x2 == nullptr) {
        return -1;
    }

    srand(time(NULL));

    double s = 0;

    for (int i = 0; i < N; i++) {
        x1[i] = rand() / (float)RAND_MAX;
        x2[i] = rand() / (float)RAND_MAX;

        s += (double)x1[i] * (double)x2[i];
    }

    std::cout << "expected: " << s << std::endl;

    for (int i = 0; i < 20; i++) {
        dotmul_stride8_test(N, x1, x2);
        dotmul_stride16_test(N, x1, x2);
        dotmul_stride32_test(N, x1, x2);
    }

    _aligned_free(x1);
    _aligned_free(x2);

    getchar();
}
