#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

int main(){
    //const unsigned int N = 1 << 16, SAMPLES = 16, STRIDE = 248;
    //
    //float* x = (float*)_aligned_malloc(N * SAMPLES * STRIDE * sizeof(float), AVX2_ALIGNMENT);
    //float* y1 = (float*)_aligned_malloc(N * STRIDE * sizeof(float), AVX2_ALIGNMENT);
    //float* y2 = (float*)_aligned_malloc(N * STRIDE * sizeof(float), AVX2_ALIGNMENT);
    //
    //if (x == nullptr || y1 == nullptr || y2 == nullptr) {
    //    return -1;
    //}
    //
    //for (int i = 0; i < N * SAMPLES * STRIDE; i++) {
    //    x[i] = rand();
    //}
    //
    //ag_sum_aligned_s_type1(N, SAMPLES, STRIDE, x, y1);
    //ag_sum_aligned_s_type2(N, SAMPLES, STRIDE, x, y2);
    //
    //for (int i = 0; i < N * STRIDE; i++) {
    //    if (y1[i] != y2[i]) {
    //        printf("mismatching!!!!\n");
    //        break;
    //    }
    //}
    //
    //printf("benchmark\n");
    //
    //for (int j = 0; j < 4; j++) {
    //    {
    //        auto start = std::chrono::system_clock::now();
    //
    //        int s = 0;
    //
    //        for (int i = 0; i < 128; i++) {
    //            int ret = ag_sum_aligned_s_type1(N, SAMPLES, STRIDE, x, y1);
    //
    //            s += ret;
    //        }
    //
    //        auto dur = std::chrono::system_clock::now() - start;
    //        auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    //
    //        std::cout << "type1 " << microsec << " micro sec \n";
    //        std::cout << "ret: " << s << "\n";
    //    }
    //
    //    {
    //        auto start = std::chrono::system_clock::now();
    //
    //        int s = 0;
    //
    //        for (int i = 0; i < 128; i++) {
    //            int ret = ag_sum_aligned_s_type2(N, SAMPLES, STRIDE, x, y2);
    //
    //            s += ret;
    //        }
    //
    //        auto dur = std::chrono::system_clock::now() - start;
    //        auto microsec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
    //
    //        std::cout << "type2 " << microsec << " micro sec \n";
    //        std::cout << "ret: " << s << "\n";
    //    }
    //}
    //
    //_aligned_free(x);
    //_aligned_free(y1);
    //_aligned_free(y2);

    uint kw = 3, kh = 3, kd = 1;
    uint cnt = 0;

    for (uint kx = 1 % kw, ky = (1 / kw) % kh, kz = 1 / (kw * kh); kz < kd; kx++, ky += kx / kw, kz += ky / kh, kx %= kw, ky %= kh) {
        printf("%d,%d,%d\n", kx, ky, kz);

        cnt++;
    }

    printf("cnt %d\n", cnt);

    printf("end");

    getchar();
}
