#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "../AvxBlas/Inline/inline_max_d.hpp"
#include "../AvxBlas/types.h"

int main(){
    __m256d x0 = _mm256_setr_pd(-14, -13, -12, -11);

    double s = _mm256_max4to1_pd(x0);
    
    printf("end");
    getchar();
}
