#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"

int main(){
    const uint iw = 3, kw = 3, pw = (kw - 1) / 2;

    for (uint ix = 0; ix < iw; ix++) {
        for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {
            printf("%d, %d\n", x, kx);
        }

        printf("\n");
    }

    printf("end");

    getchar();
}
