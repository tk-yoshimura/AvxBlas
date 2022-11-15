#include "../avxblas.h"

void AvxBlas::Aggregate::Average(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<double>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }

    Aggregate::Sum(n, samples, stride, x, y);
    Constant::Mul(n * stride, y, (double)(1.0 / samples), y);
}