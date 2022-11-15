#include "../avxblas.h"

using namespace System;

void AvxBlas::Aggregate::Average(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y) {
    if (n <= 0 || stride <= 0) {
        return;
    }
    
    Aggregate::Sum(n, samples, stride, x, y);
    Constant::Mul(n * stride, y, (float)(1.0 / samples), y);
}
