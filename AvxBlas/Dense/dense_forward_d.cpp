#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma managed

void AvxBlas::Dense::Forward(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ x, Array<double>^ w, Array<double>^ y) {
    Affine::Dotmul(n, oc, ic, x, w, y);
}