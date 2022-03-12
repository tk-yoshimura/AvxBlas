#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma managed

void AvxBlas::Dense::Forward(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ x, Array<float>^ w, Array<float>^ y) {
    Affine::Dotmul(n, oc, ic, x, w, y);
}