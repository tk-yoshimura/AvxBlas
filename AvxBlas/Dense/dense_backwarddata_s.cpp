#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma managed

void AvxBlas::Dense::BackwardData(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ dy, Array<float>^ w, Array<float>^ dx) {
    Array<float>^ transpose_w = gcnew Array<float>(w->Length, false);
    Transform::Transpose(1, oc, ic, 1, w, transpose_w);

    Affine::Dotmul(n, ic, oc, dy, transpose_w, dx);

    transpose_w->~Array();
}