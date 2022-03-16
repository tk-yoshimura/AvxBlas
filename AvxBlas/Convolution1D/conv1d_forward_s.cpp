#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_alignceil.hpp"
#include "../Inline/inline_matmul_s.hpp"

using namespace System;

#pragma unmanaged

int conv1d_forward_padnone_aligned_s(
    const unsigned int n, const unsigned int ic, const unsigned int oc, 
    const unsigned int iw, const unsigned int ow, const unsigned int kw,
    const float* __restrict x_ptr, const float* __restrict w_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int x = 0; x < ow; x++) {
            matmul_aligned_s(ic * kw, oc, x_ptr + x * ic, w_ptr, y_ptr + x * oc);
        }

        x_ptr += ic * iw;
        y_ptr += oc * ow;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Convolution1D::Forward(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
    PadMode padmode, Array<float>^ x, Array<float>^ w, Array<float>^ y) {

    if (!Enum::IsDefined(PadMode::typeid, padmode)) {
        throw gcnew System::ArgumentException(ErrorMessage::UndefinedEnum);
    }
    if ((kw & 1) == 0 || kw > MAX_KERNEL_SIZE || ((iw < kw) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_DATA_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;

    Util::CheckProdOverflow(n, ic, iw);
    Util::CheckProdOverflow(n, oc, ow);
    Util::CheckProdOverflow(ic, oc, kw);

    Util::CheckLength(n * ic * iw, x);
    Util::CheckLength(n * oc * ow, y);
    Util::CheckLength(ic * oc * kw, w);

    Util::CheckDuplicateArray(x, w, y);

    Array<float>^ transpose_w = gcnew Array<float>(w->Length, false);
    Transform::Transpose(1, kw, oc, ic, w, transpose_w);



    transpose_w->~Array();
}