#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma managed

void AvxBlas::Convolution1D::BackwardData(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
    PadMode padmode, Array<float>^ dy, Array<float>^ w, Array<float>^ dx) {

    if (!Enum::IsDefined(PadMode::typeid, padmode)) {
        throw gcnew System::ArgumentException(Util::UndefinedEnum);
    }
    if ((kw & 1) == 0 || kw > MAX_KERNEL_SIZE || ((iw < kw) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(Util::InvalidKernelSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;

    Util::CheckProdOverflow(n, ic, iw);
    Util::CheckProdOverflow(n, oc, ow);
    Util::CheckProdOverflow(ic, oc, kw);

    Util::CheckLength(n * ic * iw, dx);
    Util::CheckLength(n * oc * ow, dy);
    Util::CheckLength(ic * oc * kw, w);
}