#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline//inline_kernelfma_s.hpp"

using namespace System;

#pragma unmanaged

#pragma managed

void AvxBlas::Convolution1D::BackwardFilter(
    UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw, 
    PadMode padmode, Array<float>^ x, Array<float>^ dy, Array<float>^ dw) {

    if (!Enum::IsDefined(PadMode::typeid, padmode)) {
        throw gcnew System::ArgumentException(ErrorMessage::UndefinedEnum);
    }
    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (ic > MAX_CHANNELS || oc > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if ((kw & 1) == 0 || kw > MAX_KERNEL_SIZE || ((iw < kw) && padmode == PadMode::None)) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_DATA_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || ic <= 0 || oc <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = padmode == PadMode::None ? (iw - kw + 1) : iw;

    Util::CheckProdOverflow(n, ic, iw);
    Util::CheckProdOverflow(n, oc, ow);
    Util::CheckProdOverflow(ic, oc, kw);

    Util::CheckLength(n * ic * iw, x);
    Util::CheckLength(n * oc * ow, dy);
    Util::CheckLength(ic * oc * kw, dw);

    throw gcnew System::NotImplementedException();
}