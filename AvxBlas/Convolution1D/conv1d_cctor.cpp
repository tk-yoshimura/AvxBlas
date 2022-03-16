#include "../avxblas.h"

static AvxBlas::Convolution1D::Convolution1D() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::ErrorMessage::AvxNotSupported);
    }
}