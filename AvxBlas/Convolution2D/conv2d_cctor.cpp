#include "../avxblas.h"

static AvxBlas::Convolution2D::Convolution2D() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::ErrorMessage::AvxNotSupported);
    }
}