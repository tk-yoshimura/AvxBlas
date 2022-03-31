#include "../avxblas.h"

static AvxBlas::Convolution3D::Convolution3D() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::ErrorMessage::AvxNotSupported);
    }
}