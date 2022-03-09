#include "../avxblas.h"

static AvxBlas::Affine::Affine() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::Util::AvxNotSupported);
    }
}