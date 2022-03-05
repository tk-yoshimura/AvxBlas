#include "../avxblas.h"

static AvxBlas::Elementwise::Elementwise() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::Util::AvxNotSupported);
    }
}