#include "../../AvxBlas.h"

static AvxBlas::Initialize::Initialize() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::Util::AvxNotSupported);
    }
}