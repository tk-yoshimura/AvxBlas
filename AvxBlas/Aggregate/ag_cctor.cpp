#include "../avxblas.h"

static AvxBlas::Aggregate::Aggregate() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::ErrorMessage::AvxNotSupported);
    }
}