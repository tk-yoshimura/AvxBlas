#include "../avxblas.h"
using namespace System;

#pragma managed

UInt32 AvxBlas::Util::GCD(UInt32 a, UInt32 b) {
    if (b == 0) {
        return a;
    }

    return GCD(b, a % b);
}

UInt32 AvxBlas::Util::LCM(UInt32 a, UInt32 b) {
    UInt32 c = a / GCD(a, b);

    if (b > 0 && c > (~0u) / b) {
        throw gcnew System::OverflowException();
    }

    return c * b;
}

UInt64 AvxBlas::Util::GCD(UInt64 a, UInt64 b) {
    if (b == 0) {
        return a;
    }

    return GCD(b, a % b);
}

UInt64 AvxBlas::Util::LCM(UInt64 a, UInt64 b) {
    UInt64 c = a / GCD(a, b);

    if (b > 0 && c > (~0ul) / b) {
        throw gcnew System::OverflowException();
    }

    return c * b;
}