#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

using namespace System;
using namespace System::Runtime::CompilerServices;

namespace AvxBlas {
    extern __m256i masktable_m256(int k);
    extern __m128i masktable_m128(int k);

    public ref class Util abstract sealed {
        internal:
        static void CheckLength(unsigned int length, ... array<array<float>^>^ arrays);
        static void CheckOutOfRange(unsigned int index, unsigned int length, ... array<array<float>^>^ arrays);
        static void CheckDuplicateArray(... array<array<float>^>^ arrays);
        
        static property System::String^ AvxNotSupported { System::String^ get(); };
        static property System::String^ InvalidArrayLength { System::String^ get(); };
        static property System::String^ DuplicatedArray { System::String^ get(); };

        public:
        static property bool IsSupportedAVX { bool get(); };
        static property bool IsSupportedAVX2 { bool get(); };
        static property bool IsSupportedAVX512F { bool get(); };
        static property bool IsSupportedFMA { bool get(); };
    };

    public ref class Elementwise abstract sealed {
        static Elementwise();

        public:
        static void Add(unsigned int n, array<float>^ x1, array<float>^ x2, array<float>^ y);
    };
}