#pragma once

#pragma warning(disable: 4793)

#include <immintrin.h>

using namespace System;
using namespace System::Diagnostics;
using namespace System::Runtime::CompilerServices;

namespace AvxBlas {
    [assembly:InternalsVisibleTo("AvxBlasTest")] ;

    extern __m256i masktable_m256(int k);
    extern __m128i masktable_m128(int k);

    generic <typename T> where T : ValueType
    [DebuggerDisplayAttribute("{Overview,nq}")]
    public ref class Array {
        static Array();

        private:
        IntPtr ptr;
        UInt64 length;

        internal:
        property IntPtr Ptr { IntPtr get(); }

        public:
        property UInt64 Length { UInt64 get(); }
        static property UInt64 MaxLength { UInt64 get(); }
        property UInt64 ByteSize { UInt64 get(); }
        static property UInt64 MaxByteSize { UInt64 get(); }
        static property UInt64 ElementSize { UInt64 get(); }

        static property UInt64 Alignment { UInt64 get(); }

        property bool IsValid { bool get(); }
        static property Type^ ElementType { Type^ get(); }

        property cli::array<T>^ Value { cli::array<T>^ get(); }

        property String^ Overview { String^ get(); }

        Array(UInt64 length);
        Array(Int64 length);
        Array(UInt32 length);
        Array(Int32 length);
        Array(cli::array<T>^ array);

        property T default[UInt64] { T get(UInt64 index); void set(UInt64 index, T value); }

        static operator Array^ (cli::array<T>^ array);
        static operator cli::array<T>^ (Array^ array);

        void Write(cli::array<T>^ array);
        void Read(cli::array<T>^ array);

        void Write(cli::array<T>^ array, UInt64 count);
        void Read(cli::array<T>^ array, UInt64 count);

        void Zeroset();
        void Zeroset(UInt64 count);
        void Zeroset(UInt64 index, UInt64 count);

        void CopyTo(Array^ array, UInt64 count);
        void CopyTo(UInt64 index, Array^ dst_array, UInt64 dst_index, UInt64 count);

        static void Copy(Array^ src_array, Array^ dst_array, UInt64 count);
        static void Copy(Array^ src_array, UInt64 src_index, Array^ dst_array, UInt64 dst_index, UInt64 count);

        String^ ToString() override;

        ~Array();

        !Array();
    };

    public ref class Util abstract sealed {
        internal:
        generic <typename T> where T : ValueType
        static void CheckLength(unsigned int length, ... cli::array<Array<T>^>^ arrays);
        
        generic <typename T> where T : ValueType
        static void CheckOutOfRange(unsigned int index, unsigned int length, ... cli::array<Array<T>^>^ arrays);
        
        generic <typename T> where T : ValueType
        static void CheckDuplicateArray(... cli::array<Array<T>^>^ arrays);
        
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
        static void Add(unsigned int n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y);
        static void Add(unsigned int n, Array<double>^ x1, Array<double>^ x2, Array<double>^ y);

        static void Abs(unsigned int n, Array<float>^ x, Array<float>^ y);
        static void Abs(unsigned int n, Array<double>^ x, Array<double>^ y);
    };

    public ref class Initialize abstract sealed {
        static Initialize();

        public:
        static void Clear(unsigned int n, float c, Array<float>^ y);
        static void Clear(unsigned int n, double c, Array<double>^ y);
        static void Clear(unsigned int index, unsigned int n, float c, Array<float>^ y);
        static void Clear(unsigned int index, unsigned int n, double c, Array<double>^ y);

        static void Zeroset(unsigned int n, Array<float>^ y);
        static void Zeroset(unsigned int n, Array<double>^ y);
        static void Zeroset(unsigned int index, unsigned int n, Array<float>^ y);
        static void Zeroset(unsigned int index, unsigned int n, Array<double>^ y);
    };
}