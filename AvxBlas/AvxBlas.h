#pragma once

using namespace System;
using namespace System::Diagnostics;
using namespace System::Runtime::CompilerServices;

namespace AvxBlas {
    generic <typename T> where T : ValueType
    [DebuggerDisplayAttribute("{Overview,nq}")]
    public ref class Array {
        static Array();

        private:
        IntPtr ptr;
        UInt32 length;

        public:
        property IntPtr Ptr { IntPtr get(); }
        property UInt32 Length { UInt32 get(); }
        static property UInt32 MaxLength { UInt32 get(); }
        property UInt32 ByteSize { UInt32 get(); }
        static property UInt32 MaxByteSize { UInt32 get(); }
        static property UInt32 ElementSize { UInt32 get(); }

        static property UInt32 Alignment { UInt32 get(); }

        property bool IsValid { bool get(); }
        static property Type^ ElementType { Type^ get(); }

        property cli::array<T>^ Value { cli::array<T>^ get(); }

        property String^ Overview { String^ get(); }

        Array(UInt32 length);
        Array(Int32 length);
        Array(UInt32 length, bool zeroset);
        Array(Int32 length, bool zeroset);
        Array(cli::array<T>^ array);

        property T default[UInt32] { T get(UInt32 index); void set(UInt32 index, T value); }

        static operator Array^ (cli::array<T>^ array);
        static operator cli::array<T>^ (Array^ array);

        void Write(cli::array<T>^ array);
        void Read(cli::array<T>^ array);

        void Write(cli::array<T>^ array, UInt32 count);
        void Read(cli::array<T>^ array, UInt32 count);

        void Zeroset();
        void Zeroset(UInt32 count);
        void Zeroset(UInt32 index, UInt32 count);

        void CopyTo(Array^ array, UInt32 count);
        void CopyTo(UInt32 index, Array^ dst_array, UInt32 dst_index, UInt32 count);

        static void Copy(Array^ src_array, Array^ dst_array, UInt32 count);
        static void Copy(Array^ src_array, UInt32 src_index, Array^ dst_array, UInt32 dst_index, UInt32 count);

        String^ ToString() override;

        ~Array();

        !Array();
    };

    public ref class Util abstract sealed {
        internal:
        generic <typename T> where T : ValueType
        static void CheckLength(UInt32 length, ... cli::array<Array<T>^>^ arrays);
        
        generic <typename T> where T : ValueType
        static void CheckOutOfRange(UInt32 index, UInt32 length, ... cli::array<Array<T>^>^ arrays);
        
        generic <typename T> where T : ValueType
        static void CheckDuplicateArray(... cli::array<Array<T>^>^ arrays);

        static void CheckProdOverflow(... cli::array<UInt32>^ arrays);
        
        static property System::String^ AvxNotSupported { System::String^ get(); };
        static property System::String^ InvalidArrayLength { System::String^ get(); };
        static property System::String^ DuplicatedArray { System::String^ get(); };

        static UInt32 LCM(UInt32 a, UInt32 b);
        static UInt64 LCM(UInt64 a, UInt64 b);
        static UInt32 GCD(UInt32 a, UInt32 b);
        static UInt64 GCD(UInt64 a, UInt64 b);

        public:
        static property bool IsSupportedAVX { bool get(); };
        static property bool IsSupportedAVX2 { bool get(); };
        static property bool IsSupportedAVX512F { bool get(); };
        static property bool IsSupportedFMA { bool get(); };
    };

    public ref class Elementwise abstract sealed {
        static Elementwise();

        public:
        static void Abs(UInt32 n, Array<float>^ x, Array<float>^ y);
        static void Abs(UInt32 n, Array<double>^ x, Array<double>^ y);

        static void Add(UInt32 n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y);
        static void Add(UInt32 n, Array<double>^ x1, Array<double>^ x2, Array<double>^ y);
    };

    public ref class Vectorwise abstract sealed {
        static Vectorwise();

        public:
        static void Fill(UInt32 n, UInt32 stride, Array<float>^ v, Array<float>^ y);
        static void Fill(UInt32 n, UInt32 stride, Array<double>^ v, Array<double>^ y);
        
        static void Add(UInt32 n, UInt32 stride, Array<float>^ x, Array<float>^ v, Array<float>^ y);
        static void Add(UInt32 n, UInt32 stride, Array<double>^ x, Array<double>^ v, Array<double>^ y);
    };

    public ref class Constant abstract sealed {
        static Constant();

        public:
        static void Add(UInt32 n, Array<float>^ x, float c, Array<float>^ y);
        static void Add(UInt32 n, Array<double>^ x, double c, Array<double>^ y);
    };

    public ref class Aggregate abstract sealed {
        static Aggregate();

        public:
        static void Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y);
        static void Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<double>^ y);
    };

    public ref class Initialize abstract sealed {
        static Initialize();

        public:
        static void Clear(UInt32 n, float c, Array<float>^ y);
        static void Clear(UInt32 n, double c, Array<double>^ y);
        static void Clear(UInt32 index, UInt32 n, float c, Array<float>^ y);
        static void Clear(UInt32 index, UInt32 n, double c, Array<double>^ y);

        static void Zeroset(UInt32 n, Array<float>^ y);
        static void Zeroset(UInt32 n, Array<double>^ y);
        static void Zeroset(UInt32 index, UInt32 n, Array<float>^ y);
        static void Zeroset(UInt32 index, UInt32 n, Array<double>^ y);
    };
}