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
        UInt64 allocsize;

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

        static operator Array ^ (cli::array<T>^ array);
        static operator cli::array<T> ^ (Array^ array);

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

#ifdef _DEBUG
        void SetupCanary();
        void CheckOverflow();
#endif // _DEBUG

        String^ ToString() override;

        ~Array();

        !Array();
    };

    public enum class PadMode {
        None,
        Zero,
        Edge
    };

    public ref class Elementwise abstract sealed {
        public:
        static void Copy(UInt32 n, Array<float>^ x, Array<float>^ y);
        static void Copy(UInt32 n, Array<double>^ x, Array<double>^ y);

        static void Abs(UInt32 n, Array<float>^ x, Array<float>^ y);
        static void Abs(UInt32 n, Array<double>^ x, Array<double>^ y);

        static void Add(UInt32 n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y);
        static void Add(UInt32 n, Array<double>^ x1, Array<double>^ x2, Array<double>^ y);
    };

    public ref class Vectorwise abstract sealed {
        public:
        static void Fill(UInt32 n, UInt32 stride, Array<float>^ v, Array<float>^ y);
        static void Fill(UInt32 n, UInt32 stride, Array<double>^ v, Array<double>^ y);

        static void Add(UInt32 n, UInt32 stride, Array<float>^ x, Array<float>^ v, Array<float>^ y);
        static void Add(UInt32 n, UInt32 stride, Array<double>^ x, Array<double>^ v, Array<double>^ y);
    };

    public ref class Constant abstract sealed {
        public:
        static void Add(UInt32 n, Array<float>^ x, float c, Array<float>^ y);
        static void Add(UInt32 n, Array<double>^ x, double c, Array<double>^ y);

        static void Mul(UInt32 n, Array<float>^ x, float c, Array<float>^ y);
        static void Mul(UInt32 n, Array<double>^ x, double c, Array<double>^ y);
    };

    public ref class Aggregate abstract sealed {
        public:
        static void Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<float>^ x, Array<float>^ y);
        static void Sum(UInt32 n, UInt32 samples, UInt32 stride, Array<double>^ x, Array<double>^ y);
    };

    public ref class Transform abstract sealed {
        public:
        static void Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<float>^ x, Array<float>^ y);
        static void Transpose(UInt32 n, UInt32 r, UInt32 s, UInt32 stride, Array<double>^ x, Array<double>^ y);
    };

    public ref class Affine abstract sealed {
        public:
        static void Dotmul(UInt32 na, UInt32 nb, UInt32 stride, Array<float>^ a, Array<float>^ b, Array<float>^ y);
        static void Dotmul(UInt32 na, UInt32 nb, UInt32 stride, Array<double>^ a, Array<double>^ b, Array<double>^ y);
    };

    public ref class Dense abstract sealed {
        public:
        static void Forward(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ x, Array<float>^ w, Array<float>^ y);
        static void Forward(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ x, Array<double>^ w, Array<double>^ y);

        static void BackwardData(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ dy, Array<float>^ w, Array<float>^ dx);
        static void BackwardData(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ dy, Array<double>^ w, Array<double>^ dx);
        
        static void BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, Array<float>^ x, Array<float>^ dy, Array<float>^ dw);
        static void BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, Array<double>^ x, Array<double>^ dy, Array<double>^ dw);
    };

    public ref class Convolution1D abstract sealed {
        public:
        static void Forward(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
                            PadMode padmode, Array<float>^ x, Array<float>^ w, Array<float>^ y);
        static void BackwardData(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
                                 PadMode padmode, Array<float>^ dy, Array<float>^ w, Array<float>^ dx);
        static void BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 kw,
                                   PadMode padmode, Array<float>^ x, Array<float>^ dy, Array<float>^ dw);
    };

    public ref class Convolution2D abstract sealed {
        public:
        static void Forward(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 kw, UInt32 kh,
                            PadMode padmode, Array<float>^ x, Array<float>^ w, Array<float>^ y);
        static void BackwardData(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 kw, UInt32 kh,
                                 PadMode padmode, Array<float>^ dy, Array<float>^ w, Array<float>^ dx);
        static void BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 kw, UInt32 kh,
                                   PadMode padmode, Array<float>^ x, Array<float>^ dy, Array<float>^ dw);
    };

    public ref class Convolution3D abstract sealed {
        public:
        static void Forward(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 id, UInt32 kw, UInt32 kh, UInt32 kd,
                            PadMode padmode, Array<float>^ x, Array<float>^ w, Array<float>^ y);
        static void BackwardData(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 id, UInt32 kw, UInt32 kh, UInt32 kd,
                                 PadMode padmode, Array<float>^ dy, Array<float>^ w, Array<float>^ dx);
        static void BackwardFilter(UInt32 n, UInt32 ic, UInt32 oc, UInt32 iw, UInt32 ih, UInt32 id, UInt32 kw, UInt32 kh, UInt32 kd,
                                   PadMode padmode, Array<float>^ x, Array<float>^ dy, Array<float>^ dw);
    };

    public ref class Pool1D abstract sealed {
        public:
        static void MaxPooling(UInt32 n, UInt32 c, UInt32 iw, 
                               UInt32 sx, UInt32 kw,
                               Array<float>^ x, Array<float>^ y);
        static void MaxUnpooling(UInt32 n, UInt32 c, UInt32 iw,
                               UInt32 sx, UInt32 kw,
                               Array<float>^ x, Array<float>^ y, Array<float>^ dy, Array<float>^ dx);
        static void AveragePooling(UInt32 n, UInt32 c, UInt32 iw,
                               UInt32 sx, UInt32 kw,
                               Array<float>^ x, Array<float>^ y);
        static void AverageUnpooling(UInt32 n, UInt32 c, UInt32 iw,
                               UInt32 sx, UInt32 kw,
                               Array<float>^ dy, Array<float>^ dx);
    };

    public ref class Pool2D abstract sealed {
        public:
        static void MaxPooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                               UInt32 sx, UInt32 sy, UInt32 kw, UInt32 kh,
                               Array<float>^ x, Array<float>^ y);
        static void MaxUnpooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                               UInt32 sx, UInt32 sy, UInt32 kw, UInt32 kh,
                               Array<float>^ x, Array<float>^ y, Array<float>^ dy, Array<float>^ dx);
        static void AveragePooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                               UInt32 sx, UInt32 sy, UInt32 kw, UInt32 kh,
                               Array<float>^ x, Array<float>^ y);
        static void AverageUnpooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                               UInt32 sx, UInt32 sy, UInt32 kw, UInt32 kh,
                               Array<float>^ dy, Array<float>^ dx);
    };

    public ref class Pool3D abstract sealed {
        public:
        static void MaxPooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                               UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
                               Array<float>^ x, Array<float>^ y);
        static void MaxUnpooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                               UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
                               Array<float>^ x, Array<float>^ y, Array<float>^ dy, Array<float>^ dx);
        static void AveragePooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                               UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
                               Array<float>^ x, Array<float>^ y);
        static void AverageUnpooling(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                               UInt32 sx, UInt32 sy, UInt32 sz, UInt32 kw, UInt32 kh, UInt32 kd,
                               Array<float>^ dy, Array<float>^ dx);
    };

    public ref class Upsample1D abstract sealed {
        public:
        static void NeighborX2(UInt32 n, UInt32 c, UInt32 iw,
                             Array<float>^ x, Array<float>^ y);
        static void LinearX2(UInt32 n, UInt32 c, UInt32 iw,
                             Array<float>^ x, Array<float>^ y);
    };

    public ref class Upsample2D abstract sealed {
        public:
        static void NeighborX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                             Array<float>^ x, Array<float>^ y);
        static void LinearX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                             Array<float>^ x, Array<float>^ y);
    };

    public ref class Upsample3D abstract sealed {
        public:
        static void NeighborX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                              Array<float>^ x, Array<float>^ y);
        static void LinearX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                             Array<float>^ x, Array<float>^ y);
    };

    public ref class Downsample1D abstract sealed {
        public:
        static void InterareaX2(UInt32 n, UInt32 c, UInt32 iw,
                                Array<float>^ x, Array<float>^ y);
    };

    public ref class Downsample2D abstract sealed {
        public:
        static void InterareaX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
                                Array<float>^ x, Array<float>^ y);
    };

    public ref class Downsample3D abstract sealed {
        public:
        static void InterareaX2(UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
                                Array<float>^ x, Array<float>^ y);
    };

    public ref class Initialize abstract sealed {
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

    public ref class Util abstract sealed {
        public:        
        generic <typename T> where T : ValueType
        static void CheckLength(UInt32 length, ... cli::array<Array<T>^>^ arrays);

        generic <typename T> where T : ValueType
        static void CheckOutOfRange(UInt32 index, UInt32 length, ... cli::array<Array<T>^>^ arrays);

        generic <typename T> where T : ValueType
        static void CheckDuplicateArray(... cli::array<Array<T>^>^ arrays);

        static void CheckProdOverflow(... cli::array<UInt32>^ arrays);

        static void AssertReturnCode(int ret);

        static property bool IsSupportedAVX { bool get(); };
        static property bool IsSupportedAVX2 { bool get(); };
        static property bool IsSupportedAVX512F { bool get(); };
        static property bool IsSupportedFMA { bool get(); };
    };

    private ref class Numeric abstract sealed {
        public:
        static UInt32 LCM(UInt32 a, UInt32 b);
        static UInt64 LCM(UInt64 a, UInt64 b);
        static UInt32 GCD(UInt32 a, UInt32 b);
        static UInt64 GCD(UInt64 a, UInt64 b);
    };

    private ref class ErrorMessage abstract sealed {
        public:
        static initonly System::String^ AvxNotSupported =
            "AVX2 not supported on this platform.";
        static initonly System::String^ InvalidArrayLength =
            "The specified array length is invalid.";
        static initonly System::String^ DuplicatedArray =
            "The specified arrays are duplicated.";
        static initonly System::String^ UndefinedEnum =
            "The specified enum is undefined.";
        static initonly System::String^ InvalidBatches =
            "The specified batches is invalid.";
        static initonly System::String^ InvalidChannels =
            "The specified channels is invalid.";
        static initonly System::String^ InvalidKernelSize =
            "The specified kernel size is invalid.";
        static initonly System::String^ InvalidPoolStride =
            "The specified pool stride is invalid.";
        static initonly System::String^ InvalidDataSize =
            "The specified data size is invalid.";
        static initonly System::String^ FailedWorkspaceAllocate =
            "Failed to allocate workspace memory.";
        static initonly System::String^ InvalidNativeFuncArgument =
            "The argument of the native function is invalid.";
        static initonly System::String^ MismatchSizeofElement =
            "Element size does not match array stride for the specified array type.";
    };
}