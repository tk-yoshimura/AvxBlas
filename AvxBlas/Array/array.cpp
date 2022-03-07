#include "../avxblas.h"
#include "../constants.h"

#include <stdlib.h>
#include <memory.h>

using namespace System::Runtime::InteropServices;

generic <typename T>
static AvxBlas::Array<T>::Array() {
    cli::array<T>^ array = gcnew cli::array<T>(2);

    GCHandle pinned_handle = GCHandle::Alloc(array, GCHandleType::Pinned);

    try {
        __int64 stride = Marshal::UnsafeAddrOfPinnedArrayElement(array, 1).ToInt64()
                       - Marshal::UnsafeAddrOfPinnedArrayElement(array, 0).ToInt64();

        if ((unsigned __int64)stride != ElementSize) {
            throw gcnew NotSupportedException("Element size does not match array stride for the specified array type.");
        }
    }
    finally {
        pinned_handle.Free();
    }
}

generic <typename T>
IntPtr AvxBlas::Array<T>::Ptr::get() {
    if (ptr == IntPtr::Zero) {
        throw gcnew System::InvalidOperationException();
    }

    return ptr;
}

generic <typename T>
UInt32 AvxBlas::Array<T>::Length::get() {
    return length;
}

generic <typename T>
UInt32 AvxBlas::Array<T>::MaxLength::get() {
    return MaxByteSize / ElementSize;
}

generic <typename T>
UInt32 AvxBlas::Array<T>::ByteSize::get() {
    return ElementSize * Length;
}

generic <typename T>
UInt32 AvxBlas::Array<T>::MaxByteSize::get() {
    return 0x40000000ull;
}

generic <typename T>
UInt32 AvxBlas::Array<T>::ElementSize::get() {
    return (UInt32)Marshal::SizeOf(T::typeid);
}

generic <typename T>
UInt32 AvxBlas::Array<T>::Alignment::get() {
    return AVX2_ALIGNMENT;
}

generic <typename T>
bool AvxBlas::Array<T>::IsValid::get() {
    return ptr != IntPtr::Zero;
}

generic <typename T>
Type^ AvxBlas::Array<T>::ElementType::get() {
    return T::typeid;
}

generic <typename T>
cli::array<T>^ AvxBlas::Array<T>::Value::get(){
    return (cli::array<T>^)this;
}


generic <typename T>
String^ AvxBlas::Array<T>::Overview::get() {
    return ElementType->Name + "[" + Length + "]";
}

generic <typename T>
AvxBlas::Array<T>::Array(UInt32 length)
    : Array(length, true) {}

generic <typename T>
AvxBlas::Array<T>::Array(Int32 length)
    : Array(length, true) {}

generic <typename T>
AvxBlas::Array<T>::Array(UInt32 length, bool zeroset){
    if (length > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    if (length <= 0) {
        this->length = length;
        this->ptr = IntPtr::Zero;
        return;
    }

    size_t size = (static_cast<size_t>((length * ElementSize + AVX2_ALIGNMENT - 1) / AVX2_ALIGNMENT)) * AVX2_ALIGNMENT;

    void* ptr = _aligned_malloc(size, AVX2_ALIGNMENT);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    if (zeroset) {
        Zeroset();
    }
}

generic <typename T>
AvxBlas::Array<T>::Array(Int32 length, bool zeroset)
    : Array(length >= 0 ? (UInt32)length : throw gcnew System::ArgumentOutOfRangeException("length"), zeroset) {}

generic <typename T>
AvxBlas::Array<T>::Array(cli::array<T>^ array) {
    if (array->LongLength > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    UInt32 length = array->Length;

    if (length <= 0) {
        this->length = length;
        this->ptr = IntPtr::Zero;
        return;
    }

    size_t size = (static_cast<size_t>((length * ElementSize + AVX2_ALIGNMENT - 1) / AVX2_ALIGNMENT)) * AVX2_ALIGNMENT;

    void* ptr = _aligned_malloc(size, AVX2_ALIGNMENT);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    Write(array);
}

generic <typename T>
T AvxBlas::Array<T>::default::get(UInt32 index){
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)(Ptr.ToInt64() + ElementSize * static_cast<long long>(index));

    cli::array<T> ^buffer = gcnew cli::array<T>(1);

    pin_ptr<T> buffer_ptr = &(buffer[0]);

    memcpy_s((void*)buffer_ptr, ElementSize, ptr, ElementSize);

    return buffer[0];
}

generic <typename T>
void AvxBlas::Array<T>::default::set(UInt32 index, T value) {
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)(Ptr.ToInt64() + ElementSize * static_cast<long long>(index));

    cli::array<T>^ buffer = gcnew cli::array<T>(1) { value };

    pin_ptr<T> buffer_ptr = &(buffer[0]);

    memcpy_s(ptr, ElementSize, (void*)buffer_ptr, ElementSize);
}

generic <typename T>
AvxBlas::Array<T>::operator AvxBlas::Array<T>^ (cli::array<T>^ array) {
    return gcnew Array(array);
}

generic <typename T>
AvxBlas::Array<T>::operator cli::array<T>^ (Array^ array) {
    if (array->length <= 0) {
        return gcnew cli::array<T>(0);
    }

    cli::array<T>^ arr = gcnew cli::array<T>((int)array->length);
    array->Read(arr);

    return arr;
}

generic <typename T>
void AvxBlas::Array<T>::Write(cli::array<T>^ array){
    Write(array, length);
}

generic <typename T>
void AvxBlas::Array<T>::Read(cli::array<T>^ array) {
    Read(array, length);
}

generic <typename T>
void AvxBlas::Array<T>::Write(cli::array<T>^ array, UInt32 count) {
    if (count > length || count > (UInt32)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s(ptr, static_cast<rsize_t>(count) * ElementSize, (void*)array_ptr, static_cast<rsize_t>(count) * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Read(cli::array<T>^ array, UInt32 count) {
    if (count > length || count > (UInt32)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s((void*)array_ptr, static_cast<rsize_t>(count) * ElementSize, ptr, static_cast<rsize_t>(count) * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset() {
    Zeroset(length);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset(UInt32 count) {
    Zeroset(0, count);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset(UInt32 index, UInt32 count) {
    void* ptr = (void*)(Ptr.ToInt64() + ElementSize * static_cast<long long>(index));

    memset(ptr, 0, static_cast<size_t>(count) * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::CopyTo(Array^ array, UInt32 count) {
    Copy(this, array, count);
}

generic <typename T>
void AvxBlas::Array<T>::CopyTo(UInt32 index, Array^ dst_array, UInt32 dst_index, UInt32 count) {
    Copy(this, index, dst_array, dst_index, count);
}

generic <typename T>
void AvxBlas::Array<T>::Copy(Array^ src_array, Array^ dst_array, UInt32 count) {
    if (count > src_array->Length || count > dst_array->Length) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    void* src_ptr = src_array->Ptr.ToPointer();
    void* dst_ptr = dst_array->Ptr.ToPointer();

    memcpy_s(dst_ptr, static_cast<rsize_t>(count) * ElementSize, src_ptr, static_cast<rsize_t>(count) * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Copy(Array^ src_array, UInt32 src_index, Array^ dst_array, UInt32 dst_index, UInt32 count) {
    if (src_index >= src_array->Length || src_index + count > src_array->Length) {
        throw gcnew ArgumentOutOfRangeException("src_index");
    }
    if (dst_index >= dst_array->Length || dst_index + count > dst_array->Length) {
        throw gcnew ArgumentOutOfRangeException("dst_index");
    }

    void* src_ptr = (void*)(src_array->Ptr.ToInt64() + ElementSize * static_cast<long long>(src_index));
    void* dst_ptr = (void*)(dst_array->Ptr.ToInt64() + ElementSize * static_cast<long long>(dst_index));

    memcpy_s(dst_ptr, static_cast<rsize_t>(count) * ElementSize, src_ptr, static_cast<rsize_t>(count) * ElementSize);
}


generic <typename T>
String^ AvxBlas::Array<T>::ToString() {
    return "Array " + Overview;
}

generic <typename T>
AvxBlas::Array<T>::~Array() {
    this->!Array();
}

generic <typename T>
AvxBlas::Array<T>::!Array() {
    if (this->ptr != IntPtr::Zero) {
        void* ptr = (this->ptr).ToPointer();

        _aligned_free(ptr);

        this->length = 0;
        this->ptr = IntPtr::Zero;
    }
}