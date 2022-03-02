#include "../AvxBlas.h"

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
UInt64 AvxBlas::Array<T>::Length::get() {
    return length;
}

generic <typename T>
UInt64 AvxBlas::Array<T>::MaxLength::get() {
    return MaxByteSize / ElementSize;
}

generic <typename T>
UInt64 AvxBlas::Array<T>::ByteSize::get() {
    return ElementSize * Length;
}

generic <typename T>
UInt64 AvxBlas::Array<T>::MaxByteSize::get() {
    return 0x40000000ull;
}

generic <typename T>
UInt64 AvxBlas::Array<T>::ElementSize::get() {
    return (UInt64)Marshal::SizeOf(T::typeid);
}

generic <typename T>
UInt64 AvxBlas::Array<T>::Alignment::get() {
    return 32;
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
AvxBlas::Array<T>::Array(UInt64 length) {
    if (length > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    size_t size = ((length * ElementSize + Alignment - 1) / Alignment) * Alignment;

    void* ptr = _aligned_malloc(size, Alignment);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    Zeroset();
}

generic <typename T>
AvxBlas::Array<T>::Array(Int64 length)
    : Array(length >= 0 ? (UInt64)length : throw gcnew System::ArgumentOutOfRangeException("length")){}

generic <typename T>
AvxBlas::Array<T>::Array(UInt32 length) : Array((UInt64)length) {}

generic <typename T>
AvxBlas::Array<T>::Array(Int32 length)
    : Array(length >= 0 ? (UInt64)length : throw gcnew System::ArgumentOutOfRangeException("length")) {}

generic <typename T>
AvxBlas::Array<T>::Array(cli::array<T>^ array) {
    UInt64 length = (UInt64)array->LongLength;

    if (length > MaxLength) {
        throw gcnew System::ArgumentOutOfRangeException("length");
    }

    size_t size = ((length * ElementSize + Alignment - 1) / Alignment) * Alignment;

    void* ptr = _aligned_malloc(size, Alignment);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);

    Write(array);
}

generic <typename T>
T AvxBlas::Array<T>::default::get(UInt64 index){
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

    cli::array<T> ^buffer = gcnew cli::array<T>(1);

    pin_ptr<T> buffer_ptr = &(buffer[0]);

    memcpy_s((void*)buffer_ptr, ElementSize, ptr, ElementSize);

    return buffer[0];
}

generic <typename T>
void AvxBlas::Array<T>::default::set(UInt64 index, T value) {
    if (index >= length) {
        throw gcnew System::IndexOutOfRangeException();
    }

    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

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
    if (array->length > (UInt64)int::MaxValue) {
        throw gcnew System::ArgumentOutOfRangeException("length");
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
void AvxBlas::Array<T>::Write(cli::array<T>^ array, UInt64 count) {
    if (count > length || count > (UInt64)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s(ptr, count * ElementSize, (void*)array_ptr, count * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Read(cli::array<T>^ array, UInt64 count) {
    if (count > length || count > (UInt64)array->LongLength) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    if (count <= 0) {
        return;
    }

    void* ptr = Ptr.ToPointer();

    pin_ptr<T> array_ptr = &(array[0]);

    memcpy_s((void*)array_ptr, count * ElementSize, ptr, count * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset() {
    Zeroset(length);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset(UInt64 count) {
    Zeroset(0, count);
}

generic <typename T>
void AvxBlas::Array<T>::Zeroset(UInt64 index, UInt64 count) {
    void* ptr = (void*)((UInt64)Ptr.ToInt64() + ElementSize * index);

    memset(ptr, 0, count * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::CopyTo(Array^ array, UInt64 count) {
    Copy(this, array, count);
}

generic <typename T>
void AvxBlas::Array<T>::CopyTo(UInt64 index, Array^ dst_array, UInt64 dst_index, UInt64 count) {
    Copy(this, index, dst_array, dst_index, count);
}

generic <typename T>
void AvxBlas::Array<T>::Copy(Array^ src_array, Array^ dst_array, UInt64 count) {
    if (count > src_array->Length || count > dst_array->Length) {
        throw gcnew System::ArgumentOutOfRangeException("count");
    }

    void* src_ptr = src_array->Ptr.ToPointer();
    void* dst_ptr = dst_array->Ptr.ToPointer();

    memcpy_s(dst_ptr, count * ElementSize, src_ptr, count * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::Copy(Array^ src_array, UInt64 src_index, Array^ dst_array, UInt64 dst_index, UInt64 count) {
    if (src_index >= src_array->Length || src_index + count > src_array->Length) {
        throw gcnew ArgumentOutOfRangeException("src_index");
    }
    if (dst_index >= dst_array->Length || dst_index + count > dst_array->Length) {
        throw gcnew ArgumentOutOfRangeException("dst_index");
    }

    void* src_ptr = (void*)((UInt64)src_array->Ptr.ToInt64() + ElementSize * src_index);
    void* dst_ptr = (void*)((UInt64)dst_array->Ptr.ToInt64() + ElementSize * dst_index);

    memcpy_s(dst_ptr, count * ElementSize, src_ptr, count * ElementSize);
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