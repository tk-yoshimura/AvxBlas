#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

#include <stdlib.h>
#include <memory.h>

#pragma managed

using namespace System::Runtime::InteropServices;

generic <typename T>
static AvxBlas::Array<T>::Array() {
    if (!AvxBlas::Util::IsSupportedAVX || !AvxBlas::Util::IsSupportedAVX2) {
        throw gcnew System::PlatformNotSupportedException(AvxBlas::ErrorMessage::AvxNotSupported);
    }

    cli::array<T>^ array = gcnew cli::array<T>(2);

    GCHandle pinned_handle = GCHandle::Alloc(array, GCHandleType::Pinned);

    try {
        __int64 stride = Marshal::UnsafeAddrOfPinnedArrayElement(array, 1).ToInt64()
                       - Marshal::UnsafeAddrOfPinnedArrayElement(array, 0).ToInt64();

        if ((unsigned __int64)stride != ElementSize) {
            throw gcnew NotSupportedException(AvxBlas::ErrorMessage::MismatchSizeofElement);
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

    size_t size = (((size_t)length * ElementSize + AVX2_ALIGNMENT * 2 - 1) / AVX2_ALIGNMENT) * AVX2_ALIGNMENT;

    void* ptr = _aligned_malloc(size, AVX2_ALIGNMENT);
    if (ptr == nullptr) {
        throw gcnew System::OutOfMemoryException();
    }

    this->length = length;
    this->ptr = IntPtr(ptr);
    this->allocsize = size;

    if (zeroset) {
        Zeroset();
    }

#ifdef _DEBUG
    FillOutOfIndex();
#endif // _DEBUG

}

generic <typename T>
AvxBlas::Array<T>::Array(Int32 length, bool zeroset)
    : Array(length >= 0 ? (UInt32)length : throw gcnew System::ArgumentOutOfRangeException("length"), zeroset) {}

generic <typename T>
AvxBlas::Array<T>::Array(cli::array<T>^ array)
    : Array(
        array->LongLength <= MaxLength
        ? array->Length
        : throw gcnew System::ArgumentOutOfRangeException("length"), 
        false) {

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

#ifdef _DEBUG
    array->CheckOverflow();
#endif // _DEBUG

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

    if (T::typeid == float::typeid) {
        zeroset_s(count, (float*)ptr);
        return;
    }
    if (T::typeid == double::typeid) {
        zeroset_d(count, (double*)ptr);
        return;
    }

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

    if (T::typeid == float::typeid) {
        copy_s(count, (float*)src_ptr, (float*)dst_ptr);
        return;
    }
    if (T::typeid == double::typeid) {
        copy_d(count, (double*)src_ptr, (double*)dst_ptr);
        return;
    }

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

    if (T::typeid == float::typeid) {
        copy_s(count, (float*)src_ptr, (float*)dst_ptr);
        return;
    }
    if (T::typeid == double::typeid) {
        copy_d(count, (double*)src_ptr, (double*)dst_ptr);
        return;
    }

    memcpy_s(dst_ptr, static_cast<rsize_t>(count) * ElementSize, src_ptr, static_cast<rsize_t>(count) * ElementSize);
}

generic <typename T>
void AvxBlas::Array<T>::FillOutOfIndex() {
    unsigned char* ucptr = (unsigned char*)(this->ptr).ToPointer();
    
    for (UInt64 i = (UInt64)Length * ElementSize; i < allocsize; i++) {
        ucptr[i] = (i & 0x7Fu) | 0x80u;
    }
}

generic <typename T>
void AvxBlas::Array<T>::CheckOverflow() {
    unsigned char* ucptr = (unsigned char*)(this->ptr).ToPointer();

    for (UInt64 i = (UInt64)Length * ElementSize; i < allocsize; i++) {
        if (ucptr[i] != ((i & 0x7Fu) | 0x80u)) {
            throw gcnew System::AccessViolationException();
        }
    }
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
        this->allocsize = 0;
    }
}