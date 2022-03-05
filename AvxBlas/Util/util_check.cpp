#include "../avxblas.h"
#include "../avxblasutil.h"
#include <array>
#include <vector>
#include <bitset>
#include <intrin.h>

using namespace System;

generic <typename T> where T : ValueType
void AvxBlas::Util::CheckLength(UInt32 length, ...cli::array<Array<T>^>^ arrays) {
    if (length <= 0) {
        return;
    }
    
    for each (Array<T>^ array in arrays){
        if (length > (unsigned int)array->Length) {
            throw gcnew System::IndexOutOfRangeException(AvxBlas::Util::InvalidArrayLength);
        }
    }
}

generic <typename T> where T : ValueType
void AvxBlas::Util::CheckOutOfRange(UInt32 index, UInt32 length, ...cli::array<Array<T>^>^ arrays) {
    if (length <= 0) {
        return;
    }

    if (index + length < index) {
        throw gcnew System::IndexOutOfRangeException(AvxBlas::Util::InvalidArrayLength);
    }

    for each (Array<T>^ array in arrays) {
        if (index >= (unsigned int)array->Length || index + length > (unsigned int)array->Length) {
            throw gcnew System::IndexOutOfRangeException(AvxBlas::Util::InvalidArrayLength);
        }
    }
}

generic <typename T> where T : ValueType
void AvxBlas::Util::CheckDuplicateArray(... cli::array<Array<T>^>^ arrays) {
    for (int i = 0; i < arrays->Length; i++) {
        for (int j = 0; j < i; j++) {
            if (ReferenceEquals(arrays[i], arrays[j])) {
                throw gcnew System::ArgumentException(AvxBlas::Util::DuplicatedArray);
            }
        }
    }
}

void AvxBlas::Util::CheckProdOverflow(... cli::array<UInt32>^ arrays) {
    if (arrays->Length <= 1) {
        return;
    }

    UInt32 a = arrays[0];

    for (int i = 1; i < arrays->Length; i++) {
        UInt32 b = arrays[i];

        if (b > 0 && a > (~0u) / b) {
            throw gcnew System::OverflowException();
        }

        a *= b;
    }
}

bool is_supported_avx() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_1_ECX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++){
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 1){
        f_1_ECX = data[1][2];
    }

    return f_1_ECX[28];
}

bool AvxBlas::Util::IsSupportedAVX::get() {
    return is_supported_avx();
}

bool is_supported_avx2() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_7_EBX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 7) {
        f_7_EBX = data[7][1];
    }

    return f_7_EBX[5];
}

bool AvxBlas::Util::IsSupportedAVX2::get() {
    return is_supported_avx2();
}

bool is_supported_avx512f() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_7_EBX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 7) {
        f_7_EBX = data[7][1];
    }

    return f_7_EBX[16];
}

bool AvxBlas::Util::IsSupportedAVX512F::get() {
    return is_supported_avx512f();
}

bool is_supported_fma() {
    std::array<int, 4> cpui;
    std::vector<std::array<int, 4>> data;
    std::bitset<32> f_1_ECX;

    __cpuid(cpui.data(), 0);
    int nids = cpui[0];

    for (int i = 0; i <= nids; i++) {
        __cpuidex(cpui.data(), i, 0);
        data.push_back(cpui);
    }

    if (nids >= 1) {
        f_1_ECX = data[1][2];
    }

    return f_1_ECX[12];
}

bool AvxBlas::Util::IsSupportedFMA::get() {
    return is_supported_fma();
}

System::String^ AvxBlas::Util::AvxNotSupported::get() {
    return "AVX2 not supported on this platform.";
}

System::String^ AvxBlas::Util::InvalidArrayLength::get() {
    return "The specified array length is invalid.";
}

System::String^ AvxBlas::Util::DuplicatedArray::get() {
    return "The specified arrays are duplicated.";
}