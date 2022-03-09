//#include "../avxblas.h"
//#include "../constants.h"
//#include "../utils.h"
//#include "../Inline/inline_set.cpp"
//#include "../Inline/inline_sum.cpp"
//#include "../Inline/inline_dotmul_s.cpp"
//#include <memory.h>
//
//using namespace System;
//
//#pragma unmanaged
//
//int affine_stride1_dotmul_s(
//    const unsigned int na, const unsigned int nb,
//    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {
//        
//    const unsigned int nbb = nb & AVX2_FLOAT_BATCH_MASK, nbr = nb - nbb;
//    const __m256i mask = _mm256_set_mask(nbr);
//
//    if (nb <= 4) {
//        for (unsigned int i = 0; i < na; i++) {
//            for (unsigned int j = 0; j < nb; j++) {
//                float y = a_ptr[i] * b_ptr[j];
//
//                *y_ptr = y;
//                y_ptr++;
//            }
//        }
//    }
//    else if (nbr > 0) {
//        for (unsigned int i = 0; i < na; i++) {
//            __m256 a = _mm256_set1_ps(a_ptr[i]);
//
//            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
//                __m256 b = _mm256_loadu_ps(b_ptr + j);
//
//                __m256 y = _mm256_mul_ps(a, b);
//
//                _mm256_storeu_ps(y_ptr, y);
//
//                y_ptr += AVX2_FLOAT_STRIDE;
//            }
//            {
//                __m256 b = _mm256_maskload_ps(b_ptr + nbb, mask);
//
//                __m256 y = _mm256_mul_ps(a, b);
//
//                _mm256_maskstore_ps(y_ptr, mask, y);
//
//                y_ptr += nbr;
//            }
//        }
//    }
//    else {
//        for (unsigned int i = 0; i < na; i++) {
//            __m256 a = _mm256_set1_ps(a_ptr[i]);
//
//            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
//                __m256 b = _mm256_load_ps(b_ptr + j);
//
//                __m256 y = _mm256_mul_ps(a, b);
//
//                _mm256_store_ps(y_ptr, y);
//
//                y_ptr += AVX2_FLOAT_STRIDE;
//            }
//        }
//    }
//
//    return SUCCESS;
//}
//
//int affine_stride2_dotmul_s(
//    const unsigned int na, const unsigned int nb,
//    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {
//
//    const unsigned int nbb = nb / 4 * 4, nbr = nb - nbb;
//    const __m256i mask = _mm256_set_mask(nbr), mask4 = _mm256_set_mask(4);
//
//    if (nb <= 2) {
//        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
//            for (unsigned int j = 0, nbs = nb * 2; j < nbs; j += 2) {
//                float y = a_ptr[i] * b_ptr[j] + a_ptr[i + 1] * b_ptr[j + 1];
//
//                *y_ptr = y;
//                y_ptr++;
//            }
//        }
//    }
//    else if (nbr > 0) {
//        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
//            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);
//
//            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
//                __m256 b = _mm256_loadu_ps(b_ptr + j);
//
//                __m256 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));
//
//                _mm256_maskstore_ps(y_ptr, mask4, y);
//
//                y_ptr += AVX2_FLOAT_STRIDE / 2;
//            }
//            {
//                __m256 b = _mm256_maskload_ps(b_ptr + nbb, mask);
//
//                __m256 y = _mm256_mul_ps(a, b);
//
//                _mm256_maskstore_ps(y_ptr, mask, y);
//
//                y_ptr += nbr;
//            }
//        }
//    }
//    else {
//        for (unsigned int i = 0, nas = na * 2; i < nas; i += 2) {
//            __m256 a = _mm256_set2_ps(a_ptr[i], a_ptr[i + 1]);
//
//            for (unsigned int j = 0; j < nbb; j += AVX2_FLOAT_STRIDE) {
//                __m256 b = _mm256_load_ps(b_ptr + j);
//
//                __m256 y = _mm256_hadd2_ps(_mm256_mul_ps(a, b));
//
//                _mm256_maskstore_ps(y_ptr, mask4, y);
//
//                y_ptr += AVX2_FLOAT_STRIDE / 2;
//            }
//        }
//    }
//
//    return SUCCESS;
//}
//
//int affine_stride3_dotmul_s(
//    const unsigned int na, const unsigned int nb,
//    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {
//
//    const __m128i mask = _mm_set_mask(3);
//
//    for (unsigned int i = 0, nas = na * 3; i < nas; i += 3) {
//        __m128 a = _mm_maskload_ps(a_ptr + i, mask);
//
//        for (unsigned int j = 0, nbs = nb * 3; j < nbs; j += 3) {
//            __m128 b = _mm_maskload_ps(b_ptr + j, mask);
//
//            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));
//
//            *y_ptr = y;
//            y_ptr++;
//        }
//    }
//
//    return SUCCESS;
//}
//
//int affine_stride4_dotmul_s(
//    const unsigned int na, const unsigned int nb,
//    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {
//
//    for (unsigned int i = 0, nas = na * 4; i < nas; i += 4) {
//        __m128 a = _mm_load_ps(a_ptr + i);
//
//        for (unsigned int j = 0, nbs = nb * 4; j < nbs; j += 4) {
//            __m128 b = _mm_load_ps(b_ptr + j);
//
//            float y = _mm_sum4to1_ps(_mm_mul_ps(a, b));
//
//            *y_ptr = y;
//            y_ptr++;
//        }
//    }
//
//    return SUCCESS;
//}
//
//int affine_stride5to7_dotmul_s(
//    const unsigned int na, const unsigned int nb, const unsigned int stride,
//    const float* __restrict a_ptr, const float* __restrict b_ptr, float* __restrict y_ptr) {
//
//#ifdef _DEBUG
//    if (stride <= AVX2_FLOAT_STRIDE / 2 || stride >= AVX2_FLOAT_STRIDE) {
//        throw std::exception();
//    }
//#endif // _DEBUG
//
//    const __m256i mask = _mm256_set_mask(stride);
//
//    for (unsigned int i = 0, nas = na * stride; i < nas; i += stride) {
//        __m256 a = _mm256_maskload_ps(a_ptr + i, mask);
//
//        for (unsigned int j = 0, nbs = nb * stride; j < nbs; j += stride) {
//            __m256 b = _mm256_maskload_ps(b_ptr + j, mask);
//
//            float y = _mm256_sum8to1_ps(_mm256_mul_ps(a, b));
//
//            *y_ptr = y;
//            y_ptr++;
//        }
//    }
//
//    return SUCCESS;
//}
//
//#pragma managed
//
//void AvxBlas::Affine::Dotmul(UInt32 na, UInt32 nb, UInt32 stride, Array<float>^ a, Array<float>^ b, Array<float>^ y) {
//    if (na <= 0 || nb <= 0 || stride <= 0) {
//        return;
//    }
//
//    Util::CheckProdOverflow(na, stride);
//    Util::CheckProdOverflow(nb, stride);
//    Util::CheckProdOverflow(na, nb);
//
//    Util::CheckLength(na * stride, a);
//    Util::CheckLength(nb * stride, b);
//
//    float* x_ptr = (float*)(x->Ptr.ToPointer());
//    float* y_ptr = (float*)(y->Ptr.ToPointer());
//
//    if (stride == 1u) {
//#ifdef _DEBUG
//        Console::WriteLine("type stride1");
//#endif // _DEBUG
//
//        ag_stride1_sum_s(n, samples, x_ptr, y_ptr);
//        return;
//    }
//
//    if (stride == 2u) {
//#ifdef _DEBUG
//        Console::WriteLine("type stride2");
//#endif // _DEBUG
//
//        ag_stride2_sum_s(n, samples, x_ptr, y_ptr);
//        return;
//    }
//
//    if (stride == 4u) {
//#ifdef _DEBUG
//        Console::WriteLine("type stride4");
//#endif // _DEBUG
//
//        ag_stride4_sum_s(n, samples, x_ptr, y_ptr);
//        return;
//    }
//
//    if ((stride & AVX2_FLOAT_REMAIN_MASK) == 0u) {
//#ifdef _DEBUG
//        Console::WriteLine("type alignment");
//#endif // _DEBUG
//
//        if (ag_alignment_sum_s(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
//            throw gcnew System::OutOfMemoryException();
//        }
//        return;
//    }
//
//    if (stride <= MAX_AGGREGATE_BATCHING) {
//        UInt32 g = Util::LCM(stride, AVX2_FLOAT_STRIDE) / stride;
//
//        if (samples >= g * 4) {
//#ifdef _DEBUG
//            Console::WriteLine("type batch g:" + g.ToString());
//#endif // _DEBUG
//
//            if (ag_batch_sum_s(n, g, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
//                throw gcnew System::OutOfMemoryException();
//            }
//            return;
//        }
//    }
//
//#ifdef _DEBUG
//    Console::WriteLine("type disorder");
//#endif // _DEBUG
//
//    if (ag_disorder_sum_s(n, samples, stride, x_ptr, y_ptr) == FAILURE_BADALLOC) {
//        throw gcnew System::OutOfMemoryException();
//    }
//    return;
//}
