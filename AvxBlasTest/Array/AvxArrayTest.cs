using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace AvxBlasTest {
    [TestClass]
    public class ArrayTest {
        [TestMethod]
        public void CreateTest() {
            const int length = 15;

            Array<float> arr = new(length);

            Assert.IsTrue(arr.IsValid);
            Assert.AreEqual(Marshal.SizeOf(typeof(float)) * length, (int)arr.ByteSize);

            arr.Dispose();

            Assert.IsFalse(arr.IsValid);
            Assert.AreEqual(0, (int)arr.ByteSize);
        }

        [TestMethod]
        public void AlignmentTest() {
            for (int length = 1; length <= 100; length++) {

                Array<float> arr = new(length);

                Assert.IsTrue(arr.IsValid);
                Assert.AreEqual(Marshal.SizeOf(typeof(float)) * length, (int)arr.ByteSize);

                Assert.IsTrue((arr.Ptr.ToInt64() % (long)Array<float>.Alignment) == 0);

                arr.Dispose();

                Assert.IsFalse(arr.IsValid);
                Assert.AreEqual(0, (int)arr.ByteSize);

                Assert.ThrowsException<InvalidOperationException>(() => {
                    IntPtr ptr = arr.Ptr;
                });
            }
        }

        [TestMethod]
        public void InitializeTest() {
            const int length = 15;

            float[] v = new float[length];

            Array<float> arr = new(length);

            arr.Read(v);

            foreach (float f in v) {
                Assert.AreEqual(0.0f, f);
            }
        }

        [TestMethod]
        public void WriteReadTest() {
            for (uint length = 1; length < 100; length++) {

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];

                Array<float> arr = new(length);
                Array<float> arr2 = new(length);

                arr.Write(v);

                Array<float>.Copy(arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }

            for (uint length = 1; length < 100; length++) {

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];

                Array<double> arr = new(length);
                Array<double> arr2 = new(length);

                arr.Write(v);

                Array<double>.Copy(arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }

            for (uint length = 1; length < 100; length++) {

                short[] v = (new short[length]).Select((_, idx) => (short)idx).ToArray();
                short[] v2 = new short[length];

                Array<short> arr = new(length);
                Array<short> arr2 = new(length);

                arr.Write(v);

                Array<short>.Copy(arr, arr2, length);

                arr2.Read(v2);

                CollectionAssert.AreEqual(v, v2);
            }
        }

        [TestMethod]
        public void RegionWriteReadTest() {
            for (uint length = 1; length < 100; length++) {

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];
                float[] v3 = (new float[length]).Select((_, idx) => idx < length - 1 ? (float)idx : 0).ToArray();

                Array<float> arr = new(length);
                Array<float> arr2 = new(length);

                arr.Write(v, length - 1);

                Array<float>.Copy(arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }

            for (uint length = 1; length < 100; length++) {

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < length - 1 ? (double)idx : 0).ToArray();

                Array<double> arr = new(length);
                Array<double> arr2 = new(length);

                arr.Write(v, length - 1);

                Array<double>.Copy(arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }

            for (uint length = 1; length < 100; length++) {

                short[] v = (new short[length]).Select((_, idx) => (short)idx).ToArray();
                short[] v2 = new short[length];
                short[] v3 = (new short[length]).Select((_, idx) => idx < length - 1 ? (short)idx : (short)0).ToArray();

                Array<short> arr = new(length);
                Array<short> arr2 = new(length);

                arr.Write(v, length - 1);

                Array<short>.Copy(arr, arr2, length);

                arr2.Read(v2, length - 1);

                CollectionAssert.AreEqual(v2, v3);
            }
        }

        [TestMethod]
        public void ZerosetTest() {
            for (uint length = 1; length < 100; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                        float[] v2 = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (float)idx)
                            .ToArray();

                        Array<float> arr = new(v);

                        arr.Zeroset(index, count);

                        float[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v2 = new float[length];
                float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? 0 : (float)idx).ToArray();
                float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? 0 : (float)idx).ToArray();

                float[] v5 = new float[length];

                Array<float> arr = new(length);

                arr.Write(v, length);
                arr.Zeroset();
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v2, v5);

                arr.Write(v, length);
                arr.Zeroset(length - 1);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v3, v5);

                arr.Write(v, length);
                arr.Zeroset(3, length - 4);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v4, v5);
            }

            for (uint length = 1; length < 100; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                        double[] v2 = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (double)idx)
                            .ToArray();

                        Array<double> arr = new(v);

                        arr.Zeroset(index, count);

                        double[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v2 = new double[length];
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? 0 : (double)idx).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? 0 : (double)idx).ToArray();

                double[] v5 = new double[length];

                Array<double> arr = new(length);

                arr.Write(v, length);
                arr.Zeroset();
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v2, v5);

                arr.Write(v, length);
                arr.Zeroset(length - 1);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v3, v5);

                arr.Write(v, length);
                arr.Zeroset(3, length - 4);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v4, v5);
            }

            for (uint length = 1; length < 100; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        short[] v = (new short[length]).Select((_, idx) => (short)idx).ToArray();
                        short[] v2 = (new short[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? (short)0 : (short)idx)
                            .ToArray();

                        Array<short> arr = new(v);

                        arr.Zeroset(index, count);

                        short[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            {
                const int length = 15;

                short[] v = (new short[length]).Select((_, idx) => (short)idx).ToArray();
                short[] v2 = new short[length];
                short[] v3 = (new short[length]).Select((_, idx) => idx < length - 1 ? (short)0 : (short)idx).ToArray();
                short[] v4 = (new short[length]).Select((_, idx) => idx >= 3 && idx < length - 1 ? (short)0 : (short)idx).ToArray();

                short[] v5 = new short[length];

                Array<short> arr = new(length);

                arr.Write(v, length);
                arr.Zeroset();
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v2, v5);

                arr.Write(v, length);
                arr.Zeroset(length - 1);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v3, v5);

                arr.Write(v, length);
                arr.Zeroset(3, length - 4);
                arr.Read(v5, length);
                CollectionAssert.AreEqual(v4, v5);
            }
        }


        [TestMethod]
        public void CopyTest() {
            for (uint src_length = 1; src_length <= 16; src_length++) {
                for (uint dst_length = 1; dst_length <= 16; dst_length++) {
                    for (uint src_index = 0; src_index < src_length; src_index++) {
                        for (uint dst_index = 0; dst_index < dst_length; dst_index++) {
                            for (uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                float[] v = (new float[src_length]).Select((_, idx) => (float)idx).ToArray();
                                float[] v2 = (new float[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();

                                Array<float> arr_src = new(v);
                                Array<float> arr_dst = new(dst_length);

                                Array<float>.Copy(arr_src, src_index, arr_dst, dst_index, count);

                                float[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                float[] v = (new float[length]).Select((_, idx) => (float)idx).ToArray();
                float[] v3 = (new float[length]).Select((_, idx) => idx < (float)(length - 1) ? (float)idx : 0).ToArray();
                float[] v4 = (new float[length]).Select((_, idx) => idx >= 3 && idx < (float)(length - 1) ? (float)idx : 0).ToArray();
                float[] v5 = (new float[length]).Select((_, idx) => idx >= 2 && idx < (float)(length - 2) ? (float)idx + 1 : 0).ToArray();

                float[] v6 = new float[length];

                Array<float> arr = new(v);
                Array<float> arr2 = new(length);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v, v6);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length - 1);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v3, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 3, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v4, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 2, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v5, v6);
            }

            for (uint src_length = 1; src_length <= 16; src_length++) {
                for (uint dst_length = 1; dst_length <= 16; dst_length++) {
                    for (uint src_index = 0; src_index < src_length; src_index++) {
                        for (uint dst_index = 0; dst_index < dst_length; dst_index++) {
                            for (uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                double[] v = (new double[src_length]).Select((_, idx) => (double)idx).ToArray();
                                double[] v2 = (new double[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : 0)
                                    .ToArray();

                                Array<double> arr_src = new(v);
                                Array<double> arr_dst = new(dst_length);

                                Array<double>.Copy(arr_src, src_index, arr_dst, dst_index, count);

                                double[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                double[] v = (new double[length]).Select((_, idx) => (double)idx).ToArray();
                double[] v3 = (new double[length]).Select((_, idx) => idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v4 = (new double[length]).Select((_, idx) => idx >= 3 && idx < (double)(length - 1) ? (double)idx : 0).ToArray();
                double[] v5 = (new double[length]).Select((_, idx) => idx >= 2 && idx < (double)(length - 2) ? (double)idx + 1 : 0).ToArray();

                double[] v6 = new double[length];

                Array<double> arr = new(v);
                Array<double> arr2 = new(length);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v, v6);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length - 1);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v3, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 3, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v4, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 2, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v5, v6);
            }

            for (uint src_length = 1; src_length <= 16; src_length++) {
                for (uint dst_length = 1; dst_length <= 16; dst_length++) {
                    for (uint src_index = 0; src_index < src_length; src_index++) {
                        for (uint dst_index = 0; dst_index < dst_length; dst_index++) {
                            for (uint count = 0; count <= Math.Min(src_length - src_index, dst_length - dst_index); count++) {

                                short[] v = (new short[src_length]).Select((_, idx) => (short)idx).ToArray();
                                short[] v2 = (new short[dst_length])
                                    .Select((_, idx) => idx >= dst_index && idx < (dst_index + count) ? v[idx - dst_index + src_index] : (short)0)
                                    .ToArray();

                                Array<short> arr_src = new(v);
                                Array<short> arr_dst = new(dst_length);

                                Array<short>.Copy(arr_src, src_index, arr_dst, dst_index, count);

                                short[] v3 = arr_dst;

                                CollectionAssert.AreEqual(v2, v3);

                            }
                        }
                    }
                }
            }

            {
                const int length = 15;

                short[] v = (new short[length]).Select((_, idx) => (short)idx).ToArray();
                short[] v3 = (new short[length]).Select((_, idx) => idx < length - 1 ? (short)idx : (short)0).ToArray();
                short[] v4 = (new short[length]).Select((_, idx) => idx >= 3 && idx < length - 1 ? (short)idx : (short)0).ToArray();
                short[] v5 = (new short[length]).Select((_, idx) => idx >= 2 && idx < length - 2 ? (short)(idx + 1) : (short)0).ToArray();

                short[] v6 = new short[length];

                Array<short> arr = new(v);
                Array<short> arr2 = new(length);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v, v6);

                arr2.Zeroset();
                arr.CopyTo(arr2, arr.Length - 1);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v3, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 3, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v4, v6);

                arr2.Zeroset();
                arr.CopyTo(3, arr2, 2, arr.Length - 4);
                arr2.Read(v6);
                CollectionAssert.AreEqual(v5, v6);
            }
        }

        [TestMethod]
        public void BadCreateTest() {
            Assert.ThrowsException<TypeInitializationException>(() => {
                Array<char> arr = new(32);
            });

            Assert.ThrowsException<ArgumentOutOfRangeException>(() => {
                Array<float> arr = new(-1);
            });

            Array<byte> arr = new(32);
        }
    }
}
