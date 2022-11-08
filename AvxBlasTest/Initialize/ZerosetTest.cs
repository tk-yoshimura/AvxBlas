using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace AvxBlasTest.InitializeTest {
    [TestClass]
    public class ZerosetTest {
        [TestMethod]
        public void SZerosetTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length]).Select((_, idx) => (float)idx + 1).ToArray();
                        float[] v2 = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0f : (float)idx + 1)
                            .ToArray();

                        Array<float> arr = new(v);

                        Initialize.Zeroset(index, count, arr);

                        float[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] v = (new float[length]).Select((_, idx) => (float)idx + 1).ToArray();
                    float[] v2 = (new float[length])
                        .Select((_, idx) => idx < count ? 0f : (float)idx + 1)
                        .ToArray();

                    Array<float> arr = new(v);

                    Initialize.Zeroset(count, arr);

                    float[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }

        [TestMethod]
        public void DZerosetTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length]).Select((_, idx) => (double)idx + 1).ToArray();
                        double[] v2 = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (double)idx + 1)
                            .ToArray();

                        Array<double> arr = new(v);

                        Initialize.Zeroset(index, count, arr);

                        double[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] v = (new double[length]).Select((_, idx) => (double)idx + 1).ToArray();
                    double[] v2 = (new double[length])
                        .Select((_, idx) => idx < count ? 0d : (double)idx + 1)
                        .ToArray();

                    Array<double> arr = new(v);

                    Initialize.Zeroset(count, arr);

                    double[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }

        [TestMethod]
        public void IZerosetTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        int[] v = (new int[length]).Select((_, idx) => idx + 1).ToArray();
                        int[] v2 = (new int[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : idx + 1)
                            .ToArray();

                        Array<int> arr = new(v);

                        Initialize.Zeroset(index, count, arr);

                        int[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    int[] v = (new int[length]).Select((_, idx) => idx + 1).ToArray();
                    int[] v2 = (new int[length])
                        .Select((_, idx) => idx < count ? 0 : idx + 1)
                        .ToArray();

                    Array<int> arr = new(v);

                    Initialize.Zeroset(count, arr);

                    int[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }

        [TestMethod]
        public void LZerosetTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        long[] v = (new long[length]).Select((_, idx) => (long)idx + 1).ToArray();
                        long[] v2 = (new long[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 0 : (long)idx + 1)
                            .ToArray();

                        Array<long> arr = new(v);

                        Initialize.Zeroset(index, count, arr);

                        long[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    long[] v = (new long[length]).Select((_, idx) => (long)idx + 1).ToArray();
                    long[] v2 = (new long[length])
                        .Select((_, idx) => idx < count ? 0 : (long)idx + 1)
                        .ToArray();

                    Array<long> arr = new(v);

                    Initialize.Zeroset(count, arr);

                    long[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }
    }
}
