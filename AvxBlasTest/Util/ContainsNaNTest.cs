using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace AvxBlasTest.UtilTest {
    [TestClass]
    public class ContainsNaNTest {
        [TestMethod]
        public void SContainsNaNTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? float.NaN : (float)idx + 1)
                            .ToArray();

                        Array<float> arr = new(v);

                        bool isnan = Util.ContainsNaN(index, count, arr);

                        Assert.AreEqual(count != 0, isnan);
                    }
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] v = (new float[length])
                        .Select((_, idx) => idx < count ? float.NaN : (float)idx + 1)
                        .ToArray();

                    Array<float> arr = new(v);

                    bool isnan = Util.ContainsNaN(count, arr);

                    Assert.AreEqual(count != 0, isnan);
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length; count++) {
                        float[] v = (new float[length])
                            .Select((_, idx) => idx == index ? float.NaN : (float)idx + 1)
                            .ToArray();

                        Array<float> arr = new(v);

                        bool isnan = Util.ContainsNaN(count, arr);

                        Assert.AreEqual(index < count, isnan);
                    }
                }
            }
        }

        [TestMethod]
        public void DContainsNaNTest() {
            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? double.NaN : (double)idx + 1)
                            .ToArray();

                        Array<double> arr = new(v);

                        bool isnan = Util.ContainsNaN(index, count, arr);

                        Assert.AreEqual(count != 0, isnan);
                    }
                }
            }

            for (uint length = 1; length <= 16; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] v = (new double[length])
                        .Select((_, idx) => idx < count ? double.NaN : (double)idx + 1)
                        .ToArray();

                    Array<double> arr = new(v);

                    bool isnan = Util.ContainsNaN(count, arr);

                    Assert.AreEqual(count != 0, isnan);
                }
            }

            for (uint length = 1; length <= 64; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length; count++) {
                        double[] v = (new double[length])
                            .Select((_, idx) => idx == index ? double.NaN : (double)idx + 1)
                            .ToArray();

                        Array<double> arr = new(v);

                        bool isnan = Util.ContainsNaN(count, arr);

                        Assert.AreEqual(index < count, isnan);
                    }
                }
            }
        }
    }
}
