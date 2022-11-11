using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class MinimumTest {
        [TestMethod]
        public void SMinimumTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x1 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x2 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Min(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Minimum(count, x1, x2, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DMinimumTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x1 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x2 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Min(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Minimum(count, x1, x2, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }

        [TestMethod]
        public void SMinimumArrayTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[][] xs = new float[16][];

                    for (int i = 0; i < xs.Length; i++) {
                        xs[i] = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    }

                    for (int n = 0; n < xs.Length; n++) {
                        var xsn = xs.Take(n).ToArray();

                        float[] t = (new float[length])
                            .Select((_, idx) => idx < count ? (n > 0) ? (float)xsn.Select((x) => x[idx]).Min() : float.NaN : 0)
                            .ToArray();

                        Array<float>[] xsr = xsn.Select((x) => (Array<float>)x).ToArray();
                        Array<float> y = new(length);

                        Elementwise.Minimum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (float[])y);
                    }
                }
            }
        }

        [TestMethod]
        public void DMinimumArrayTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[][] xs = new double[16][];

                    for (int i = 0; i < xs.Length; i++) {
                        xs[i] = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    }

                    for (int n = 0; n < xs.Length; n++) {
                        var xsn = xs.Take(n).ToArray();

                        double[] t = (new double[length])
                            .Select((_, idx) => idx < count ? (n > 0) ? (double)xsn.Select((x) => x[idx]).Min() : double.NaN : 0)
                            .ToArray();

                        Array<double>[] xsr = xsn.Select((x) => (Array<double>)x).ToArray();
                        Array<double> y = new(length);

                        Elementwise.Minimum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (double[])y);
                    }
                }
            }
        }
    }
}
