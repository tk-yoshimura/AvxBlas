using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class MaximumTest {
        [TestMethod]
        public void SMaximumTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x1 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x2 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Max(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Maximum(count, x1, x2, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DMaximumTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x1 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x2 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Max(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Maximum(count, x1, x2, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }

        [TestMethod]
        public void SMaximumArrayTest() {
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
                            .Select((_, idx) => idx < count ? (n > 0) ? (float)xsn.Select((x) => x[idx]).Max() : float.NaN : 0)
                            .ToArray();

                        Array<float>[] xsr = xsn.Select((x) => (Array<float>)x).ToArray();
                        Array<float> y = new(length);

                        Elementwise.Maximum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (float[])y);
                    }
                }
            }
        }

        [TestMethod]
        public void DMaximumArrayTest() {
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
                            .Select((_, idx) => idx < count ? (n > 0) ? (double)xsn.Select((x) => x[idx]).Max() : double.NaN : 0)
                            .ToArray();

                        Array<double>[] xsr = xsn.Select((x) => (Array<double>)x).ToArray();
                        Array<double> y = new(length);

                        Elementwise.Maximum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (double[])y);
                    }
                }
            }
        }
    }
}
