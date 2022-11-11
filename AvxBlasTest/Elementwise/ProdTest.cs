using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void SSumTest() {
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
                            .Select((_, idx) => idx < count ? (float)xsn.Select((x) => x[idx]).Sum() : 0)
                            .ToArray();

                        Array<float>[] xsr = xsn.Select((x) => (Array<float>)x).ToArray();
                        Array<float> y = new(length);

                        Elementwise.Sum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (float[])y);
                    }
                }
            }
        }

        [TestMethod]
        public void DSumTest() {
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
                            .Select((_, idx) => idx < count ? (double)xsn.Select((x) => x[idx]).Sum() : 0)
                            .ToArray();

                        Array<double>[] xsr = xsn.Select((x) => (Array<double>)x).ToArray();
                        Array<double> y = new(length);

                        Elementwise.Sum(count, xsr, y);

                        CollectionAssert.AreEqual(t, (double[])y);
                    }
                }
            }
        }
    }
}
