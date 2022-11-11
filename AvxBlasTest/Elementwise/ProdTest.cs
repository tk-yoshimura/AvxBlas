using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Collections.Generic;

namespace AvxBlasTest.ElementwiseTest {
    public static class Extensions {
        public static float Prod(this IEnumerable<float> xs) {
            float p = 1;
            foreach (float x in xs) {
                p *= x;
            }

            return p;
        }

        public static double Prod(this IEnumerable<double> xs) {
            double p = 1;
            foreach (double x in xs) {
                p *= x;
            }

            return p;
        }
    }

    [TestClass]
    public class ProdTest {

        [TestMethod]
        public void SProdTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[][] xs = new float[16][];

                    for (int i = 0; i < xs.Length; i++) {
                        xs[i] = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    }

                    for (int n = 0; n < xs.Length; n++) {
                        var xsn = xs.Take(n).ToArray();

                        float[] ts = (new float[length])
                            .Select((_, idx) => idx < count ? (float)xsn.Select((x) => x[idx]).Prod() : 0)
                            .ToArray();

                        Array<float>[] xsr = xsn.Select((x) => (Array<float>)x).ToArray();
                        Array<float> y = new(length);

                        Elementwise.Prod(count, xsr, y);

                        float[] ys = y;

                        for (int i = 0; i < ts.Length; i++) {
                            Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-6);
                        }
                    }
                }
            }
        }

        [TestMethod]
        public void DProdTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[][] xs = new double[16][];

                    for (int i = 0; i < xs.Length; i++) {
                        xs[i] = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    }

                    for (int n = 0; n < xs.Length; n++) {
                        var xsn = xs.Take(n).ToArray();

                        double[] ts = (new double[length])
                            .Select((_, idx) => idx < count ? (double)xsn.Select((x) => x[idx]).Prod() : 0)
                            .ToArray();

                        Array<double>[] xsr = xsn.Select((x) => (Array<double>)x).ToArray();
                        Array<double> y = new(length);

                        Elementwise.Prod(count, xsr, y);

                        double[] ys = y;

                        for (int i = 0; i < ts.Length; i++) {
                            Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-15);
                        }
                    }
                }
            }
        }
    }
}
