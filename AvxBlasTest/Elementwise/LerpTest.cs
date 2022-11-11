using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class LerpTest {
        [TestMethod]
        public void SLerpTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x1 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x2 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x3 = (new float[length]).Select((_, idx) => (float)random.NextDouble()).ToArray();

                    float[] ts = (new float[length])
                        .Select((_, idx) => idx < count ? x1[idx] * x3[idx] + (x2[idx] * (1 - x3[idx])) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Lerp(count, x1, x2, x3, y);

                    float[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], 1e-5);
                    }
                }
            }
        }

        [TestMethod]
        public void DLerpTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x1 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x2 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x3 = (new double[length]).Select((_, idx) => (double)random.NextDouble()).ToArray();

                    double[] ts = (new double[length])
                        .Select((_, idx) => idx < count ? x1[idx] * x3[idx] + (x2[idx] * (1 - x3[idx])) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Lerp(count, x1, x2, x3, y);

                    double[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], 1e-14);
                    }
                }
            }
        }
    }
}
