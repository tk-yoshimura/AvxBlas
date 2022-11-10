using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class Arctan2Test {
        [TestMethod]
        public void SArctan2Test() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x1 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x2 = (new float[length]).Select((_, idx) => (float)random.Next(32) + 1).ToArray();

                    float[] ts = (new float[length])
                        .Select((_, idx) => idx < count ? (float)Math.Atan2(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Arctan2(count, x1, x2, y);

                    float[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-6);
                    }
                }
            }
        }

        [TestMethod]
        public void DArctan2Test() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x1 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x2 = (new double[length]).Select((_, idx) => (double)random.Next(32) + 1).ToArray();

                    double[] ts = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Atan2(x1[idx], x2[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Arctan2(count, x1, x2, y);

                    double[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-12);
                    }
                }
            }
        }
    }
}
