using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class SquareTest {
        [TestMethod]
        public void SSquareTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] ts = (new float[length])
                        .Select((_, idx) => idx < count ? x[idx] * x[idx] : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Square(count, x, y);

                    float[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-6);
                    }
                }
            }
        }

        [TestMethod]
        public void DSquareTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] ts = (new double[length])
                        .Select((_, idx) => idx < count ? x[idx] * x[idx] : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Square(count, x, y);

                    double[] ys = y;

                    for (int i = 0; i < ts.Length; i++) {
                        Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-15);
                    }
                }
            }
        }
    }
}
