using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class LogTest {
        [TestMethod]
        public void SLogTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) + 1).ToArray();

                    float[] ts = (new float[length])
                        .Select((_, idx) => idx < count ? (float)Math.Log(x[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Log(count, x, y);

                    float[] ys = y;

                    for (int i = 0; i < ts.Length; i++) { 
                        Assert.AreEqual(ts[i], ys[i], 1e-7);
                    }
                }
            }
        }

        [TestMethod]
        public void DLogTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) + 1).ToArray();

                    double[] ts = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Log(x[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Log(count, x, y);

                    double[] ys = y;

                    for (int i = 0; i < ts.Length; i++) { 
                        Assert.AreEqual(ts[i], ys[i], 1e-15);
                    }
                }
            }
        }
    }
}
