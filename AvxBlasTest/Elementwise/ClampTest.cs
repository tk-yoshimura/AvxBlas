using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class ClampTest {
        [TestMethod]
        public void SClampTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x1 = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] x2 = (new float[length]).Select((_, idx) => (float)-random.Next(32)).ToArray();
                    float[] x3 = (new float[length]).Select((_, idx) => (float)random.Next(32)).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Min(Math.Max(x1[idx], x2[idx]), x3[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Clamp(count, x1, x2, x3, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DClampTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x1 = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    double[] x2 = (new double[length]).Select((_, idx) => (double)-random.Next(32)).ToArray();
                    double[] x3 = (new double[length]).Select((_, idx) => (double)random.Next(32)).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Min(Math.Max(x1[idx], x2[idx]), x3[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Clamp(count, x1, x2, x3, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }
    }
}
