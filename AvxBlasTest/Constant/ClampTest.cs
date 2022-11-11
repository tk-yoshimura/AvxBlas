using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ConstantTest {
    [TestClass]
    public class ClampTest {
        [TestMethod]
        public void SClampTest() {
            Random random = new(1234);

            const float cmin = -5f, cmax = 4f;

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Min(Math.Max(x[idx], cmin), cmax) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Constant.Clamp(count, x, cmin, cmax, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DClampTest() {
            Random random = new(1234);

            const double cmin = -5d, cmax = 4d;

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Min(Math.Max(x[idx], cmin), cmax) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Constant.Clamp(count, x, cmin, cmax, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }
    }
}
