using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ConstantTest {
    [TestClass]
    public class RDivTest {
        [TestMethod]
        public void SRDivTest() {
            Random random = new(1234);

            const float c = 5f;

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? c / x[idx] : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Constant.RDiv(count, x, c, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DRDivTest() {
            Random random = new(1234);

            const double c = 5d;

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? c / x[idx] : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Constant.RDiv(count, x, c, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }
    }
}
