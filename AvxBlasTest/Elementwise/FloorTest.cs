using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class FloorTest {
        [TestMethod]
        public void SFloorTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)(random.NextDouble() - 0.5) * 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? (float)Math.Floor(x[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Floor(count, x, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DFloorTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)(random.NextDouble() - 0.5) * 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Floor(x[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Floor(count, x, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }
    }
}
