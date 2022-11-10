using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class NanAsZeroTest {
        [TestMethod]
        public void SNanAsZeroTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    if (count > 0) {
                        x[random.Next((int)count)] = float.NaN;
                    }

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count && !float.IsNaN(x[idx]) ? x[idx] : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.NanAsZero(count, x, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }

        [TestMethod]
        public void DNanAsZeroTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                    if (count > 0) {
                        x[random.Next((int)count)] = double.NaN;
                    }

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count && !double.IsNaN(x[idx]) ? x[idx] : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.NanAsZero(count, x, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }
        }
    }
}
