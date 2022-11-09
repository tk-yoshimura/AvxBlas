using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class NegTest {
        [TestMethod]
        public void SNegTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? -x[idx] : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Neg(count, x, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }

            {
                float[] x = new float[] {
                    -float.Epsilon, float.Epsilon, float.MinValue, float.MaxValue, float.NegativeInfinity, float.PositiveInfinity
                };

                float[] t = new float[] {
                    float.Epsilon, -float.Epsilon, float.MaxValue, float.MinValue, float.PositiveInfinity, float.NegativeInfinity
                };

                Array<float> y = new(x.Length);

                Elementwise.Neg((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (float[])y);
            }
        }

        [TestMethod]
        public void DNegTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? -x[idx] : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Neg(count, x, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }

            {
                double[] x = new double[] {
                    -double.Epsilon, double.Epsilon, double.MinValue, double.MaxValue, double.NegativeInfinity, double.PositiveInfinity
                };

                double[] t = new double[] {
                    double.Epsilon, -double.Epsilon, double.MaxValue, double.MinValue, double.PositiveInfinity, double.NegativeInfinity
                };

                Array<double> y = new(x.Length);

                Elementwise.Neg((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (double[])y);
            }
        }
    }
}
