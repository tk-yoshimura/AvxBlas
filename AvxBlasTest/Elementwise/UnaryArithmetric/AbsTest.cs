using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class AbsTest {
        [TestMethod]
        public void SAbsTest() {
            Random random = new Random(1234);

            for (uint length = 1; length <= 16; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Abs(x[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Abs(count, x, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }

            {
                float[] x = new float[] {
                    -float.Epsilon, float.Epsilon, float.MinValue, float.MaxValue, float.NegativeInfinity, float.PositiveInfinity
                };

                float[] t = new float[] {
                    float.Epsilon, float.Epsilon, float.MaxValue, float.MaxValue, float.PositiveInfinity, float.PositiveInfinity
                };

                Array<float> y = new(x.Length);

                Elementwise.Abs((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (float[])y);
            }
        }

        [TestMethod]
        public void DAbsTest() {
            Random random = new Random(1234);

            for (uint length = 1; length <= 16; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Abs(x[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Abs(count, x, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }

            {
                double[] x = new double[] {
                    -double.Epsilon, double.Epsilon, double.MinValue, double.MaxValue, double.NegativeInfinity, double.PositiveInfinity
                };

                double[] t = new double[] {
                    double.Epsilon, double.Epsilon, double.MaxValue, double.MaxValue, double.PositiveInfinity, double.PositiveInfinity
                };

                Array<double> y = new(x.Length);

                Elementwise.Abs((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (double[])y);
            }
        }
    }
}
