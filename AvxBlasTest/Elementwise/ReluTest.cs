using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class ReluTest {
        [TestMethod]
        public void SReluTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[length])
                        .Select((_, idx) => idx < count ? Math.Max(0, x[idx]) : 0)
                        .ToArray();

                    Array<float> y = new(length);

                    Elementwise.Relu(count, x, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }

            {
                float[] x = new float[] {
                    -float.Epsilon, float.Epsilon, float.MinValue, float.MaxValue, float.NegativeInfinity, float.PositiveInfinity
                };

                float[] t = new float[] {
                    0, float.Epsilon, 0, float.MaxValue, 0, float.PositiveInfinity
                };

                Array<float> y = new(x.Length);

                Elementwise.Relu((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (float[])y);
            }
        }

        [TestMethod]
        public void DReluTest() {
            Random random = new(1234);

            for (uint length = 1; length <= 64; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[length])
                        .Select((_, idx) => idx < count ? Math.Max(0, x[idx]) : 0)
                        .ToArray();

                    Array<double> y = new(length);

                    Elementwise.Relu(count, x, y);

                    CollectionAssert.AreEqual(t, (double[])y);
                }
            }

            {
                double[] x = new double[] {
                    -double.Epsilon, double.Epsilon, double.MinValue, double.MaxValue, double.NegativeInfinity, double.PositiveInfinity
                };

                double[] t = new double[] {
                    0, double.Epsilon, 0, double.MaxValue, 0, double.PositiveInfinity
                };

                Array<double> y = new(x.Length);

                Elementwise.Relu((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (double[])y);
            }
        }
    }
}
