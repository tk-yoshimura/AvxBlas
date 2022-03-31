using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.DenseTest {
    [TestClass]
    public class BackwardDataTest {
        [TestMethod]
        public void SBackwardDataTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] yval = (new float[oc * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[ic * oc]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                        Map0D y = new((int)oc, (int)n, yval);
                        Filter0D w = new((int)ic, (int)oc, wval);

                        Map0D x = Reference(y, w);

                        Array<float> y_tensor = yval;
                        Array<float> w_tensor = wval;

                        Array<float> x_tensor = new(ic * n, zeroset: false);

                        Dense.BackwardData(n, ic, oc, y_tensor, w_tensor, x_tensor);

                        float[] x_expect = x.ToFloatArray();
                        float[] x_actual = x_tensor;

                        CollectionAssert.AreEqual(yval, (float[])y_tensor);
                        CollectionAssert.AreEqual(wval, (float[])w_tensor);

                        AssertError.Tolerance(x_expect, x_actual, 1e-10f, 1e-5f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void DBackwardDataTest() {
            double max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        double[] yval = (new double[oc * n]).Select((_, idx) => idx * 1e-3).ToArray();
                        double[] wval = (new double[ic * oc]).Select((_, idx) => (idx + 1) * 1e-3).Reverse().ToArray();

                        Map0D y = new((int)oc, (int)n, yval);
                        Filter0D w = new((int)ic, (int)oc, wval);

                        Map0D x = Reference(y, w);

                        Array<double> y_tensor = yval;
                        Array<double> w_tensor = wval;

                        Array<double> x_tensor = new(ic * n, zeroset: false);

                        Dense.BackwardData(n, ic, oc, y_tensor, w_tensor, x_tensor);

                        double[] x_expect = x.ToDoubleArray();
                        double[] x_actual = x_tensor;

                        CollectionAssert.AreEqual(yval, (double[])y_tensor);
                        CollectionAssert.AreEqual(wval, (double[])w_tensor);

                        AssertError.Tolerance(x_expect, x_actual, 1e-10f, 1e-5f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map0D Reference(Map0D y, Filter0D w) {
            int outchannels = y.Channels, inchannels = w.InChannels, batch = y.Batch;

            Map0D x = new(inchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int inch = 0; inch < inchannels; inch++) {
                    double sum = 0;

                    for (int outch = 0; outch < outchannels; outch++) {
                        sum += y[outch, th] * w[inch, outch];
                    }

                    x[inch, th] = sum;
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] yval = (new float[outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D y = new(outchannels, batch, yval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D x = Reference(y, w);

            float[] x_expect = {
                1.4850e-03f, 1.4300e-03f, 1.3750e-03f, 1.3200e-03f, 1.2650e-03f,
                1.2100e-03f, 1.1550e-03f, 6.4460e-03f, 6.2700e-03f, 6.0940e-03f,
                5.9180e-03f, 5.7420e-03f, 5.5660e-03f, 5.3900e-03f
            };

            float[] x_actual = x.ToFloatArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-8f, 1e-6f);
        }
    }
}
