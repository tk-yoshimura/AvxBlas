using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;

namespace AvxBlasTest.DenseTest {
    [TestClass]
    public class BackwardFilterTest {
        [TestMethod]
        public void SBackwardFilterTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] yval = (new float[oc * n]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                        Map0D x = new((int)ic, (int)n, xval);
                        Map0D y = new((int)oc, (int)n, yval);

                        Filter0D gw = Reference(x, y);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = yval;

                        Array<float> gw_tensor = new(ic * oc);

                        Dense.BackwardFilter(n, ic, oc, x_tensor, y_tensor, gw_tensor);

                        float[] gw_expect = gw.ToFloatArray();
                        float[] gw_actual = gw_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);
                        CollectionAssert.AreEqual(yval, (float[])y_tensor);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-8f, 1e-6f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void DBackwardFilterTest() {
            double max_err = 0;

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        double[] xval = (new double[ic * n]).Select((_, idx) => idx * 1e-3).ToArray();
                        double[] yval = (new double[oc * n]).Select((_, idx) => idx * 1e-3).Reverse().ToArray();

                        Map0D x = new((int)ic, (int)n, xval);
                        Map0D y = new((int)oc, (int)n, yval);

                        Filter0D gw = Reference(x, y);

                        Array<double> x_tensor = xval;
                        Array<double> y_tensor = yval;

                        Array<double> gw_tensor = new(ic * oc);

                        Dense.BackwardFilter(n, ic, oc, x_tensor, y_tensor, gw_tensor);

                        double[] gw_expect = gw.ToDoubleArray();
                        double[] gw_actual = gw_tensor;

                        CollectionAssert.AreEqual(xval, (double[])x_tensor);
                        CollectionAssert.AreEqual(yval, (double[])y_tensor);

                        AssertError.Tolerance(gw_expect, gw_actual, 1e-8f, 1e-6f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Filter0D Reference(Map0D x, Map0D y) {
            int inchannels = x.Channels, outchannels = y.Channels, batch = x.Batch;

            Filter0D w = new(inchannels, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int inch, outch = 0; outch < outchannels; outch++) {
                    for (inch = 0; inch < inchannels; inch++) {
                        w[inch, outch] += x[inch, th] * y[outch, th];
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[batch * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] yval = (new float[batch * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D x = new(inchannels, batch, xval);
            Map0D y = new(outchannels, batch, yval);

            Filter0D gw = Reference(x, y);

            float[] gw_expect = {
                7.0000e-05f, 1.0100e-04f, 1.3200e-04f, 1.6300e-04f, 1.9400e-04f,
                2.2500e-04f, 2.5600e-04f, 6.3000e-05f, 9.2000e-05f, 1.2100e-04f,
                1.5000e-04f, 1.7900e-04f, 2.0800e-04f, 2.3700e-04f, 5.6000e-05f,
                8.3000e-05f, 1.1000e-04f, 1.3700e-04f, 1.6400e-04f, 1.9100e-04f,
                2.1800e-04f, 4.9000e-05f, 7.4000e-05f, 9.9000e-05f, 1.2400e-04f,
                1.4900e-04f, 1.7400e-04f, 1.9900e-04f, 4.2000e-05f, 6.5000e-05f,
                8.8000e-05f, 1.1100e-04f, 1.3400e-04f, 1.5700e-04f, 1.8000e-04f,
                3.5000e-05f, 5.6000e-05f, 7.7000e-05f, 9.8000e-05f, 1.1900e-04f,
                1.4000e-04f, 1.6100e-04f, 2.8000e-05f, 4.7000e-05f, 6.6000e-05f,
                8.5000e-05f, 1.0400e-04f, 1.2300e-04f, 1.4200e-04f, 2.1000e-05f,
                3.8000e-05f, 5.5000e-05f, 7.2000e-05f, 8.9000e-05f, 1.0600e-04f,
                1.2300e-04f, 1.4000e-05f, 2.9000e-05f, 4.4000e-05f, 5.9000e-05f,
                7.4000e-05f, 8.9000e-05f, 1.0400e-04f, 7.0000e-06f, 2.0000e-05f,
                3.3000e-05f, 4.6000e-05f, 5.9000e-05f, 7.2000e-05f, 8.5000e-05f,
                0.0000e+00f, 1.1000e-05f, 2.2000e-05f, 3.3000e-05f, 4.4000e-05f,
                5.5000e-05f, 6.6000e-05f
            };

            float[] gw_actual = gw.ToFloatArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-8f, 1e-6f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
