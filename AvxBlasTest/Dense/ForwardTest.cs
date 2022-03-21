using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.DenseTest {
    [TestClass]
    public class ForwardTest {
        [TestMethod]
        public void SForwardTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                        float[] wval = (new float[ic * oc]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                        Map0D x = new((int)ic, (int)n, xval);
                        Filter0D w = new((int)ic, (int)oc, wval);

                        Map0D y = Reference(x, w);

                        Array<float> x_tensor = xval;
                        Array<float> w_tensor = wval;
                        Array<float> y_tensor = new(oc * n);

                        Dense.Forward(n, ic, oc, x_tensor, w_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);
                        CollectionAssert.AreEqual(wval, (float[])w_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-8f, 1e-6f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        [TestMethod]
        public void DForwardTest() {
            double max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                    foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        double[] xval = (new double[ic * n]).Select((_, idx) => idx * 1e-3).ToArray();
                        double[] wval = (new double[ic * oc]).Select((_, idx) => (idx + 1) * 1e-3).Reverse().ToArray();

                        Map0D x = new((int)ic, (int)n, xval);
                        Filter0D w = new((int)ic, (int)oc, wval);

                        Map0D y = Reference(x, w);

                        Array<double> x_tensor = xval;
                        Array<double> w_tensor = wval;
                        Array<double> y_tensor = new(oc * n);

                        Dense.Forward(n, ic, oc, x_tensor, w_tensor, y_tensor);

                        double[] y_expect = y.ToDoubleArray();
                        double[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (double[])x_tensor);
                        CollectionAssert.AreEqual(wval, (double[])w_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-8f, 1e-6f, ref max_err, $"NG {ic},{oc},{n}");

                        Console.WriteLine($"OK: {ic},{oc},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map0D Reference(Map0D x, Filter0D w) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;

            Map0D y = new(outchannels, batch);

            for (int th = 0; th < batch; th++) {
                for (int outch = 0; outch < outchannels; outch++) {
                    double sum = 0;

                    for (int inch = 0; inch < inchannels; inch++) {
                        sum += x[inch, th] * w[inch, outch];
                    }

                    y[outch, th] = sum;
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, batch = 2;

            float[] xval = (new float[inchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map0D x = new(inchannels, batch, xval);
            Filter0D w = new(inchannels, outchannels, wval);

            Map0D y = Reference(x, w);

            float[] y_expect = {
                1.5050e-03f, 1.3580e-03f, 1.2110e-03f, 1.0640e-03f, 9.1700e-04f,
                7.7000e-04f, 6.2300e-04f, 4.7600e-04f, 3.2900e-04f, 1.8200e-04f,
                3.5000e-05f, 5.0820e-03f, 4.5920e-03f, 4.1020e-03f, 3.6120e-03f,
                3.1220e-03f, 2.6320e-03f, 2.1420e-03f, 1.6520e-03f, 1.1620e-03f,
                6.7200e-04f, 1.8200e-04f
            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-8f, 1e-6f, $"mismatch value {inchannels},{outchannels},{batch}");
        }
    }
}
