using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool1DTest {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void SAveragePoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    foreach (uint sx in new int[] { 1, 2, 3, 4 }) {
                        uint ow = (iw - 1) / sx + 1;

                        foreach (uint kw in new int[] { 2, 3, 4, 5 }) {

                            foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {

                                float[] xval = (new float[c * iw * n]).Select((_, idx) => (float)(idx * 4547 % 17)).ToArray();

                                Map1D x = new((int)c, (int)iw, (int)n, xval);

                                Map1D y = Reference(x, (int)sx, (int)kw);

                                Array<float> x_tensor = xval;
                                Array<float> y_tensor = new(c * ow * n, zeroset: false);

                                Pool1D.AveragePooling(n, c, iw, sx, kw, x_tensor, y_tensor);

                                float[] y_expect = y.ToFloatArray();
                                float[] y_actual = y_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);

                                AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{sx},{kw},{n}");

                                Console.WriteLine($"OK: {c},{iw},{sx},{kw},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x, int sx, int kw) {
            int channels = x.Channels, batch = x.Batch;
            int iw = x.Width, ow = (iw - 1) / sx + 1;

            Map1D y = new(channels, ow, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
                    for (int c = 0; c < channels; c++) {
                        double v = 0;

                        for (int kx = 0; kx < kw; kx++) {
                            int cx = Math.Min(iw - 1, Math.Max(0, kx + ix - (kw - 1) / 2));

                            v += x[c, cx, th];
                        }

                        y[c, ox, th] = v / kw;
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, stridex = 2, kwidth = 2, inwidth = 13, batch = 2;

            float[] xval = (new float[batch * inwidth * channels]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

            Map1D x = new(channels, inwidth, batch, xval);

            Map1D y = Reference(x, stridex, kwidth);

            float[] y_expect = {
                3.00e+00f, 2.10e+01f, 3.05e+01f, 1.70e+01f, 2.65e+01f, 1.30e+01f, 2.25e+01f, 1.50e+01f, 1.60e+01f, 2.25e+01f, 1.20e+01f, 3.00e+01f, 1.65e+01f, 2.60e+01f,
                1.00e+01f, 2.80e+01f, 6.00e+00f, 2.40e+01f, 2.50e+01f, 2.00e+01f, 2.10e+01f, 1.35e+01f, 2.30e+01f, 9.50e+00f, 1.90e+01f, 1.40e+01f, 1.50e+01f, 3.30e+01f,
                1.70e+01f, 2.65e+01f, 1.30e+01f, 2.25e+01f, 9.00e+00f, 1.85e+01f, 1.65e+01f, 2.05e+01f, 3.00e+01f, 1.65e+01f, 2.60e+01f, 1.25e+01f, 2.20e+01f, 8.50e+00f,
                2.10e+01f, 2.20e+01f, 1.70e+01f, 3.50e+01f, 1.30e+01f, 3.10e+01f, 9.00e+00f, 2.15e+01f, 8.00e+00f, 1.75e+01f, 2.70e+01f, 1.35e+01f, 2.30e+01f, 1.80e+01f,
                2.50e+01f, 1.15e+01f, 2.10e+01f, 1.90e+01f, 1.70e+01f, 2.65e+01f, 1.30e+01f, 2.00e+01f, 1.50e+01f, 2.45e+01f, 1.10e+01f, 2.05e+01f, 3.00e+01f, 1.65e+01f,
                3.20e+01f, 1.00e+01f, 2.80e+01f, 6.00e+00f, 2.40e+01f, 1.05e+01f, 2.00e+01f, 2.70e+01f, 2.20e+01f, 2.30e+01f, 1.80e+01f, 1.90e+01f, 1.40e+01f, 1.50e+01f,
                7.50e+00f, 1.70e+01f, 2.65e+01f, 1.30e+01f, 3.10e+01f, 9.00e+00f, 2.70e+01f, 8.00e+00f, 2.60e+01f, 2.70e+01f, 2.20e+01f, 2.30e+01f, 1.80e+01f, 1.90e+01f,
            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
