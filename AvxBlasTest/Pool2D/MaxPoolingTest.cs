using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool2DTest {
    [TestClass]
    public class MaxPoolingTest {
        [TestMethod]
        public void SMaxPoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (6, 7), (7, 6), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    foreach ((uint sx, uint sy) in new (uint, uint)[] { (1, 1), (1, 2), (2, 1), (2, 3), (3, 3), (4, 2) }) {
                        uint ow = (iw - 1) / sx + 1;
                        uint oh = (ih - 1) / sy + 1;

                        foreach ((uint kw, uint kh) in new (uint, uint)[] { (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 3), (4, 4), (2, 5), (5, 3) }) {

                            foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {

                                float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                                Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                                Map2D y = Reference(x, (int)sx, (int)kw, (int)sy, (int)kh);

                                Array<float> x_tensor = xval;
                                Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                                Pool2D.MaxPooling(n, c, iw, ih, sx, sy, kw, kh, x_tensor, y_tensor);

                                float[] y_expect = y.ToFloatArray();
                                float[] y_actual = y_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);

                                AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{sx},{sy},{kw},{kh},{n}");

                                Console.WriteLine($"OK: {c},{iw},{ih},{sx},{sy},{kw},{kh},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, int sx, int kw, int sy, int kh) {
            int channels = x.Channels, batch = x.Batch;
            int iw = x.Width, ow = (iw - 1) / sx + 1;
            int ih = x.Height, oh = (ih - 1) / sy + 1;

            Map2D y = new(channels, ow, oh, batch);

            for (int th = 0; th < batch; th++) {
                for (int isy = 0, oy = 0; oy < oh; isy += sy, oy++) {
                    for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                        for (int c = 0; c < channels; c++) {
                            double v = double.MinValue;

                            for (int ky = 0, iy = isy + ky; ky < kh && iy < ih; ky++, iy = isy + ky) {
                                for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                                    v = Math.Max(v, x[c, ix, iy, th]);
                                }
                            }


                            y[c, ox, oy, th] = v;
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, stridex = 2, kwidth = 2, inwidth = 13, stridey = 2, kheight = 2, inheight = 9, batch = 2;

            float[] xval = (new float[batch * inwidth * inheight * channels]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

            Map2D x = new(channels, inwidth, inheight, batch, xval);

            Map2D y = Reference(x, stridex, kwidth, stridey, kheight);

            float[] y_expect = {
                2.70e+01f, 2.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 2.60e+01f, 2.80e+01f, 2.80e+01f, 1.90e+01f, 3.10e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f,
                2.30e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f, 2.20e+01f, 2.70e+01f, 1.80e+01f, 3.60e+01f,
                3.00e+01f, 3.20e+01f, 2.60e+01f, 2.80e+01f, 2.20e+01f, 2.40e+01f, 2.50e+01f, 2.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.40e+01f, 2.50e+01f, 3.00e+01f,
                2.10e+01f, 2.60e+01f, 2.70e+01f, 3.50e+01f, 2.30e+01f, 3.10e+01f, 1.90e+01f, 3.00e+01f, 2.10e+01f, 2.60e+01f, 1.70e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f,
                2.50e+01f, 2.70e+01f, 2.10e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 2.60e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f, 3.80e+01f,
                2.60e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 2.40e+01f, 2.60e+01f, 2.00e+01f, 1.70e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.20e+01f,
                2.90e+01f, 2.80e+01f, 2.50e+01f, 2.60e+01f, 3.10e+01f, 3.30e+01f, 2.70e+01f, 1.80e+01f, 2.30e+01f, 1.80e+01f, 3.20e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f,
                2.70e+01f, 2.90e+01f, 2.30e+01f, 2.50e+01f, 1.90e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.00e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f, 2.20e+01f, 3.40e+01f,
                2.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 2.60e+01f, 2.80e+01f, 2.20e+01f, 1.90e+01f, 3.10e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f,
                3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f, 2.60e+01f, 3.10e+01f, 2.20e+01f, 2.70e+01f, 1.80e+01f, 3.60e+01f, 3.50e+01f,
                3.20e+01f, 2.00e+01f, 2.80e+01f, 1.60e+01f, 2.40e+01f, 2.50e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f,
                2.60e+01f, 2.70e+01f, 3.20e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 2.40e+01f, 2.10e+01f, 2.60e+01f, 1.70e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f,
                2.70e+01f, 2.10e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 2.60e+01f, 3.10e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f, 3.80e+01f, 3.70e+01f,
                3.40e+01f, 2.80e+01f, 3.00e+01f, 2.40e+01f, 2.60e+01f, 2.00e+01f, 2.20e+01f, 2.90e+01f, 3.40e+01f, 2.50e+01f, 3.00e+01f, 2.10e+01f, 2.60e+01f, 1.70e+01f,
                2.80e+01f, 2.30e+01f, 2.40e+01f, 2.50e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 2.30e+01f, 1.80e+01f, 1.90e+01f, 3.70e+01f, 1.50e+01f, 3.30e+01f, 1.10e+01f,
                2.90e+01f, 1.30e+01f, 2.50e+01f, 9.00e+00f, 2.70e+01f, 2.80e+01f, 2.30e+01f, 3.00e+01f, 1.90e+01f, 2.60e+01f, 1.50e+01f, 2.20e+01f, 3.40e+01f, 1.80e+01f,
                3.60e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 2.80e+01f, 1.20e+01f, 2.40e+01f, 3.10e+01f, 1.50e+01f, 3.30e+01f, 2.20e+01f, 2.90e+01f, 1.80e+01f, 2.50e+01f,
                3.00e+00f, 2.10e+01f, 2.20e+01f, 1.70e+01f, 1.80e+01f, 1.30e+01f, 3.10e+01f, 2.50e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.00e+01f, 3.50e+01f, 2.60e+01f,
                3.10e+01f, 2.80e+01f, 2.70e+01f, 2.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.20e+01f, 2.30e+01f, 2.80e+01f, 1.90e+01f, 2.40e+01f, 2.60e+01f, 3.30e+01f,
                2.70e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.60e+01f, 2.30e+01f, 3.00e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f, 2.20e+01f, 2.70e+01f,
                2.40e+01f, 3.60e+01f, 2.40e+01f, 3.20e+01f, 2.60e+01f, 2.80e+01f, 2.20e+01f, 3.00e+01f, 1.80e+01f, 2.60e+01f, 2.70e+01f, 3.20e+01f, 2.30e+01f, 2.80e+01f,
                3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.80e+01f, 2.60e+01f, 3.40e+01f, 3.40e+01f, 2.50e+01f, 3.00e+01f, 2.10e+01f, 2.60e+01f, 1.70e+01f, 3.50e+01f,
                2.30e+01f, 3.10e+01f, 2.50e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 3.20e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f,
                2.60e+01f, 2.50e+01f, 2.20e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 2.40e+01f, 2.10e+01f, 2.60e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.10e+01f, 3.60e+01f,
                2.70e+01f, 3.20e+01f, 2.30e+01f, 2.80e+01f, 1.90e+01f, 3.70e+01f, 2.50e+01f, 3.60e+01f, 2.70e+01f, 3.20e+01f, 2.30e+01f, 2.80e+01f, 1.90e+01f, 3.10e+01f,
                2.10e+01f, 3.30e+01f, 2.70e+01f, 2.90e+01f, 2.30e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.00e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f,
                2.80e+01f, 2.70e+01f, 2.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 2.60e+01f, 2.30e+01f, 2.80e+01f, 1.90e+01f, 2.40e+01f, 2.60e+01f, 3.30e+01f, 3.20e+01f,
                3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.60e+01f, 2.70e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f, 2.50e+01f, 2.00e+01f, 2.10e+01f, 1.60e+01f,
                2.30e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 2.50e+01f, 2.70e+01f, 2.10e+01f, 2.90e+01f, 2.30e+01f, 2.50e+01f, 3.20e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f,
                3.00e+01f, 2.90e+01f, 2.60e+01f, 3.80e+01f, 2.60e+01f, 3.40e+01f, 2.80e+01f, 2.50e+01f, 3.00e+01f, 2.10e+01f, 2.60e+01f, 1.70e+01f, 3.50e+01f, 3.40e+01f,
                3.10e+01f, 2.50e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 3.20e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f,
                2.50e+01f, 1.60e+01f, 3.40e+01f, 2.20e+01f, 3.00e+01f, 1.80e+01f, 2.60e+01f, 2.00e+01f, 1.50e+01f, 2.70e+01f, 3.40e+01f, 2.30e+01f, 3.00e+01f, 1.90e+01f,
                3.20e+01f, 1.00e+01f, 2.80e+01f, 6.00e+00f, 2.40e+01f, 3.60e+01f, 2.00e+01f, 2.70e+01f, 2.20e+01f, 2.30e+01f, 1.80e+01f, 1.90e+01f, 3.10e+01f, 1.50e+01f,
                3.30e+01f, 1.70e+01f, 2.90e+01f, 1.30e+01f, 3.10e+01f, 9.00e+00f, 2.70e+01f, 2.80e+01f, 2.30e+01f, 3.00e+01f, 1.90e+01f, 2.60e+01f, 1.50e+01f, 2.20e+01f,
                1.70e+01f, 2.40e+01f, 3.60e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 2.80e+01f, 1.20e+01f, 1.30e+01f, 8.00e+00f, 2.60e+01f, 2.70e+01f, 2.20e+01f, 2.30e+01f,

            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
