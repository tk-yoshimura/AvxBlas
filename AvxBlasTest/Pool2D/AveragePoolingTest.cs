using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool2DTest {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void SAveragePoolingTest() {
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

                                Pool2D.AveragePooling(n, c, iw, ih, sx, sy, kw, kh, x_tensor, y_tensor);

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
                            double v = 0;

                            for (int ky = 0, iy = isy + ky; ky < kh; ky++, iy = Math.Min(ih - 1, isy + ky)) {
                                for (int kx = 0, ix = isx + kx; kx < kw; kx++, ix = Math.Min(iw - 1, isx + kx)) {
                                    v += x[c, ix, iy, th];
                                }
                            }


                            y[c, ox, oy, th] = v / (kw * kh);
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
                1.225e+01f, 1.450e+01f, 2.400e+01f, 2.200e+01f, 2.000e+01f, 1.800e+01f, 2.025e+01f, 2.000e+01f, 1.375e+01f, 2.175e+01f, 1.550e+01f, 2.350e+01f, 2.150e+01f, 1.950e+01f,
                1.500e+01f, 2.150e+01f, 1.525e+01f, 1.750e+01f, 2.275e+01f, 2.500e+01f, 1.875e+01f, 2.275e+01f, 1.650e+01f, 1.875e+01f, 1.250e+01f, 1.900e+01f, 1.275e+01f, 2.650e+01f,
                2.200e+01f, 2.425e+01f, 1.800e+01f, 2.025e+01f, 1.400e+01f, 1.625e+01f, 1.575e+01f, 1.400e+01f, 2.350e+01f, 2.150e+01f, 1.950e+01f, 2.175e+01f, 1.550e+01f, 1.775e+01f,
                1.450e+01f, 2.400e+01f, 2.200e+01f, 2.850e+01f, 1.800e+01f, 2.450e+01f, 1.400e+01f, 2.200e+01f, 1.575e+01f, 1.800e+01f, 1.175e+01f, 2.550e+01f, 1.925e+01f, 2.150e+01f,
                1.700e+01f, 1.925e+01f, 1.300e+01f, 1.950e+01f, 1.900e+01f, 2.125e+01f, 2.075e+01f, 2.475e+01f, 1.850e+01f, 2.075e+01f, 1.450e+01f, 1.675e+01f, 1.050e+01f, 2.850e+01f,
                1.825e+01f, 2.200e+01f, 2.000e+01f, 2.225e+01f, 1.600e+01f, 1.825e+01f, 1.200e+01f, 1.175e+01f, 2.550e+01f, 2.350e+01f, 2.150e+01f, 1.950e+01f, 1.750e+01f, 1.975e+01f,
                1.950e+01f, 1.325e+01f, 1.550e+01f, 2.075e+01f, 2.300e+01f, 2.100e+01f, 1.900e+01f, 1.150e+01f, 1.800e+01f, 1.600e+01f, 2.550e+01f, 2.350e+01f, 2.150e+01f, 1.950e+01f,
                1.900e+01f, 2.125e+01f, 1.500e+01f, 1.725e+01f, 1.100e+01f, 1.750e+01f, 2.275e+01f, 2.250e+01f, 2.050e+01f, 2.275e+01f, 1.650e+01f, 1.875e+01f, 1.250e+01f, 2.050e+01f,
                1.450e+01f, 2.400e+01f, 2.200e+01f, 2.000e+01f, 1.800e+01f, 2.025e+01f, 1.400e+01f, 1.375e+01f, 2.175e+01f, 1.550e+01f, 2.350e+01f, 2.150e+01f, 1.950e+01f, 1.750e+01f,
                2.150e+01f, 1.525e+01f, 1.750e+01f, 2.275e+01f, 2.500e+01f, 1.875e+01f, 2.100e+01f, 1.650e+01f, 1.875e+01f, 1.250e+01f, 1.900e+01f, 1.275e+01f, 2.650e+01f, 2.025e+01f,
                2.550e+01f, 1.500e+01f, 2.150e+01f, 1.100e+01f, 1.750e+01f, 1.850e+01f, 2.500e+01f, 1.875e+01f, 2.250e+01f, 2.050e+01f, 1.850e+01f, 2.075e+01f, 1.450e+01f, 1.675e+01f,
                1.650e+01f, 2.175e+01f, 2.400e+01f, 2.200e+01f, 2.000e+01f, 1.800e+01f, 1.600e+01f, 1.575e+01f, 1.800e+01f, 1.175e+01f, 2.550e+01f, 1.925e+01f, 2.150e+01f, 1.950e+01f,
                1.925e+01f, 1.300e+01f, 1.950e+01f, 1.900e+01f, 2.125e+01f, 2.075e+01f, 2.300e+01f, 1.850e+01f, 2.075e+01f, 1.450e+01f, 1.675e+01f, 1.050e+01f, 2.850e+01f, 2.225e+01f,
                2.200e+01f, 2.000e+01f, 2.225e+01f, 1.600e+01f, 1.825e+01f, 1.200e+01f, 1.425e+01f, 2.250e+01f, 2.900e+01f, 1.850e+01f, 2.500e+01f, 1.450e+01f, 2.100e+01f, 1.050e+01f,
                2.500e+01f, 1.450e+01f, 2.100e+01f, 2.200e+01f, 1.700e+01f, 2.650e+01f, 1.300e+01f, 2.000e+01f, 1.500e+01f, 1.600e+01f, 3.400e+01f, 1.200e+01f, 3.000e+01f, 8.000e+00f,
                2.350e+01f, 1.000e+01f, 1.950e+01f, 6.000e+00f, 2.400e+01f, 2.500e+01f, 2.000e+01f, 2.700e+01f, 1.350e+01f, 2.300e+01f, 9.500e+00f, 1.900e+01f, 1.700e+01f, 1.500e+01f,
                3.050e+01f, 1.700e+01f, 2.650e+01f, 1.300e+01f, 2.250e+01f, 9.000e+00f, 1.850e+01f, 2.250e+01f, 1.200e+01f, 3.000e+01f, 1.650e+01f, 2.600e+01f, 1.250e+01f, 2.200e+01f,
                3.000e+00f, 2.100e+01f, 2.200e+01f, 1.700e+01f, 1.800e+01f, 1.300e+01f, 3.100e+01f, 1.700e+01f, 1.500e+01f, 1.875e+01f, 2.100e+01f, 2.050e+01f, 2.275e+01f, 1.650e+01f,
                1.625e+01f, 1.850e+01f, 1.650e+01f, 1.450e+01f, 2.400e+01f, 2.200e+01f, 2.425e+01f, 2.400e+01f, 1.775e+01f, 2.000e+01f, 1.375e+01f, 1.600e+01f, 1.400e+01f, 2.350e+01f,
                1.900e+01f, 2.550e+01f, 1.925e+01f, 2.150e+01f, 1.525e+01f, 1.750e+01f, 1.700e+01f, 1.525e+01f, 2.050e+01f, 2.275e+01f, 1.650e+01f, 2.300e+01f, 1.675e+01f, 1.900e+01f,
                1.450e+01f, 2.250e+01f, 1.625e+01f, 2.425e+01f, 1.800e+01f, 2.025e+01f, 1.400e+01f, 2.350e+01f, 1.300e+01f, 1.950e+01f, 2.050e+01f, 2.700e+01f, 1.650e+01f, 2.300e+01f,
                1.825e+01f, 2.050e+01f, 1.425e+01f, 1.650e+01f, 2.025e+01f, 1.825e+01f, 2.200e+01f, 2.600e+01f, 1.975e+01f, 2.200e+01f, 1.575e+01f, 1.800e+01f, 1.175e+01f, 2.550e+01f,
                1.525e+01f, 2.325e+01f, 1.700e+01f, 2.350e+01f, 1.725e+01f, 1.950e+01f, 1.325e+01f, 1.725e+01f, 2.250e+01f, 2.475e+01f, 1.850e+01f, 2.075e+01f, 1.450e+01f, 2.100e+01f,
                1.650e+01f, 1.450e+01f, 1.250e+01f, 2.625e+01f, 2.000e+01f, 2.225e+01f, 1.600e+01f, 1.575e+01f, 1.800e+01f, 1.600e+01f, 1.975e+01f, 1.775e+01f, 2.150e+01f, 2.375e+01f,
                2.050e+01f, 1.850e+01f, 1.650e+01f, 1.450e+01f, 1.250e+01f, 3.050e+01f, 2.000e+01f, 2.375e+01f, 1.750e+01f, 2.400e+01f, 1.775e+01f, 2.000e+01f, 1.375e+01f, 2.175e+01f,
                1.150e+01f, 2.525e+01f, 1.900e+01f, 2.125e+01f, 1.500e+01f, 2.150e+01f, 1.525e+01f, 1.500e+01f, 1.875e+01f, 2.100e+01f, 2.050e+01f, 2.275e+01f, 1.650e+01f, 1.875e+01f,
                1.850e+01f, 1.650e+01f, 1.450e+01f, 2.400e+01f, 2.200e+01f, 2.425e+01f, 1.800e+01f, 1.775e+01f, 2.000e+01f, 1.375e+01f, 1.600e+01f, 1.400e+01f, 2.350e+01f, 2.150e+01f,
                2.550e+01f, 1.925e+01f, 2.150e+01f, 1.525e+01f, 1.750e+01f, 1.700e+01f, 1.925e+01f, 1.750e+01f, 2.400e+01f, 1.350e+01f, 2.000e+01f, 1.800e+01f, 1.600e+01f, 1.400e+01f,
                1.350e+01f, 2.725e+01f, 2.100e+01f, 2.325e+01f, 1.700e+01f, 1.925e+01f, 1.300e+01f, 1.700e+01f, 1.500e+01f, 1.300e+01f, 2.250e+01f, 2.475e+01f, 1.850e+01f, 2.075e+01f,
                2.050e+01f, 1.425e+01f, 1.650e+01f, 2.025e+01f, 1.825e+01f, 2.200e+01f, 2.000e+01f, 1.975e+01f, 2.200e+01f, 1.575e+01f, 1.800e+01f, 1.175e+01f, 2.550e+01f, 2.350e+01f,
                2.325e+01f, 1.700e+01f, 2.350e+01f, 1.725e+01f, 1.950e+01f, 1.325e+01f, 1.550e+01f, 2.250e+01f, 2.475e+01f, 1.850e+01f, 2.075e+01f, 1.450e+01f, 2.100e+01f, 1.475e+01f,
                2.000e+01f, 9.500e+00f, 2.750e+01f, 1.700e+01f, 2.350e+01f, 1.300e+01f, 1.950e+01f, 1.700e+01f, 1.200e+01f, 2.150e+01f, 3.100e+01f, 1.750e+01f, 2.700e+01f, 1.350e+01f,
                2.900e+01f, 7.000e+00f, 2.500e+01f, 3.000e+00f, 2.100e+01f, 3.050e+01f, 1.700e+01f, 2.400e+01f, 1.900e+01f, 2.000e+01f, 1.500e+01f, 1.600e+01f, 2.250e+01f, 1.200e+01f,
                2.750e+01f, 1.400e+01f, 2.350e+01f, 1.000e+01f, 2.800e+01f, 6.000e+00f, 2.400e+01f, 1.950e+01f, 1.750e+01f, 2.700e+01f, 1.350e+01f, 2.300e+01f, 9.500e+00f, 1.900e+01f,
                1.150e+01f, 2.100e+01f, 3.050e+01f, 1.700e+01f, 2.650e+01f, 1.300e+01f, 2.250e+01f, 1.200e+01f, 1.300e+01f, 8.000e+00f, 2.600e+01f, 2.700e+01f, 2.200e+01f, 2.300e+01f,

            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
