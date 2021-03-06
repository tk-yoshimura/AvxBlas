using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool3DTest {
    [TestClass]
    public class AveragePoolingTest {
        [TestMethod]
        public void SAveragePoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih, uint id, uint sx, uint sy, uint sz) in new (uint, uint, uint, uint, uint, uint)[] {
                    (1, 1, 1, 1, 1, 1), (1, 1, 4, 1, 1, 2), (1, 4, 1, 1, 2, 1), (4, 1, 1, 2, 1, 1), (4, 3, 2, 2, 2, 3), (5, 8, 4, 2, 2, 2), (7, 16, 15, 2, 2, 3), (17, 6, 28, 4, 2, 2), (32, 31, 30, 2, 2, 3),
                    (1, 1, 1, 1, 1, 2), (1, 1, 4, 1, 2, 1), (1, 4, 1, 1, 1, 1), (4, 1, 1, 2, 2, 2), (4, 3, 2, 2, 1, 1), (5, 8, 4, 4, 2, 2), (7, 16, 15, 2, 2, 2), (17, 6, 28, 2, 2, 3), (32, 31, 30, 1, 1, 1)}) {

                    uint ow = (iw - 1) / sx + 1;
                    uint oh = (ih - 1) / sy + 1;
                    uint od = (id - 1) / sz + 1;

                    foreach ((uint kw, uint kh, uint kd) in new (uint, uint, uint)[] {
                        (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2), (2, 2, 4), (2, 4, 3), (4, 4, 4), (3, 2, 5), (2, 5, 3) }) {

                        foreach (uint c in new uint[] { 1, 2, 4, 5, 8, 10, 15, 16, 20, 32, 33 }) {

                            float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                            Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                            Map3D y = Reference(x, (int)sx, (int)kw, (int)sy, (int)kh, (int)sz, (int)kd);

                            Array<float> x_tensor = xval;
                            Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                            Pool3D.AveragePooling(n, c, iw, ih, id, sx, sy, sz, kw, kh, kd, x_tensor, y_tensor);

                            float[] y_expect = y.ToFloatArray();
                            float[] y_actual = y_tensor;

                            CollectionAssert.AreEqual(xval, (float[])x_tensor);

                            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{sx},{sy},{sz},{kw},{kh},{kd},{n}");

                            Console.WriteLine($"OK: {c},{iw},{ih},{id},{sx},{sy},{sz},{kw},{kh},{kd},{n}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, int sx, int kw, int sy, int kh, int sz, int kd) {
            int channels = x.Channels, batch = x.Batch;
            int iw = x.Width, ow = (iw - 1) / sx + 1;
            int ih = x.Height, oh = (ih - 1) / sy + 1;
            int id = x.Depth, od = (id - 1) / sz + 1;

            Map3D y = new(channels, ow, oh, od, batch);

            for (int th = 0; th < batch; th++) {
                for (int isz = 0, oz = 0; oz < od; isz += sz, oz++) {
                    for (int isy = 0, oy = 0; oy < oh; isy += sy, oy++) {
                        for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                            for (int c = 0; c < channels; c++) {
                                double v = 0;

                                for (int kz = 0, iz = isz + kz; kz < kd; kz++, iz = Math.Min(id - 1, isz + kz)) {
                                    for (int ky = 0, iy = isy + ky; ky < kh; ky++, iy = Math.Min(ih - 1, isy + ky)) {
                                        for (int kx = 0, ix = isx + kx; kx < kw; kx++, ix = Math.Min(iw - 1, isx + kx)) {
                                            v += x[c, ix, iy, iz, th];
                                        }
                                    }
                                }

                                y[c, ox, oy, oz, th] = v / (kw * kh * kd);
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int channels = 7, stridex = 2, kwidth = 2, inwidth = 13, stridey = 2, kheight = 2, inheight = 9, stridez = 2, kdepth = 2, indepth = 8, batch = 2;

            float[] xval = (new float[batch * inwidth * inheight * indepth * channels]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

            Map3D x = new(channels, inwidth, inheight, indepth, batch, xval);

            Map3D y = Reference(x, stridex, kwidth, stridey, kheight, stridez, kdepth);

            float[] y_expect = {
                1.4625e+01f, 1.4750e+01f, 2.1375e+01f, 2.1500e+01f, 2.0250e+01f, 2.0375e+01f, 1.8375e+01f, 1.8125e+01f, 1.6125e+01f, 1.9125e+01f, 1.5000e+01f, 2.3750e+01f, 2.1750e+01f, 2.1875e+01f,
                1.9500e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.9375e+01f, 1.9500e+01f, 2.1125e+01f, 2.0875e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.7125e+01f, 1.5125e+01f, 2.1750e+01f,
                1.8625e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.8500e+01f, 1.6500e+01f, 1.7375e+01f, 1.4250e+01f, 2.3000e+01f, 1.8875e+01f, 2.1875e+01f, 1.9875e+01f, 1.7875e+01f, 1.5875e+01f,
                1.9000e+01f, 1.8500e+01f, 2.0750e+01f, 2.4500e+01f, 2.2500e+01f, 2.0500e+01f, 1.8500e+01f, 2.0125e+01f, 1.8125e+01f, 1.6125e+01f, 1.4125e+01f, 2.2875e+01f, 1.8750e+01f, 2.1750e+01f,
                2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.7625e+01f, 1.8500e+01f, 1.6500e+01f, 2.3125e+01f, 2.0000e+01f, 2.0875e+01f, 1.8875e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f, 2.0875e+01f,
                1.7750e+01f, 2.2250e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.6375e+01f, 1.6500e+01f, 1.4125e+01f, 2.0000e+01f, 1.8000e+01f, 2.3875e+01f, 1.9750e+01f, 1.9875e+01f, 1.7875e+01f,
                1.7625e+01f, 1.5625e+01f, 1.5750e+01f, 2.0250e+01f, 2.0375e+01f, 2.1250e+01f, 2.1375e+01f, 1.6000e+01f, 1.8250e+01f, 1.6250e+01f, 2.0000e+01f, 1.8000e+01f, 2.6000e+01f, 1.9750e+01f,
                2.1375e+01f, 1.9375e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.5625e+01f, 2.2250e+01f, 1.7000e+01f, 2.2875e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.7000e+01f, 1.7875e+01f,
                1.4750e+01f, 2.1375e+01f, 2.1500e+01f, 2.0250e+01f, 2.0375e+01f, 1.8375e+01f, 1.6375e+01f, 1.6125e+01f, 1.9125e+01f, 1.5000e+01f, 2.3750e+01f, 2.1750e+01f, 2.1875e+01f, 1.7750e+01f,
                1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.9375e+01f, 1.9500e+01f, 2.1125e+01f, 2.1250e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.7125e+01f, 1.5125e+01f, 2.1750e+01f, 1.9750e+01f,
                2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.7750e+01f, 1.7250e+01f, 1.9500e+01f, 1.6125e+01f, 2.4875e+01f, 2.0750e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f,
                1.6750e+01f, 1.8375e+01f, 1.8500e+01f, 2.2250e+01f, 2.2375e+01f, 1.8250e+01f, 1.8375e+01f, 1.8125e+01f, 1.6125e+01f, 1.4125e+01f, 2.2875e+01f, 1.8750e+01f, 2.1750e+01f, 1.9750e+01f,
                1.9500e+01f, 1.7500e+01f, 1.7625e+01f, 1.8500e+01f, 1.6500e+01f, 2.3125e+01f, 2.3250e+01f, 2.0875e+01f, 1.8875e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f, 2.0875e+01f, 1.8875e+01f,
                2.2250e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.6375e+01f, 1.6500e+01f, 1.4500e+01f, 2.1250e+01f, 1.9250e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f,
                2.1000e+01f, 1.3250e+01f, 2.1250e+01f, 2.6500e+01f, 1.7250e+01f, 2.6750e+01f, 1.3250e+01f, 2.4500e+01f, 1.1000e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 3.0250e+01f, 1.2500e+01f,
                2.3750e+01f, 1.4500e+01f, 1.9750e+01f, 1.0500e+01f, 2.0000e+01f, 2.3750e+01f, 1.6000e+01f, 2.7250e+01f, 1.3750e+01f, 2.3250e+01f, 9.7500e+00f, 2.3500e+01f, 1.1500e+01f, 1.9500e+01f,
                2.5000e+01f, 1.7250e+01f, 2.6750e+01f, 1.3250e+01f, 2.2750e+01f, 9.2500e+00f, 1.8750e+01f, 1.7000e+01f, 1.6500e+01f, 3.0250e+01f, 1.6750e+01f, 2.6250e+01f, 1.2750e+01f, 2.2250e+01f,
                7.5000e+00f, 1.7000e+01f, 1.5000e+01f, 2.1500e+01f, 2.2500e+01f, 1.7500e+01f, 2.7000e+01f, 1.9875e+01f, 1.7875e+01f, 1.5875e+01f, 1.6000e+01f, 1.9750e+01f, 1.7750e+01f, 2.3625e+01f,
                2.3375e+01f, 1.9250e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.3375e+01f, 2.2125e+01f, 1.6875e+01f, 2.2750e+01f, 2.0750e+01f, 2.0875e+01f, 1.6750e+01f, 1.6875e+01f, 1.7750e+01f,
                1.4625e+01f, 2.1250e+01f, 2.1375e+01f, 2.0125e+01f, 2.0250e+01f, 1.8250e+01f, 1.8375e+01f, 1.6000e+01f, 1.9000e+01f, 1.7000e+01f, 2.3625e+01f, 2.1625e+01f, 2.1750e+01f, 1.7625e+01f,
                1.7375e+01f, 1.7500e+01f, 1.5500e+01f, 1.9250e+01f, 1.9375e+01f, 2.3125e+01f, 2.1125e+01f, 2.0000e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 1.2000e+01f, 2.5750e+01f, 1.9500e+01f,
                2.0375e+01f, 2.1250e+01f, 2.1375e+01f, 1.7250e+01f, 1.7375e+01f, 1.5375e+01f, 1.9125e+01f, 1.6000e+01f, 2.4750e+01f, 2.0625e+01f, 2.0750e+01f, 1.8750e+01f, 1.8875e+01f, 1.4750e+01f,
                1.6625e+01f, 2.0375e+01f, 1.8375e+01f, 2.2125e+01f, 2.2250e+01f, 1.8125e+01f, 1.8250e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 2.2750e+01f, 2.0750e+01f, 2.1625e+01f, 1.9625e+01f,
                1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.8375e+01f, 1.6375e+01f, 2.3000e+01f, 2.3125e+01f, 2.2875e+01f, 1.8750e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f, 1.8625e+01f, 1.8750e+01f,
                2.1250e+01f, 2.3500e+01f, 2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.3500e+01f, 1.5125e+01f, 2.1750e+01f, 1.9750e+01f, 2.2750e+01f, 1.8625e+01f, 1.8750e+01f, 1.6750e+01f,
                1.6500e+01f, 1.7375e+01f, 1.7500e+01f, 2.4125e+01f, 2.2125e+01f, 2.0125e+01f, 2.0250e+01f, 1.7875e+01f, 1.5875e+01f, 1.6000e+01f, 1.9750e+01f, 1.7750e+01f, 2.3625e+01f, 2.1625e+01f,
                1.9250e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.3375e+01f, 2.2125e+01f, 2.0125e+01f, 2.2750e+01f, 2.0750e+01f, 2.0875e+01f, 1.6750e+01f, 1.6875e+01f, 1.7750e+01f, 1.5750e+01f,
                2.1250e+01f, 2.1375e+01f, 2.0125e+01f, 2.0250e+01f, 1.8250e+01f, 1.8375e+01f, 1.4250e+01f, 2.1000e+01f, 1.9000e+01f, 2.2750e+01f, 2.0750e+01f, 2.3000e+01f, 1.6750e+01f, 1.9000e+01f,
                1.6375e+01f, 1.6500e+01f, 1.4500e+01f, 2.1125e+01f, 2.1250e+01f, 2.2125e+01f, 2.0125e+01f, 1.9875e+01f, 1.7875e+01f, 1.5875e+01f, 1.6750e+01f, 1.6875e+01f, 2.3500e+01f, 2.1500e+01f,
                2.1250e+01f, 2.1375e+01f, 1.7250e+01f, 1.7375e+01f, 1.5375e+01f, 1.9125e+01f, 1.7125e+01f, 2.4750e+01f, 2.0625e+01f, 2.0750e+01f, 1.8750e+01f, 1.8875e+01f, 1.4750e+01f, 1.4875e+01f,
                2.0375e+01f, 1.8375e+01f, 2.2125e+01f, 2.2250e+01f, 1.8125e+01f, 1.8250e+01f, 1.6250e+01f, 1.6000e+01f, 1.4000e+01f, 2.2750e+01f, 2.0750e+01f, 2.1625e+01f, 1.9625e+01f, 1.9750e+01f,
                1.6500e+01f, 1.4500e+01f, 1.8250e+01f, 2.0500e+01f, 2.0000e+01f, 2.2250e+01f, 2.0250e+01f, 2.6250e+01f, 1.2750e+01f, 2.2250e+01f, 8.7500e+00f, 1.8250e+01f, 2.2000e+01f, 1.8500e+01f,
                2.5500e+01f, 1.6250e+01f, 2.5750e+01f, 1.2250e+01f, 2.1750e+01f, 1.4000e+01f, 1.7750e+01f, 2.3250e+01f, 1.5500e+01f, 2.9250e+01f, 1.1500e+01f, 2.5250e+01f, 1.1750e+01f, 2.1250e+01f,
                1.5250e+01f, 1.9000e+01f, 2.8500e+01f, 1.5000e+01f, 2.4500e+01f, 1.5250e+01f, 2.0500e+01f, 8.7500e+00f, 2.2500e+01f, 1.6250e+01f, 1.8500e+01f, 2.8000e+01f, 1.4500e+01f, 2.4000e+01f,
                1.2250e+01f, 2.1750e+01f, 8.2500e+00f, 1.7750e+01f, 2.5750e+01f, 1.3750e+01f, 2.7500e+01f, 1.7000e+01f, 2.6500e+01f, 1.3000e+01f, 2.2500e+01f, 2.0500e+01f, 1.8500e+01f, 2.8000e+01f,
                1.9375e+01f, 2.3125e+01f, 2.1125e+01f, 1.9125e+01f, 1.9250e+01f, 1.5125e+01f, 1.8125e+01f, 1.5000e+01f, 2.1625e+01f, 1.9625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.6625e+01f,
                1.6375e+01f, 1.7250e+01f, 1.5250e+01f, 2.4000e+01f, 2.2000e+01f, 2.0000e+01f, 2.0125e+01f, 1.9875e+01f, 1.5750e+01f, 1.5875e+01f, 1.9625e+01f, 1.7625e+01f, 2.1375e+01f, 2.1500e+01f,
                1.9125e+01f, 1.9250e+01f, 1.7250e+01f, 1.7375e+01f, 1.3250e+01f, 2.2000e+01f, 2.0000e+01f, 2.2625e+01f, 2.0625e+01f, 2.0750e+01f, 1.6625e+01f, 1.6750e+01f, 1.7625e+01f, 1.7750e+01f,
                2.1000e+01f, 2.3250e+01f, 2.1250e+01f, 1.9250e+01f, 1.7250e+01f, 1.9500e+01f, 1.3250e+01f, 1.4875e+01f, 1.8625e+01f, 1.8750e+01f, 2.2500e+01f, 2.0500e+01f, 2.0625e+01f, 1.8625e+01f,
                1.6250e+01f, 1.6375e+01f, 1.4375e+01f, 2.1000e+01f, 1.9000e+01f, 2.2000e+01f, 2.0000e+01f, 1.9750e+01f, 1.7750e+01f, 1.7875e+01f, 1.6625e+01f, 1.6750e+01f, 2.3375e+01f, 2.1375e+01f,
                2.1125e+01f, 2.1250e+01f, 1.7125e+01f, 1.7250e+01f, 1.5250e+01f, 2.1125e+01f, 1.7000e+01f, 2.4625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.8750e+01f, 1.4625e+01f, 1.4750e+01f,
                2.0250e+01f, 1.8250e+01f, 2.2000e+01f, 2.2125e+01f, 2.0125e+01f, 1.8125e+01f, 1.6125e+01f, 1.5000e+01f, 1.3000e+01f, 2.2500e+01f, 2.0500e+01f, 2.2750e+01f, 1.6500e+01f, 1.8750e+01f,
                1.8250e+01f, 1.8375e+01f, 1.4250e+01f, 2.0125e+01f, 1.8125e+01f, 2.1875e+01f, 1.9875e+01f, 2.1750e+01f, 1.7625e+01f, 1.7750e+01f, 1.5750e+01f, 1.5875e+01f, 2.0375e+01f, 2.0500e+01f,
                2.3125e+01f, 2.1125e+01f, 1.9125e+01f, 1.9250e+01f, 1.5125e+01f, 1.8125e+01f, 1.6125e+01f, 2.1625e+01f, 1.9625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.6625e+01f, 1.6750e+01f,
                1.7250e+01f, 1.5250e+01f, 2.4000e+01f, 2.2000e+01f, 2.0000e+01f, 2.0125e+01f, 1.8125e+01f, 1.5750e+01f, 1.5875e+01f, 1.9625e+01f, 1.7625e+01f, 2.1375e+01f, 2.1500e+01f, 1.9500e+01f,
                2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 1.4500e+01f, 2.4000e+01f, 2.2000e+01f, 2.0000e+01f, 2.1625e+01f, 1.9625e+01f, 1.9750e+01f, 1.5625e+01f, 1.5750e+01f, 1.9500e+01f, 1.9625e+01f,
                2.3000e+01f, 2.3125e+01f, 2.1125e+01f, 1.9125e+01f, 1.7125e+01f, 1.7250e+01f, 1.3125e+01f, 1.8625e+01f, 1.8750e+01f, 2.2500e+01f, 2.0500e+01f, 2.0625e+01f, 1.8625e+01f, 1.6625e+01f,
                1.6375e+01f, 1.4375e+01f, 2.1000e+01f, 1.9000e+01f, 2.2000e+01f, 2.0000e+01f, 1.8000e+01f, 1.7750e+01f, 1.7875e+01f, 1.6625e+01f, 1.6750e+01f, 2.3375e+01f, 2.1375e+01f, 1.9375e+01f,
                2.1250e+01f, 1.7125e+01f, 1.7250e+01f, 1.5250e+01f, 2.1125e+01f, 1.7000e+01f, 2.2875e+01f, 2.1750e+01f, 1.9750e+01f, 1.7750e+01f, 2.0000e+01f, 1.3750e+01f, 1.6000e+01f, 2.5500e+01f,
                2.5750e+01f, 1.3750e+01f, 2.7500e+01f, 1.4000e+01f, 2.3500e+01f, 1.0000e+01f, 1.9500e+01f, 1.3500e+01f, 1.7250e+01f, 2.6750e+01f, 1.7500e+01f, 2.2750e+01f, 1.3500e+01f, 2.3000e+01f,
                1.1250e+01f, 2.0750e+01f, 1.8750e+01f, 1.6750e+01f, 2.6250e+01f, 1.2750e+01f, 2.6500e+01f, 1.0500e+01f, 2.4250e+01f, 1.0750e+01f, 2.0250e+01f, 2.4000e+01f, 1.6250e+01f, 2.5750e+01f,
                1.4000e+01f, 2.3500e+01f, 1.4250e+01f, 1.9500e+01f, 1.6000e+01f, 1.9750e+01f, 2.9250e+01f, 1.7500e+01f, 2.7000e+01f, 1.3500e+01f, 2.3000e+01f, 9.5000e+00f, 2.3250e+01f, 1.7000e+01f,
                1.8000e+01f, 2.7500e+01f, 1.4000e+01f, 2.3500e+01f, 1.0000e+01f, 1.9500e+01f, 6.0000e+00f, 1.6750e+01f, 1.7625e+01f, 1.7750e+01f, 2.2250e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f,
                1.8125e+01f, 1.8250e+01f, 1.4125e+01f, 2.0000e+01f, 1.8000e+01f, 2.3875e+01f, 1.9750e+01f, 2.1625e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.5750e+01f, 2.0250e+01f, 2.0375e+01f,
                2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.9125e+01f, 1.7125e+01f, 1.8000e+01f, 1.6000e+01f, 2.1500e+01f, 1.9500e+01f, 2.0375e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 1.6625e+01f,
                1.9250e+01f, 1.5125e+01f, 2.3875e+01f, 2.1875e+01f, 1.9875e+01f, 1.7875e+01f, 1.8000e+01f, 1.4750e+01f, 1.7000e+01f, 2.0750e+01f, 1.8750e+01f, 2.2500e+01f, 2.0500e+01f, 1.8500e+01f,
                2.0125e+01f, 1.8125e+01f, 1.6125e+01f, 1.9125e+01f, 1.5000e+01f, 2.3750e+01f, 2.1750e+01f, 2.1500e+01f, 1.9500e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.9375e+01f, 1.9500e+01f,
                2.2875e+01f, 2.0875e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.7125e+01f, 1.5125e+01f, 1.8500e+01f, 1.8625e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.8500e+01f, 1.6500e+01f,
                1.6250e+01f, 1.4250e+01f, 2.3000e+01f, 1.8875e+01f, 2.1875e+01f, 1.9875e+01f, 1.7875e+01f, 1.7625e+01f, 1.7750e+01f, 1.6500e+01f, 1.6625e+01f, 2.3250e+01f, 2.3375e+01f, 1.9250e+01f,
                2.0250e+01f, 1.8250e+01f, 1.6250e+01f, 1.4250e+01f, 2.2250e+01f, 1.6000e+01f, 2.4000e+01f, 2.1375e+01f, 2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.7625e+01f, 1.8500e+01f, 1.6500e+01f,
                2.2000e+01f, 2.0000e+01f, 2.0875e+01f, 1.8875e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f, 1.7625e+01f, 1.7750e+01f, 2.2250e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.6375e+01f,
                1.8250e+01f, 1.4125e+01f, 2.0000e+01f, 1.8000e+01f, 2.3875e+01f, 1.9750e+01f, 1.9875e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.5750e+01f, 2.0250e+01f, 2.0375e+01f, 2.1250e+01f,
                2.1000e+01f, 1.9000e+01f, 1.9125e+01f, 1.7125e+01f, 1.8000e+01f, 1.6000e+01f, 2.4750e+01f, 2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.9750e+01f, 1.3500e+01f, 1.5750e+01f, 1.9500e+01f,
                2.1125e+01f, 1.7000e+01f, 2.2875e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.7000e+01f, 1.4625e+01f, 1.4750e+01f, 2.1375e+01f, 2.1500e+01f, 2.0250e+01f, 2.0375e+01f, 1.8375e+01f,
                1.8125e+01f, 1.6125e+01f, 1.9125e+01f, 1.5000e+01f, 2.3750e+01f, 2.1750e+01f, 2.1875e+01f, 1.9500e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.9375e+01f, 1.9500e+01f, 2.1125e+01f,
                2.0875e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.7125e+01f, 1.5125e+01f, 2.1750e+01f, 1.8625e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.8500e+01f, 1.6500e+01f, 1.7375e+01f,
                1.5500e+01f, 2.5000e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f, 9.5000e+00f, 2.3250e+01f, 1.7000e+01f, 1.9250e+01f, 2.8750e+01f, 1.5250e+01f, 2.4750e+01f,
                1.3000e+01f, 2.2500e+01f, 9.0000e+00f, 1.8500e+01f, 2.6500e+01f, 1.4500e+01f, 2.8250e+01f, 1.6500e+01f, 2.1750e+01f, 1.2500e+01f, 2.2000e+01f, 1.4250e+01f, 1.8000e+01f, 2.7500e+01f,
                1.5750e+01f, 2.5250e+01f, 1.1750e+01f, 2.5500e+01f, 7.7500e+00f, 2.1500e+01f, 1.9500e+01f, 1.9250e+01f, 2.8750e+01f, 1.5250e+01f, 2.4750e+01f, 1.1250e+01f, 2.0750e+01f, 1.1500e+01f,
                1.8500e+01f, 2.0750e+01f, 1.8750e+01f, 2.8250e+01f, 1.4750e+01f, 2.4250e+01f, 1.0750e+01f, 1.9000e+01f, 5.5000e+00f, 2.3500e+01f, 2.4500e+01f, 1.9500e+01f, 2.9000e+01f, 1.5500e+01f,
                1.9875e+01f, 1.7875e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 2.2750e+01f, 2.0750e+01f, 2.1250e+01f, 2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.8375e+01f, 1.6375e+01f,
                2.1875e+01f, 1.9875e+01f, 2.2875e+01f, 1.8750e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f, 1.7500e+01f, 1.7625e+01f, 2.2125e+01f, 2.2250e+01f, 2.0250e+01f, 2.0375e+01f, 1.6250e+01f,
                1.8125e+01f, 1.6125e+01f, 1.9875e+01f, 1.7875e+01f, 2.3750e+01f, 1.9625e+01f, 1.9750e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.5625e+01f, 2.2250e+01f, 2.0250e+01f, 2.1125e+01f,
                2.0000e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 2.2000e+01f, 1.5750e+01f, 2.3750e+01f, 2.3250e+01f, 2.3375e+01f, 1.9250e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.3375e+01f,
                2.1000e+01f, 1.6875e+01f, 2.2750e+01f, 2.0750e+01f, 2.0875e+01f, 1.6750e+01f, 1.6875e+01f, 1.6625e+01f, 1.4625e+01f, 2.1250e+01f, 2.1375e+01f, 2.0125e+01f, 2.0250e+01f, 1.8250e+01f,
                1.8000e+01f, 1.6000e+01f, 1.9000e+01f, 1.7000e+01f, 2.3625e+01f, 2.1625e+01f, 2.1750e+01f, 1.9375e+01f, 1.7375e+01f, 1.7500e+01f, 1.5500e+01f, 1.9250e+01f, 1.9375e+01f, 2.3125e+01f,
                2.0750e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f, 1.5000e+01f, 2.1625e+01f, 1.9750e+01f, 2.3500e+01f, 2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.9250e+01f,
                1.8000e+01f, 1.6000e+01f, 2.4750e+01f, 2.0625e+01f, 2.0750e+01f, 1.8750e+01f, 1.8875e+01f, 1.6500e+01f, 1.6625e+01f, 2.0375e+01f, 1.8375e+01f, 2.2125e+01f, 2.2250e+01f, 1.8125e+01f,
                1.7875e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 2.2750e+01f, 2.0750e+01f, 2.1625e+01f, 2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.8375e+01f, 1.6375e+01f, 2.3000e+01f,
                1.9875e+01f, 2.2875e+01f, 1.8750e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f, 1.8625e+01f, 1.7625e+01f, 2.2125e+01f, 2.2250e+01f, 2.0250e+01f, 2.0375e+01f, 1.6250e+01f, 1.6375e+01f,
                1.5250e+01f, 1.9000e+01f, 1.7000e+01f, 2.5000e+01f, 1.8750e+01f, 2.1000e+01f, 1.9000e+01f, 1.8500e+01f, 1.6500e+01f, 1.7375e+01f, 1.7500e+01f, 2.4125e+01f, 2.2125e+01f, 2.0125e+01f,
                1.9875e+01f, 1.7875e+01f, 1.5875e+01f, 1.6000e+01f, 1.9750e+01f, 1.7750e+01f, 2.3625e+01f, 2.3375e+01f, 1.9250e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.3375e+01f, 2.2125e+01f,
                1.6875e+01f, 2.2750e+01f, 2.0750e+01f, 2.0875e+01f, 1.6750e+01f, 1.6875e+01f, 1.7750e+01f, 1.4625e+01f, 2.1250e+01f, 2.1375e+01f, 2.0125e+01f, 2.0250e+01f, 1.8250e+01f, 1.8375e+01f,
                1.6000e+01f, 1.9000e+01f, 1.7000e+01f, 2.3625e+01f, 2.1625e+01f, 2.1750e+01f, 1.7625e+01f, 1.6500e+01f, 1.4500e+01f, 1.6750e+01f, 1.6250e+01f, 1.8500e+01f, 2.2250e+01f, 2.0250e+01f,
                1.4750e+01f, 2.4250e+01f, 1.0750e+01f, 2.0250e+01f, 1.2500e+01f, 2.0500e+01f, 2.5750e+01f, 1.8250e+01f, 2.7750e+01f, 1.4250e+01f, 2.3750e+01f, 1.0250e+01f, 1.9750e+01f, 1.7750e+01f,
                1.7500e+01f, 3.1250e+01f, 1.3500e+01f, 2.7250e+01f, 1.3750e+01f, 2.3250e+01f, 9.7500e+00f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 2.6500e+01f, 1.7250e+01f, 2.2500e+01f, 1.3250e+01f,
                2.4500e+01f, 6.7500e+00f, 2.0500e+01f, 2.4250e+01f, 1.6500e+01f, 2.6000e+01f, 1.2500e+01f, 2.3750e+01f, 1.0250e+01f, 1.9750e+01f, 1.6250e+01f, 1.5750e+01f, 2.9500e+01f, 1.6000e+01f,
                2.8500e+01f, 1.5000e+01f, 2.4500e+01f, 1.1000e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 2.2250e+01f, 2.0250e+01f, 2.1125e+01f, 2.1250e+01f, 1.7125e+01f, 1.7250e+01f, 1.5250e+01f,
                1.7875e+01f, 1.5875e+01f, 2.4625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.8750e+01f, 1.6375e+01f, 1.4375e+01f, 2.0250e+01f, 1.8250e+01f, 2.2000e+01f, 2.2125e+01f, 2.0125e+01f,
                1.7750e+01f, 1.7875e+01f, 1.5875e+01f, 1.3875e+01f, 2.0500e+01f, 2.0625e+01f, 2.1500e+01f, 2.1250e+01f, 1.9250e+01f, 1.9375e+01f, 1.5250e+01f, 1.8250e+01f, 1.6250e+01f, 2.2875e+01f,
                1.9750e+01f, 2.2750e+01f, 1.8625e+01f, 1.8750e+01f, 1.6750e+01f, 1.6875e+01f, 1.8500e+01f, 1.9500e+01f, 2.3250e+01f, 2.1250e+01f, 1.9250e+01f, 2.1500e+01f, 1.5250e+01f, 1.7500e+01f,
                1.4875e+01f, 1.5000e+01f, 2.1625e+01f, 1.9625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.8375e+01f, 1.6375e+01f, 1.7250e+01f, 1.5250e+01f, 2.4000e+01f, 2.2000e+01f, 2.0000e+01f,
                1.9750e+01f, 1.9875e+01f, 1.5750e+01f, 1.5875e+01f, 1.9625e+01f, 1.7625e+01f, 2.1375e+01f, 2.3250e+01f, 1.9125e+01f, 1.9250e+01f, 1.7250e+01f, 1.7375e+01f, 1.3250e+01f, 2.2000e+01f,
                1.8875e+01f, 2.2625e+01f, 2.0625e+01f, 2.0750e+01f, 1.6625e+01f, 1.6750e+01f, 1.7625e+01f, 1.4500e+01f, 2.1125e+01f, 2.1250e+01f, 2.2125e+01f, 2.0125e+01f, 1.8125e+01f, 1.8250e+01f,
                1.5000e+01f, 1.8750e+01f, 1.6750e+01f, 2.4750e+01f, 1.8500e+01f, 2.0750e+01f, 1.8750e+01f, 2.0375e+01f, 1.6250e+01f, 1.6375e+01f, 1.4375e+01f, 2.1000e+01f, 1.9000e+01f, 2.2000e+01f,
                1.9625e+01f, 1.9750e+01f, 1.7750e+01f, 1.7875e+01f, 1.6625e+01f, 1.6750e+01f, 2.3375e+01f, 2.0250e+01f, 2.1125e+01f, 2.1250e+01f, 1.7125e+01f, 1.7250e+01f, 1.5250e+01f, 2.1125e+01f,
                1.5875e+01f, 2.4625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.8750e+01f, 1.4625e+01f, 1.4375e+01f, 2.0250e+01f, 1.8250e+01f, 2.2000e+01f, 2.2125e+01f, 2.0125e+01f, 1.8125e+01f,
                1.7875e+01f, 1.5875e+01f, 1.3875e+01f, 2.0500e+01f, 2.0625e+01f, 2.1500e+01f, 1.9500e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 2.0250e+01f, 1.8250e+01f, 2.2000e+01f, 2.0000e+01f,
                2.1625e+01f, 2.1750e+01f, 1.7625e+01f, 1.7750e+01f, 1.5750e+01f, 1.5875e+01f, 2.0375e+01f, 1.9375e+01f, 2.3125e+01f, 2.1125e+01f, 1.9125e+01f, 1.9250e+01f, 1.5125e+01f, 1.8125e+01f,
                1.5000e+01f, 2.1625e+01f, 1.9625e+01f, 2.2625e+01f, 2.0625e+01f, 1.8625e+01f, 1.6625e+01f, 1.6375e+01f, 1.7250e+01f, 1.5250e+01f, 2.4000e+01f, 2.2000e+01f, 2.0000e+01f, 2.0125e+01f,
                1.9875e+01f, 1.5750e+01f, 1.5875e+01f, 1.9625e+01f, 1.7625e+01f, 2.1375e+01f, 2.1500e+01f, 1.9125e+01f, 1.9250e+01f, 1.7250e+01f, 1.7375e+01f, 1.3250e+01f, 2.2000e+01f, 2.0000e+01f,
                2.1750e+01f, 1.9750e+01f, 2.2000e+01f, 1.5750e+01f, 1.8000e+01f, 2.1750e+01f, 1.9750e+01f, 1.5750e+01f, 2.9500e+01f, 1.6000e+01f, 2.5500e+01f, 1.2000e+01f, 2.1500e+01f, 1.2250e+01f,
                1.9250e+01f, 1.7250e+01f, 1.9500e+01f, 2.4750e+01f, 1.5500e+01f, 2.5000e+01f, 1.1500e+01f, 2.2750e+01f, 9.2500e+00f, 1.8750e+01f, 2.2500e+01f, 1.4750e+01f, 2.8500e+01f, 1.0750e+01f,
                2.6250e+01f, 1.2750e+01f, 2.2250e+01f, 1.4500e+01f, 1.8250e+01f, 2.7750e+01f, 1.4250e+01f, 2.5500e+01f, 1.6250e+01f, 2.1500e+01f, 1.2250e+01f, 2.1750e+01f, 1.9750e+01f, 1.7750e+01f,
                2.9000e+01f, 1.5500e+01f, 2.5000e+01f, 1.1500e+01f, 2.5250e+01f, 7.5000e+00f, 2.1250e+01f, 1.8000e+01f, 1.6000e+01f, 2.5500e+01f, 1.2000e+01f, 2.1500e+01f, 8.0000e+00f, 1.7500e+01f,
                1.6750e+01f, 1.6875e+01f, 1.8500e+01f, 1.8625e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 2.0250e+01f, 1.6125e+01f, 1.6250e+01f, 1.4250e+01f, 2.3000e+01f, 1.8875e+01f, 2.1875e+01f,
                2.1625e+01f, 1.9625e+01f, 1.7625e+01f, 1.7750e+01f, 1.6500e+01f, 1.6625e+01f, 2.3250e+01f, 2.0125e+01f, 2.1000e+01f, 2.1125e+01f, 1.9125e+01f, 1.7125e+01f, 1.5125e+01f, 2.1000e+01f,
                1.5750e+01f, 2.2375e+01f, 2.2500e+01f, 2.0500e+01f, 1.8500e+01f, 1.8625e+01f, 1.6625e+01f, 1.4250e+01f, 2.0125e+01f, 1.8125e+01f, 2.1875e+01f, 1.9875e+01f, 2.0000e+01f, 1.8000e+01f,
                1.9000e+01f, 1.7000e+01f, 1.5000e+01f, 2.4500e+01f, 2.2500e+01f, 2.0500e+01f, 1.8500e+01f, 2.0125e+01f, 1.8125e+01f, 1.8250e+01f, 1.4125e+01f, 2.0000e+01f, 1.8000e+01f, 2.3875e+01f,
                2.1500e+01f, 2.1625e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.5750e+01f, 2.0250e+01f, 1.7125e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.9125e+01f, 1.7125e+01f, 1.8000e+01f,
                1.4875e+01f, 2.1500e+01f, 1.9500e+01f, 2.0375e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 1.6250e+01f, 1.9250e+01f, 1.5125e+01f, 2.3875e+01f, 2.1875e+01f, 1.9875e+01f, 1.7875e+01f,
                1.9750e+01f, 1.5625e+01f, 1.5750e+01f, 1.9500e+01f, 1.9625e+01f, 2.1250e+01f, 2.1375e+01f, 2.0250e+01f, 1.8250e+01f, 1.6250e+01f, 1.8500e+01f, 1.2250e+01f, 2.6000e+01f, 2.4000e+01f,
                2.0625e+01f, 2.1500e+01f, 1.9500e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.9375e+01f, 1.6250e+01f, 2.2875e+01f, 2.0875e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.7125e+01f,
                1.6875e+01f, 1.8500e+01f, 1.8625e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.8500e+01f, 1.6125e+01f, 1.6250e+01f, 1.4250e+01f, 2.3000e+01f, 1.8875e+01f, 2.1875e+01f, 1.9875e+01f,
                1.9625e+01f, 1.7625e+01f, 1.7750e+01f, 1.6500e+01f, 1.6625e+01f, 2.3250e+01f, 2.3375e+01f, 2.1000e+01f, 2.1125e+01f, 1.9125e+01f, 1.7125e+01f, 1.5125e+01f, 2.1000e+01f, 1.6875e+01f,
                2.1500e+01f, 1.9500e+01f, 2.1750e+01f, 1.5500e+01f, 1.7750e+01f, 1.5750e+01f, 1.3750e+01f, 1.3250e+01f, 2.2000e+01f, 2.0000e+01f, 2.0875e+01f, 1.8875e+01f, 1.9000e+01f, 1.7000e+01f,
                1.6750e+01f, 1.7625e+01f, 1.7750e+01f, 2.2250e+01f, 2.2375e+01f, 2.0375e+01f, 1.8375e+01f, 1.8125e+01f, 1.8250e+01f, 1.4125e+01f, 2.0000e+01f, 1.8000e+01f, 2.3875e+01f, 1.9750e+01f,
                2.1625e+01f, 1.9625e+01f, 1.7625e+01f, 1.5625e+01f, 1.5750e+01f, 2.0250e+01f, 2.0375e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.9125e+01f, 1.7125e+01f, 1.8000e+01f, 1.6000e+01f,
                2.1500e+01f, 1.9500e+01f, 2.0375e+01f, 2.0500e+01f, 1.8500e+01f, 1.6500e+01f, 1.6625e+01f, 2.1250e+01f, 1.9250e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f, 1.7000e+01f, 1.5000e+01f,
                2.5250e+01f, 7.5000e+00f, 2.1250e+01f, 2.5000e+01f, 1.7250e+01f, 2.6750e+01f, 1.3250e+01f, 2.4500e+01f, 1.1000e+01f, 2.0500e+01f, 1.7000e+01f, 1.6500e+01f, 3.0250e+01f, 1.6750e+01f,
                2.3750e+01f, 1.4500e+01f, 2.4000e+01f, 1.0500e+01f, 2.0000e+01f, 1.8000e+01f, 2.0250e+01f, 2.7250e+01f, 1.3750e+01f, 2.7500e+01f, 9.7500e+00f, 2.3500e+01f, 1.0000e+01f, 1.9500e+01f,
                1.9250e+01f, 1.7250e+01f, 2.6750e+01f, 1.3250e+01f, 2.2750e+01f, 1.3500e+01f, 1.8750e+01f, 1.1250e+01f, 2.0750e+01f, 2.4500e+01f, 1.6750e+01f, 2.6250e+01f, 1.2750e+01f, 2.2250e+01f,
                7.5000e+00f, 2.5500e+01f, 1.5000e+01f, 2.1500e+01f, 3.1000e+01f, 1.7500e+01f, 2.7000e+01f, 1.9875e+01f, 2.0000e+01f, 1.8000e+01f, 1.6000e+01f, 1.9000e+01f, 1.7000e+01f, 2.3625e+01f,
                2.0500e+01f, 2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.7500e+01f, 1.5500e+01f, 1.9250e+01f, 1.6125e+01f, 2.4875e+01f, 2.0750e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f,
                1.6750e+01f, 1.8375e+01f, 1.8500e+01f, 2.2250e+01f, 2.2375e+01f, 1.8250e+01f, 1.8375e+01f, 1.8125e+01f, 1.6125e+01f, 1.4125e+01f, 2.2875e+01f, 1.8750e+01f, 2.1750e+01f, 1.9750e+01f,
                1.9500e+01f, 1.7500e+01f, 1.7625e+01f, 1.8500e+01f, 1.6500e+01f, 2.3125e+01f, 2.3250e+01f, 2.0000e+01f, 1.8000e+01f, 1.6000e+01f, 1.8250e+01f, 1.2000e+01f, 2.0000e+01f, 1.8000e+01f,
                1.9625e+01f, 2.1250e+01f, 2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.8375e+01f, 1.3125e+01f, 2.1875e+01f, 1.9875e+01f, 2.2875e+01f, 1.8750e+01f, 1.8875e+01f, 1.6875e+01f,
                1.6625e+01f, 1.7500e+01f, 1.7625e+01f, 2.2125e+01f, 2.2250e+01f, 2.0250e+01f, 2.0375e+01f, 1.8000e+01f, 1.8125e+01f, 1.6125e+01f, 1.9875e+01f, 1.7875e+01f, 2.3750e+01f, 1.9625e+01f,
                1.9375e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.5625e+01f, 2.2250e+01f, 2.0250e+01f, 2.2875e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.7000e+01f, 1.7875e+01f, 1.5875e+01f,
                2.5500e+01f, 2.3500e+01f, 2.1500e+01f, 1.9500e+01f, 1.7500e+01f, 1.5500e+01f, 1.3500e+01f, 1.5125e+01f, 2.1000e+01f, 1.6875e+01f, 2.2750e+01f, 2.0750e+01f, 2.0875e+01f, 1.6750e+01f,
                1.8625e+01f, 1.6625e+01f, 1.4625e+01f, 2.1250e+01f, 2.1375e+01f, 2.0125e+01f, 2.0250e+01f, 2.0000e+01f, 1.8000e+01f, 1.6000e+01f, 1.9000e+01f, 1.7000e+01f, 2.3625e+01f, 2.1625e+01f,
                2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.7500e+01f, 1.5500e+01f, 1.9250e+01f, 1.9375e+01f, 2.4875e+01f, 2.0750e+01f, 2.0875e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f, 1.5000e+01f,
                1.8375e+01f, 1.8500e+01f, 2.2250e+01f, 2.2375e+01f, 1.8250e+01f, 1.8375e+01f, 1.6375e+01f, 1.5250e+01f, 1.3250e+01f, 2.7000e+01f, 2.0750e+01f, 2.3000e+01f, 2.1000e+01f, 1.9000e+01f,
                1.8500e+01f, 1.6500e+01f, 1.6625e+01f, 2.0375e+01f, 1.8375e+01f, 2.2125e+01f, 2.2250e+01f, 1.9875e+01f, 1.7875e+01f, 1.8000e+01f, 1.6000e+01f, 1.4000e+01f, 2.2750e+01f, 2.0750e+01f,
                2.1250e+01f, 2.1375e+01f, 1.9375e+01f, 1.7375e+01f, 1.5375e+01f, 1.8375e+01f, 1.6375e+01f, 2.1875e+01f, 1.9875e+01f, 2.2875e+01f, 1.8750e+01f, 1.8875e+01f, 1.6875e+01f, 1.4875e+01f,
                1.7500e+01f, 1.7625e+01f, 2.2125e+01f, 2.2250e+01f, 2.0250e+01f, 2.0375e+01f, 1.6250e+01f, 1.8125e+01f, 1.6125e+01f, 1.9875e+01f, 1.7875e+01f, 2.3750e+01f, 1.9625e+01f, 1.9750e+01f,
                1.6500e+01f, 1.8750e+01f, 1.2500e+01f, 1.4750e+01f, 2.4250e+01f, 2.2250e+01f, 2.0250e+01f, 2.6250e+01f, 1.2750e+01f, 2.2250e+01f, 8.7500e+00f, 2.2500e+01f, 1.6250e+01f, 1.8500e+01f,
                2.9750e+01f, 1.6250e+01f, 2.5750e+01f, 1.2250e+01f, 2.1750e+01f, 8.2500e+00f, 1.7750e+01f, 2.1750e+01f, 1.5500e+01f, 2.9250e+01f, 1.5750e+01f, 2.5250e+01f, 1.1750e+01f, 2.1250e+01f,
                9.5000e+00f, 1.9000e+01f, 2.2750e+01f, 1.9250e+01f, 2.4500e+01f, 1.5250e+01f, 2.4750e+01f, 8.7500e+00f, 2.2500e+01f, 1.4750e+01f, 1.8500e+01f, 2.8000e+01f, 1.4500e+01f, 2.8250e+01f,
                1.2250e+01f, 2.1750e+01f, 1.2500e+01f, 1.7750e+01f, 2.0000e+01f, 1.8000e+01f, 2.7500e+01f, 1.7000e+01f, 2.6500e+01f, 1.3000e+01f, 2.2500e+01f, 9.0000e+00f, 1.8500e+01f, 2.8000e+01f,
            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
