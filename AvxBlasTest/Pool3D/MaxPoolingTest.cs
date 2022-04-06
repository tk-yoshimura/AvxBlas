using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool3DTest {
    [TestClass]
    public class MaxPoolingTest {
        [TestMethod]
        public void SMaxPoolingTest() {
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

                        foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {

                            float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                            Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                            Map3D y = Reference(x, (int)sx, (int)kw, (int)sy, (int)kh, (int)sz, (int)kd);

                            Array<float> x_tensor = xval;
                            Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                            Pool3D.MaxPooling(n, c, iw, ih, id, sx, sy, sz, kw, kh, kd, x_tensor, y_tensor);

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
                                double v = double.MinValue;

                                for (int kz = 0, iz = isz + kz; kz < kd && iz < id; kz++, iz = isz + kz) {
                                    for (int ky = 0, iy = isy + ky; ky < kh && iy < ih; ky++, iy = isy + ky) {
                                        for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {

                                            v = Math.Max(v, x[c, ix, iy, iz, th]);
                                        }
                                    }
                                }

                                y[c, ox, oy, oz, th] = v;
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
                2.70e+01f, 2.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f,
                3.20e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.30e+01f, 3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f,
                3.00e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 2.70e+01f, 2.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f,
                3.00e+01f, 2.60e+01f, 2.70e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.80e+01f, 3.40e+01f, 3.40e+01f,
                3.40e+01f, 2.70e+01f, 3.00e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.50e+01f, 3.70e+01f, 3.10e+01f, 3.30e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.80e+01f,
                2.60e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.60e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 3.20e+01f,
                2.90e+01f, 2.80e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f, 3.60e+01f, 2.70e+01f, 3.20e+01f, 2.30e+01f, 3.20e+01f, 3.70e+01f, 3.70e+01f, 3.30e+01f,
                3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 2.70e+01f, 3.10e+01f, 3.30e+01f, 3.30e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.10e+01f, 3.40e+01f,
                2.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 2.80e+01f,
                3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f, 3.50e+01f,
                3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 2.40e+01f, 2.50e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 3.30e+01f, 2.70e+01f, 2.90e+01f,
                2.90e+01f, 2.70e+01f, 3.20e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.80e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f,
                2.70e+01f, 3.00e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.30e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.80e+01f, 3.70e+01f,
                3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.20e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f,
                2.80e+01f, 2.30e+01f, 2.70e+01f, 3.40e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f, 3.70e+01f, 2.40e+01f, 3.60e+01f, 2.00e+01f,
                2.90e+01f, 2.20e+01f, 2.50e+01f, 1.80e+01f, 2.70e+01f, 3.10e+01f, 2.30e+01f, 3.30e+01f, 1.90e+01f, 2.90e+01f, 1.50e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f,
                3.60e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 2.80e+01f, 1.50e+01f, 2.40e+01f, 3.10e+01f, 2.40e+01f, 3.60e+01f, 2.20e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f,
                1.20e+01f, 2.10e+01f, 2.20e+01f, 2.60e+01f, 2.70e+01f, 2.20e+01f, 3.10e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 3.00e+01f, 3.30e+01f, 2.70e+01f, 3.50e+01f,
                3.80e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.30e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f,
                2.60e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f, 3.00e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f,
                2.70e+01f, 3.00e+01f, 2.30e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 2.30e+01f, 2.30e+01f, 3.20e+01f, 2.90e+01f,
                3.40e+01f, 3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 3.20e+01f, 3.70e+01f, 3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 2.50e+01f,
                2.80e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f,
                2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 3.10e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.10e+01f, 2.50e+01f, 3.40e+01f, 2.80e+01f,
                2.80e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 2.40e+01f, 2.70e+01f, 3.60e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f,
                2.40e+01f, 2.70e+01f, 3.00e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.00e+01f, 3.00e+01f, 3.30e+01f, 2.70e+01f, 3.50e+01f, 3.40e+01f,
                3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f, 2.40e+01f,
                3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 3.30e+01f, 3.80e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f,
                2.60e+01f, 2.90e+01f, 2.20e+01f, 3.20e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 3.20e+01f, 2.60e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 3.40e+01f,
                3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 3.20e+01f, 3.70e+01f, 3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 2.50e+01f, 2.50e+01f,
                2.90e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 3.20e+01f,
                2.60e+01f, 2.50e+01f, 2.60e+01f, 3.10e+01f, 3.10e+01f, 3.60e+01f, 2.70e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f, 1.40e+01f, 2.40e+01f, 3.30e+01f, 2.60e+01f,
                3.30e+01f, 2.20e+01f, 3.10e+01f, 1.80e+01f, 2.70e+01f, 2.80e+01f, 2.30e+01f, 3.60e+01f, 2.30e+01f, 3.50e+01f, 1.90e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f,
                2.50e+01f, 2.60e+01f, 3.60e+01f, 2.20e+01f, 3.20e+01f, 2.10e+01f, 2.80e+01f, 1.40e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 3.50e+01f, 2.20e+01f, 3.10e+01f,
                1.80e+01f, 2.70e+01f, 1.40e+01f, 2.30e+01f, 3.30e+01f, 1.90e+01f, 3.50e+01f, 2.10e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f, 3.60e+01f, 2.30e+01f, 3.20e+01f,
                3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f,
                2.60e+01f, 2.90e+01f, 3.20e+01f, 3.80e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.40e+01f,
                3.10e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 3.50e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.80e+01f, 2.60e+01f,
                3.20e+01f, 3.70e+01f, 2.80e+01f, 3.30e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f, 2.50e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 3.30e+01f, 3.20e+01f, 3.10e+01f,
                2.80e+01f, 3.10e+01f, 2.40e+01f, 3.40e+01f, 3.60e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 2.50e+01f, 3.30e+01f, 3.60e+01f,
                3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.70e+01f, 3.40e+01f, 3.30e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f,
                3.10e+01f, 3.60e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 2.40e+01f, 2.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 2.50e+01f,
                3.00e+01f, 3.30e+01f, 2.60e+01f, 3.30e+01f, 3.80e+01f, 3.20e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 3.20e+01f,
                3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.90e+01f,
                2.90e+01f, 3.20e+01f, 3.80e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f,
                3.00e+01f, 2.90e+01f, 2.60e+01f, 2.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 3.30e+01f, 3.20e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 3.10e+01f,
                3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 3.30e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f,
                3.10e+01f, 2.40e+01f, 3.40e+01f, 3.60e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 2.50e+01f, 3.30e+01f, 3.60e+01f, 2.90e+01f,
                3.20e+01f, 2.90e+01f, 2.80e+01f, 2.70e+01f, 3.40e+01f, 3.30e+01f, 3.30e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f,
                3.30e+01f, 1.90e+01f, 3.50e+01f, 2.10e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f, 3.60e+01f, 2.30e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 2.10e+01f, 3.00e+01f,
                1.70e+01f, 2.60e+01f, 3.30e+01f, 2.20e+01f, 3.20e+01f, 1.80e+01f, 3.40e+01f, 1.80e+01f, 3.00e+01f, 1.60e+01f, 2.60e+01f, 3.50e+01f, 2.20e+01f, 3.10e+01f,
                2.10e+01f, 3.10e+01f, 2.00e+01f, 2.70e+01f, 3.00e+01f, 2.50e+01f, 3.50e+01f, 2.50e+01f, 3.40e+01f, 2.10e+01f, 3.00e+01f, 1.70e+01f, 2.90e+01f, 2.70e+01f,
                2.20e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f, 1.40e+01f, 2.40e+01f, 1.00e+01f, 2.90e+01f, 3.80e+01f, 2.60e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f,
                3.00e+01f, 2.90e+01f, 2.60e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f,
                3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.20e+01f, 3.70e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.80e+01f,
                3.10e+01f, 3.40e+01f, 3.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 2.50e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f,
                3.20e+01f, 3.10e+01f, 2.80e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 2.90e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f,
                3.30e+01f, 3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f,
                2.70e+01f, 2.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 3.40e+01f, 2.80e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.70e+01f, 3.50e+01f, 3.80e+01f, 3.10e+01f,
                3.40e+01f, 2.50e+01f, 3.00e+01f, 2.10e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f,
                3.50e+01f, 3.70e+01f, 3.10e+01f, 3.30e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.80e+01f, 2.60e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f,
                2.90e+01f, 2.60e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f,
                3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.20e+01f, 3.70e+01f, 3.70e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.60e+01f, 2.30e+01f, 2.20e+01f, 2.70e+01f,
                3.40e+01f, 3.30e+01f, 3.30e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f,
                3.10e+01f, 2.80e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.30e+01f,
                3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 2.70e+01f,
                2.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 2.40e+01f, 1.70e+01f, 2.90e+01f, 2.70e+01f, 2.50e+01f, 3.40e+01f, 2.10e+01f, 3.00e+01f,
                2.00e+01f, 3.00e+01f, 1.60e+01f, 2.60e+01f, 3.80e+01f, 2.20e+01f, 3.40e+01f, 2.40e+01f, 2.70e+01f, 2.00e+01f, 2.90e+01f, 2.40e+01f, 2.50e+01f, 3.50e+01f,
                2.10e+01f, 3.10e+01f, 1.70e+01f, 3.30e+01f, 1.30e+01f, 2.90e+01f, 3.80e+01f, 2.50e+01f, 3.40e+01f, 2.10e+01f, 3.00e+01f, 1.70e+01f, 2.60e+01f, 1.90e+01f,
                2.60e+01f, 3.50e+01f, 2.40e+01f, 3.40e+01f, 2.00e+01f, 3.00e+01f, 1.60e+01f, 2.30e+01f, 1.00e+01f, 2.80e+01f, 2.90e+01f, 2.40e+01f, 3.30e+01f, 2.00e+01f,
                3.00e+01f, 3.20e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 3.10e+01f,
                3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.10e+01f, 2.50e+01f, 2.70e+01f, 2.80e+01f, 3.40e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f,
                3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 2.70e+01f, 3.10e+01f, 3.60e+01f, 3.30e+01f,
                2.90e+01f, 2.90e+01f, 2.50e+01f, 2.50e+01f, 3.40e+01f, 2.20e+01f, 3.00e+01f, 3.50e+01f, 3.80e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f,
                3.60e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 2.90e+01f, 2.60e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f,
                3.30e+01f, 3.00e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f,
                3.10e+01f, 3.10e+01f, 3.30e+01f, 2.70e+01f, 2.90e+01f, 2.90e+01f, 3.20e+01f, 2.60e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.60e+01f, 2.70e+01f,
                3.20e+01f, 3.70e+01f, 3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f,
                3.20e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 3.10e+01f, 3.70e+01f,
                3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.10e+01f, 2.50e+01f, 3.40e+01f, 2.80e+01f, 3.40e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f,
                2.20e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 2.70e+01f, 3.00e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f,
                3.40e+01f, 2.80e+01f, 3.00e+01f, 3.00e+01f, 3.30e+01f, 2.70e+01f, 3.50e+01f, 3.80e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.30e+01f,
                3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f, 2.60e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f,
                3.00e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 2.70e+01f, 2.40e+01f, 2.30e+01f, 2.40e+01f, 2.90e+01f, 2.90e+01f, 3.40e+01f,
                2.00e+01f, 3.00e+01f, 1.60e+01f, 2.60e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 3.30e+01f, 2.00e+01f, 2.90e+01f, 1.60e+01f, 2.50e+01f, 3.20e+01f,
                2.50e+01f, 3.70e+01f, 2.10e+01f, 3.30e+01f, 1.90e+01f, 2.90e+01f, 1.50e+01f, 2.80e+01f, 2.90e+01f, 2.40e+01f, 3.40e+01f, 2.30e+01f, 3.00e+01f, 1.90e+01f,
                3.20e+01f, 1.20e+01f, 2.80e+01f, 3.70e+01f, 2.40e+01f, 3.30e+01f, 2.00e+01f, 2.90e+01f, 1.60e+01f, 2.50e+01f, 2.60e+01f, 2.10e+01f, 3.70e+01f, 2.30e+01f,
                3.30e+01f, 1.90e+01f, 2.90e+01f, 1.50e+01f, 2.50e+01f, 3.40e+01f, 2.10e+01f, 3.10e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.70e+01f,
                3.40e+01f, 2.70e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.10e+01f, 3.10e+01f, 2.40e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f,
                2.80e+01f, 2.80e+01f, 3.00e+01f, 2.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 3.30e+01f, 3.30e+01f,
                3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.60e+01f, 3.00e+01f, 3.00e+01f, 3.50e+01f, 2.60e+01f, 3.10e+01f, 2.20e+01f, 2.70e+01f,
                2.90e+01f, 2.90e+01f, 3.20e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 3.20e+01f, 3.80e+01f, 3.70e+01f, 3.40e+01f,
                3.00e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 3.50e+01f,
                3.70e+01f, 3.40e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.80e+01f, 2.20e+01f, 3.20e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f,
                2.60e+01f, 3.50e+01f, 2.30e+01f, 3.10e+01f, 2.80e+01f, 2.70e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 3.40e+01f, 3.60e+01f, 3.60e+01f,
                3.20e+01f, 3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 2.50e+01f, 3.30e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.70e+01f, 3.40e+01f,
                2.70e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f, 2.40e+01f, 3.10e+01f, 3.60e+01f, 3.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f,
                2.80e+01f, 3.00e+01f, 2.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f,
                3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.00e+01f,
                2.90e+01f, 3.20e+01f, 2.90e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 2.90e+01f, 3.20e+01f, 3.80e+01f, 3.70e+01f, 3.40e+01f, 3.30e+01f,
                3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 3.50e+01f, 3.70e+01f,
                2.80e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.80e+01f, 2.60e+01f, 2.10e+01f, 3.70e+01f, 2.30e+01f, 3.30e+01f, 1.90e+01f, 2.90e+01f, 1.80e+01f,
                2.50e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 2.80e+01f, 1.50e+01f, 2.40e+01f, 3.40e+01f, 2.00e+01f, 3.60e+01f, 1.60e+01f,
                3.20e+01f, 1.80e+01f, 2.80e+01f, 3.70e+01f, 2.40e+01f, 3.30e+01f, 2.00e+01f, 3.30e+01f, 2.20e+01f, 2.90e+01f, 1.80e+01f, 2.70e+01f, 3.40e+01f, 2.30e+01f,
                3.60e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 3.10e+01f, 1.50e+01f, 2.70e+01f, 2.50e+01f, 2.00e+01f, 3.00e+01f, 1.60e+01f, 2.60e+01f, 1.20e+01f, 2.20e+01f,
                2.70e+01f, 2.70e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 2.70e+01f, 2.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f,
                3.40e+01f, 3.10e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.70e+01f, 3.50e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f,
                2.40e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.30e+01f, 3.80e+01f, 3.20e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f,
                3.00e+01f, 2.60e+01f, 2.60e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f,
                3.40e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.80e+01f, 2.90e+01f, 3.70e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.20e+01f,
                2.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.80e+01f, 3.10e+01f, 3.40e+01f, 3.40e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f,
                3.20e+01f, 2.80e+01f, 2.80e+01f, 3.70e+01f, 3.10e+01f, 3.30e+01f, 3.60e+01f, 2.70e+01f, 3.20e+01f, 2.30e+01f, 2.80e+01f, 1.90e+01f, 3.70e+01f, 3.30e+01f,
                3.00e+01f, 2.90e+01f, 3.20e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 2.80e+01f, 3.30e+01f, 3.30e+01f, 3.50e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f,
                2.70e+01f, 3.60e+01f, 3.00e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 2.70e+01f, 2.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f, 3.40e+01f,
                3.10e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.70e+01f, 3.50e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 3.00e+01f, 2.70e+01f, 3.60e+01f, 3.50e+01f,
                3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 2.40e+01f, 2.90e+01f, 2.00e+01f, 2.50e+01f, 3.50e+01f, 3.70e+01f, 3.10e+01f, 3.30e+01f, 3.30e+01f, 3.20e+01f,
                2.90e+01f, 3.80e+01f, 2.60e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.50e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f,
                3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.70e+01f, 3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 3.20e+01f, 3.70e+01f,
                3.40e+01f, 3.10e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f, 2.60e+01f, 2.60e+01f,
                3.10e+01f, 1.50e+01f, 2.70e+01f, 3.60e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f, 3.10e+01f, 2.40e+01f, 3.60e+01f, 2.20e+01f,
                2.90e+01f, 2.20e+01f, 3.10e+01f, 1.80e+01f, 2.70e+01f, 2.80e+01f, 2.60e+01f, 3.30e+01f, 1.90e+01f, 3.50e+01f, 1.50e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f,
                3.60e+01f, 2.30e+01f, 3.20e+01f, 1.90e+01f, 2.80e+01f, 2.10e+01f, 2.40e+01f, 1.70e+01f, 2.60e+01f, 3.60e+01f, 2.20e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f,
                1.20e+01f, 3.00e+01f, 2.20e+01f, 2.60e+01f, 3.50e+01f, 2.20e+01f, 3.10e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f,
                3.20e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 3.30e+01f, 2.70e+01f, 2.90e+01f,
                2.90e+01f, 2.70e+01f, 3.20e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 3.00e+01f, 2.90e+01f, 2.60e+01f, 3.80e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f,
                2.70e+01f, 3.00e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.50e+01f, 3.40e+01f, 3.10e+01f, 2.70e+01f, 2.70e+01f, 3.20e+01f, 2.30e+01f, 3.20e+01f, 3.70e+01f,
                3.10e+01f, 3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 2.50e+01f, 3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.10e+01f,
                2.80e+01f, 2.70e+01f, 2.80e+01f, 3.40e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f,
                2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 2.70e+01f, 3.10e+01f, 3.60e+01f, 3.30e+01f, 3.50e+01f, 2.90e+01f, 3.10e+01f, 3.10e+01f, 3.40e+01f, 2.70e+01f,
                3.60e+01f, 3.30e+01f, 3.20e+01f, 2.90e+01f, 2.80e+01f, 2.50e+01f, 2.40e+01f, 2.70e+01f, 3.60e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 2.70e+01f,
                3.00e+01f, 2.90e+01f, 2.60e+01f, 3.00e+01f, 3.50e+01f, 3.20e+01f, 3.10e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 3.30e+01f, 3.80e+01f, 3.50e+01f, 3.40e+01f,
                3.10e+01f, 3.40e+01f, 2.70e+01f, 3.00e+01f, 2.30e+01f, 3.00e+01f, 3.50e+01f, 3.50e+01f, 3.10e+01f, 3.10e+01f, 3.30e+01f, 2.70e+01f, 2.90e+01f, 2.90e+01f,
                2.70e+01f, 3.20e+01f, 3.40e+01f, 3.70e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 2.90e+01f, 2.00e+01f, 3.80e+01f, 3.40e+01f, 3.40e+01f, 3.00e+01f, 3.00e+01f,
                2.60e+01f, 2.90e+01f, 2.80e+01f, 2.90e+01f, 3.40e+01f, 3.40e+01f, 3.30e+01f, 3.00e+01f, 3.20e+01f, 3.20e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f,
                3.30e+01f, 3.60e+01f, 2.90e+01f, 3.20e+01f, 2.50e+01f, 2.80e+01f, 3.10e+01f, 3.70e+01f, 3.30e+01f, 3.30e+01f, 2.90e+01f, 2.90e+01f, 3.10e+01f, 2.50e+01f,
                2.70e+01f, 2.80e+01f, 3.40e+01f, 3.30e+01f, 3.20e+01f, 3.50e+01f, 2.80e+01f, 3.10e+01f, 2.80e+01f, 3.70e+01f, 3.60e+01f, 3.60e+01f, 3.20e+01f, 3.20e+01f,
                2.60e+01f, 2.50e+01f, 2.20e+01f, 2.10e+01f, 3.10e+01f, 3.60e+01f, 2.70e+01f, 3.20e+01f, 1.80e+01f, 2.80e+01f, 1.40e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f,
                3.50e+01f, 2.20e+01f, 3.10e+01f, 1.80e+01f, 2.70e+01f, 1.40e+01f, 2.30e+01f, 3.60e+01f, 2.30e+01f, 3.50e+01f, 2.10e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f,
                1.70e+01f, 2.60e+01f, 3.00e+01f, 2.50e+01f, 3.20e+01f, 2.10e+01f, 3.00e+01f, 1.40e+01f, 3.00e+01f, 3.30e+01f, 2.60e+01f, 3.50e+01f, 2.20e+01f, 3.40e+01f,
                1.80e+01f, 2.70e+01f, 2.00e+01f, 2.30e+01f, 3.00e+01f, 2.50e+01f, 3.50e+01f, 2.10e+01f, 3.10e+01f, 1.70e+01f, 2.70e+01f, 1.30e+01f, 2.30e+01f, 3.20e+01f,
            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
