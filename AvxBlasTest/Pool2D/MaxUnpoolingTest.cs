using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool2DTest {
    [TestClass]
    public class MaxUnpoolingTest {
        [TestMethod]
        public void SMaxUnpoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (6, 7), (7, 6), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    foreach ((uint sx, uint sy) in new (uint, uint)[] { (1, 1), (1, 2), (2, 1), (2, 3), (3, 3), (4, 2) }) {
                        uint ow = (iw - 1) / sx + 1;
                        uint oh = (ih - 1) / sy + 1;

                        foreach ((uint kw, uint kh) in new (uint, uint)[] { (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 3), (4, 4), (2, 5), (5, 3) }) {

                            foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {

                                float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17)).ToArray();
                                float[] dyval = (new float[c * ow * oh * n]).Select((_, idx) => (float)((idx + 1) * 4547 % 17)).Reverse().ToArray();

                                Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);
                                Map2D y = MaxPoolingTest.Reference(x, (int)sx, (int)kw, (int)sy, (int)kh);
                                Map2D dy = new((int)c, (int)ow, (int)oh, (int)n, dyval);
                                Map2D dx = Reference(x, y, dy, (int)iw, (int)sx, (int)kw, (int)ih, (int)sy, (int)kh);

                                Array<float> x_tensor = xval;
                                Array<float> y_tensor = y.ToFloatArray();
                                Array<float> dy_tensor = dyval;
                                Array<float> dx_tensor = new(c * iw * ih * n, zeroset: false);

                                Pool2D.MaxUnpooling(n, c, iw, ih, sx, sy, kw, kh, x_tensor, y_tensor, dy_tensor, dx_tensor);

                                float[] dx_expect = dx.ToFloatArray();
                                float[] dx_actual = dx_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);
                                CollectionAssert.AreEqual(y.ToFloatArray(), (float[])y_tensor);
                                CollectionAssert.AreEqual(dyval, (float[])dy_tensor);

                                AssertError.Tolerance(dx_expect, dx_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{sx},{sy},{kw},{kh},{n}");

                                Console.WriteLine($"OK: {c},{iw},{ih},{sx},{sy},{kw},{kh},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, Map2D y, Map2D dy, int iw, int sx, int kw, int ih, int sy, int kh) {
            int channels = y.Channels, batch = y.Batch;
            int ow = (iw - 1) / sx + 1, oh = (ih - 1) / sy + 1;

            if (y.Width != ow || dy.Width != ow || y.Height != oh || dy.Height != oh) {
                throw new ArgumentException("mismatch shape");
            }

            Map2D dx = new(channels, iw, ih, batch);

            for (int th = 0; th < batch; th++) {
                for (int isy = 0, oy = 0; oy < oh; isy += sy, oy++) {
                    for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                        for (int c = 0; c < channels; c++) {
                            for (int ky = 0, iy = isy + ky; ky < kh && iy < ih; ky++, iy = isy + ky) {
                                for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                                    if (y[c, ox, oy, th] <= x[c, ix, iy, th]) {
                                        dx[c, ix, iy, th] += dy[c, ox, oy, th];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return dx;
        }
    }
}
