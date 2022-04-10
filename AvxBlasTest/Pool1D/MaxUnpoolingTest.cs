using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool1DTest {
    [TestClass]
    public class MaxUnpoolingTest {
        [TestMethod]
        public void SMaxUnpoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    foreach (uint sx in new int[] { 1, 2, 3, 4 }) {
                        uint ow = (iw - 1) / sx + 1;

                        foreach (uint kw in new int[] { 2, 3, 4, 5 }) {

                            for (uint c = 1; c <= 65; c++) {

                                float[] xval = (new float[c * iw * n]).Select((_, idx) => (float)(idx * 4547 % 17)).ToArray();
                                float[] dyval = (new float[c * ow * n]).Select((_, idx) => (float)((idx + 1) * 4547 % 17)).Reverse().ToArray();

                                Map1D x = new((int)c, (int)iw, (int)n, xval);
                                Map1D y = MaxPoolingTest.Reference(x, (int)sx, (int)kw);
                                Map1D dy = new((int)c, (int)ow, (int)n, dyval);
                                Map1D dx = Reference(x, y, dy, (int)iw, (int)sx, (int)kw);

                                Array<float> x_tensor = xval;
                                Array<float> y_tensor = y.ToFloatArray();
                                Array<float> dy_tensor = dyval;
                                Array<float> dx_tensor = new(c * iw * n, zeroset: false);

                                Pool1D.MaxUnpooling(n, c, iw, sx, kw, x_tensor, y_tensor, dy_tensor, dx_tensor);

                                float[] dx_expect = dx.ToFloatArray();
                                float[] dx_actual = dx_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);
                                CollectionAssert.AreEqual(y.ToFloatArray(), (float[])y_tensor);
                                CollectionAssert.AreEqual(dyval, (float[])dy_tensor);

                                AssertError.Tolerance(dx_expect, dx_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{sx},{kw},{n}");

                                Console.WriteLine($"OK: {c},{iw},{sx},{kw},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x, Map1D y, Map1D dy, int iw, int sx, int kw) {
            int channels = y.Channels, batch = y.Batch;
            int ow = (iw - 1) / sx + 1;

            if (y.Width != ow || dy.Width != ow) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D dx = new(channels, iw, batch);

            for (int th = 0; th < batch; th++) {
                for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                    for (int c = 0; c < channels; c++) {
                        for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                            if (y[c, ox, th] <= x[c, ix, th]) {
                                dx[c, ix, th] += dy[c, ox, th];
                            }
                        }
                    }
                }
            }

            return dx;
        }
    }
}
