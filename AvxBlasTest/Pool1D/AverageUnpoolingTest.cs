using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool1DTest {
    [TestClass]
    public class AverageUnpoolingTest {
        [TestMethod]
        public void SMaxUnpoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    foreach (uint sx in new int[] { 1, 2, 3, 4 }) {
                        uint ow = (iw - 1) / sx + 1;

                        foreach (uint kw in new int[] { 2, 3, 4, 5 }) {

                            for (uint c = 1; c <= 65; c++) {

                                float[] dyval = (new float[c * ow * n]).Select((_, idx) => (float)((idx + 1) * 4547 % 17)).Reverse().ToArray();

                                Map1D dy = new((int)c, (int)ow, (int)n, dyval);
                                Map1D dx = Reference(dy, (int)iw, (int)sx, (int)kw);

                                Array<float> dy_tensor = dyval;
                                Array<float> dx_tensor = new(c * iw * n, zeroset: false);

                                Pool1D.AverageUnpooling(n, c, iw, sx, kw, dy_tensor, dx_tensor);

                                float[] dx_expect = dx.ToFloatArray();
                                float[] dx_actual = dx_tensor;

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

        public static Map1D Reference(Map1D dy, int iw, int sx, int kw) {
            int channels = dy.Channels, batch = dy.Batch;
            int ow = (iw - 1) / sx + 1;

            if (dy.Width != ow) { 
                throw new ArgumentException("mismatch shape");
            }

            Map1D dx = new(channels, iw, batch);

            for (int th = 0; th < batch; th++) {
                for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                    for (int c = 0; c < channels; c++) {
                        for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                            dx[c, ix, th] += dy[c, ox, th];
                        }
                    }
                }
            }

            for (int th = 0; th < batch; th++) {
                for (int ix = 0; ix < iw; ix++) {
                    for (int c = 0; c < channels; c++) {
                        dx[c, ix, th] /= kw;
                    }
                }
            }

            return dx;
        }

        [TestMethod]
        public void ReferenceTest() {
            //int channels = 7, stridex = 2, kwidth = 2, inwidth = 13, batch = 2;
            //
            //float[] xval = (new float[batch * inwidth * channels]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();
            //
            //Map1D x = new(channels, inwidth, batch, xval);
            //
            //Map1D y = Reference(x, stridex, kwidth);
            //
            //float[] y_expect = {
            //    6.00e+00f, 2.40e+01f, 3.60e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 2.80e+01f, 1.80e+01f, 1.90e+01f, 3.10e+01f, 1.50e+01f, 3.30e+01f, 2.20e+01f, 2.90e+01f,
            //    1.30e+01f, 3.10e+01f, 9.00e+00f, 2.70e+01f, 2.80e+01f, 2.30e+01f, 2.40e+01f, 1.90e+01f, 2.60e+01f, 1.50e+01f, 2.20e+01f, 1.70e+01f, 1.80e+01f, 3.60e+01f,
            //    2.00e+01f, 3.20e+01f, 1.60e+01f, 2.80e+01f, 1.20e+01f, 2.40e+01f, 2.50e+01f, 2.60e+01f, 3.30e+01f, 2.20e+01f, 2.90e+01f, 1.80e+01f, 2.50e+01f, 1.40e+01f,
            //    2.10e+01f, 2.20e+01f, 1.70e+01f, 3.50e+01f, 1.30e+01f, 3.10e+01f, 9.00e+00f, 2.70e+01f, 1.10e+01f, 2.30e+01f, 3.00e+01f, 1.90e+01f, 2.60e+01f, 2.10e+01f,
            //    2.80e+01f, 1.70e+01f, 2.40e+01f, 3.60e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 2.30e+01f, 1.80e+01f, 3.00e+01f, 1.40e+01f, 2.60e+01f, 3.30e+01f, 2.20e+01f,
            //    3.50e+01f, 1.30e+01f, 3.10e+01f, 9.00e+00f, 2.70e+01f, 1.60e+01f, 2.30e+01f, 3.00e+01f, 2.50e+01f, 2.60e+01f, 2.10e+01f, 2.20e+01f, 1.70e+01f, 1.80e+01f,
            //    1.30e+01f, 2.00e+01f, 3.20e+01f, 1.60e+01f, 3.40e+01f, 1.20e+01f, 3.00e+01f, 8.00e+00f, 2.60e+01f, 2.70e+01f, 2.20e+01f, 2.30e+01f, 1.80e+01f, 1.90e+01f,
            //};
            //
            //float[] y_actual = y.ToFloatArray();
            //
            //AssertError.Tolerance(y_expect, y_actual, 1e-10f, 1e-5f);
        }
    }
}
