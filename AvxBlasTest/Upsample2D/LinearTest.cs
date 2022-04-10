using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Upsample2DTest {
    [TestClass]
    public class LinearTest {
        [TestMethod]
        public void SLinearTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    uint ow = iw * 2, oh = ih * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Upsample2D.LinearX2(n, c, iw, ih, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{n}");
                    }
                }
            }

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ih in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    for (uint iw = 1; iw <= 65; iw++) {
                        const uint c = 1;

                        uint ow = iw * 2, oh = ih * 2;

                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Upsample2D.LinearX2(n, c, iw, ih, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x) {
            int inw = x.Width, inh = x.Height, channels = x.Channels, batch = x.Batch;

            int outw = inw * 2, outh = inh * 2;
            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        for (int f = 0; f < channels; f++) {
                            double c = x[f, ix, iy, th];

                            double l = x[f, Math.Max(0, ix - 1), iy, th];
                            double r = x[f, Math.Min(inw - 1, ix + 1), iy, th];
                            double u = x[f, ix, Math.Max(0, iy - 1), th];
                            double d = x[f, ix, Math.Min(inh - 1, iy + 1), th];

                            double lu = x[f, Math.Max(0, ix - 1),       Math.Max(0, iy - 1), th];
                            double ru = x[f, Math.Min(inw - 1, ix + 1), Math.Max(0, iy - 1), th];
                            double ld = x[f, Math.Max(0, ix - 1),       Math.Min(inh - 1, iy + 1), th];
                            double rd = x[f, Math.Min(inw - 1, ix + 1), Math.Min(inh - 1, iy + 1), th];

                            y[f, ix * 2,     iy * 2,     th] = (4 * c + 2 * l + 2 * u + lu) / 9;
                            y[f, ix * 2 + 1, iy * 2,     th] = (4 * c + 2 * r + 2 * u + ru) / 9;
                            y[f, ix * 2,     iy * 2 + 1, th] = (4 * c + 2 * l + 2 * d + ld) / 9;
                            y[f, ix * 2 + 1, iy * 2 + 1, th] = (4 * c + 2 * r + 2 * d + rd) / 9;
                        }
                    }
                }

            }

            return y;
        }
    }
}
