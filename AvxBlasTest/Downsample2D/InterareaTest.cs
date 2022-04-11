using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Downsample2DTest {
    [TestClass]
    public class InterareaTest {
        [TestMethod]
        public void SInterareaTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint ow, uint oh) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (2, 3), (5, 8), (7, 6), (16, 15), (17, 28), (32, 30) }) {
                    uint iw = ow * 2, ih = oh * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Downsample2D.InterareaX2(n, c, iw, ih, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{n}");
                    }
                }
            }

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach (uint oh in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    for (uint ow = 1; ow <= 65; ow++) {
                        const uint c = 1;

                        uint iw = ow * 2, ih = oh * 2;

                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Downsample2D.InterareaX2(n, c, iw, ih, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{n}");

                        Console.WriteLine($"OK: {c},{iw},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x) {
            int inw = x.Width, inh = x.Height, channels = x.Channels, batch = x.Batch;

            int outw = inw / 2, outh = inh / 2;
            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int iy = 0; iy < inh; iy += 2) {
                    for (int ix = 0; ix < inw; ix += 2) {
                        for (int f = 0; f < channels; f++) {
                            double lu = x[f, ix, iy, th];
                            double ru = x[f, ix + 1, iy, th];
                            double ld = x[f, ix, iy + 1, th];
                            double rd = x[f, ix + 1, iy + 1, th];

                            y[f, ix / 2, iy / 2, th] = (lu + ru + ld + rd) / 4;
                        }
                    }
                }

            }

            return y;
        }
    }
}
