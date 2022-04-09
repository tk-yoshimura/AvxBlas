using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Upsample2DTest {
    [TestClass]
    public class NeighborTest {
        [TestMethod]
        public void SNeighborTest() {
            float max_err = 0;

            uint scale = 2;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    uint ow = iw * scale, oh = ih * scale;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x, (int)scale);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Upsample2D.NeighborX2(n, c, iw, ih, x_tensor, y_tensor);

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

                        uint ow = iw * scale, oh = ih * scale;

                        float[] xval = (new float[c * iw * ih * n]).Select((_, idx) => idx * 1e-3f).ToArray();

                        Map2D x = new((int)c, (int)iw, (int)ih, (int)n, xval);

                        Map2D y = Reference(x, (int)scale);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * n, zeroset: false);

                        Upsample2D.NeighborX2(n, c, iw, ih, x_tensor, y_tensor);

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

        public static Map2D Reference(Map2D x, int scale) {
            int inw = x.Width, inh = x.Height, channels = x.Channels, batch = x.Batch;

            int outw = inw * scale, outh = inh * scale;
            Map2D y = new Map2D(channels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy = 0; oy < outh; oy++) {
                    for (ox = 0; ox < outw; ox++) {
                        for (int f = 0; f < channels; f++) {
                            y[f, ox, oy, th] = x[f, ox / scale, oy / scale, th];
                        }

                    }
                }

            }

            return y;
        }
    }
}
