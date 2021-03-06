using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Upsample1DTest {
    [TestClass]
    public class NeighborTest {
        [TestMethod]
        public void SNeighborTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    uint ow = iw * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map1D x = new((int)c, (int)iw, (int)n, xval);

                        Map1D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * n, zeroset: false);

                        Upsample1D.NeighborX2(n, c, iw, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{n}");

                        Console.WriteLine($"OK: {c},{iw},{n}");

                    }
                }
            }

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                for (uint iw = 1; iw <= 65; iw++) {
                    const uint c = 1;

                    uint ow = iw * 2;

                    float[] xval = (new float[c * iw * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                    Map1D x = new((int)c, (int)iw, (int)n, xval);

                    Map1D y = Reference(x);

                    Array<float> x_tensor = xval;
                    Array<float> y_tensor = new(c * ow * n, zeroset: false);

                    Upsample1D.NeighborX2(n, c, iw, x_tensor, y_tensor);

                    float[] y_expect = y.ToFloatArray();
                    float[] y_actual = y_tensor;

                    CollectionAssert.AreEqual(xval, (float[])x_tensor);

                    AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{n}");

                    Console.WriteLine($"OK: {c},{iw},{n}");
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D x) {
            int inw = x.Width, channels = x.Channels, batch = x.Batch;

            int outw = inw * 2;
            Map1D y = new Map1D(channels, outw, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox = 0; ox < outw; ox++) {
                    for (int f = 0; f < channels; f++) {
                        y[f, ox, th] = x[f, ox / 2, th];
                    }
                }

            }

            return y;
        }
    }
}
