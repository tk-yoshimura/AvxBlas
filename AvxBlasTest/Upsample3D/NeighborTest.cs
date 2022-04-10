using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Upsample3DTest {
    [TestClass]
    public class NeighborTest {
        [TestMethod]
        public void SNeighborTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih, uint id) in new (uint, uint, uint)[] {
                    (1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (5, 2, 8), (3, 9, 6), (12, 16, 15) }) {

                    uint ow = iw * 2, oh = ih * 2, od = id * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                        Map3D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                        Upsample3D.NeighborX2(n, c, iw, ih, id, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{id},{n}");

                    }
                }
            }

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach ((uint ih, uint id) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    for (uint iw = 1; iw <= 65; iw++) {
                        const uint c = 1;

                        uint ow = iw * 2, oh = ih * 2, od = id * 2;

                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                        Map3D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                        Upsample3D.NeighborX2(n, c, iw, ih, id, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{id},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, channels = x.Channels, batch = x.Batch;

            int outw = inw * 2, outh = inh * 2, outd = ind * 2;
            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int f = 0; f < channels; f++) {
                                y[f, ox, oy, oz, th] = x[f, ox / 2, oy / 2, oz / 2, th];
                            }

                        }
                    }
                }

            }

            return y;
        }
    }
}
