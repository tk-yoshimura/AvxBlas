using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.IO;
using System.Linq;

namespace AvxBlasTest.Downsample3DTest {
    [TestClass]
    public class InterareaTest {
        [TestMethod]
        public void SInterareaTest() {
            using StreamWriter sw = new("downsample3d.txt");

            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint ow, uint oh, uint od) in new (uint, uint, uint)[] {
                    (1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (5, 2, 8), (3, 9, 6), (12, 16, 15) }) {

                    uint iw = ow * 2, ih = oh * 2, id = od * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                        Map3D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                        sw.WriteLine($"{c},{iw},{ih},{id},{n}");
                        sw.Flush();

                        Downsample3D.InterareaX2(n, c, iw, ih, id, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{id},{n}");
                    }
                }
            }

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach ((uint oh, uint od) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (2, 3), (5, 8), (7, 6), (16, 15), (17, 28), (32, 30) }) {
                    for (uint ow = 1; ow <= 65; ow++) {
                        const uint c = 1;

                        uint iw = ow * 2, ih = oh * 2, id = od * 2;

                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                        Map3D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                        sw.WriteLine($"{c},{iw},{ih},{id},{n}");
                        sw.Flush();

                        Downsample3D.InterareaX2(n, c, iw, ih, id, x_tensor, y_tensor);

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

            int outw = inw / 2, outh = inh / 2, outd = ind / 2;
            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int iz = 0; iz < ind; iz += 2) {
                    for (int iy = 0; iy < inh; iy += 2) {
                        for (int ix = 0; ix < inw; ix += 2) {
                            for (int f = 0; f < channels; f++) {
                                double luf = x[f, ix, iy, iz, th];
                                double ruf = x[f, ix + 1, iy, iz, th];
                                double ldf = x[f, ix, iy + 1, iz, th];
                                double rdf = x[f, ix + 1, iy + 1, iz, th];
                                double lub = x[f, ix, iy, iz + 1, th];
                                double rub = x[f, ix + 1, iy, iz + 1, th];
                                double ldb = x[f, ix, iy + 1, iz + 1, th];
                                double rdb = x[f, ix + 1, iy + 1, iz + 1, th];

                                y[f, ix / 2, iy / 2, iz / 2, th] = (luf + ruf + ldf + rdf + lub + rub + ldb + rdb) / 8;
                            }
                        }
                    }
                }

            }

            return y;
        }
    }
}
