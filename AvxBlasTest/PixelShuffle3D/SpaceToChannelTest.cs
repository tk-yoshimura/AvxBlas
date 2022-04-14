using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.PixelShuffle3DTest {
    [TestClass]
    public class SpaceToChannelTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17 }) {
                    foreach (uint s in new int[] { 4 }) {
                        foreach ((uint ow, uint oh, uint od) in new (uint, uint, uint)[] {
                             (1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (5, 2, 8), (3, 9, 6), (12, 16, 15) }) {

                            uint iw = ow * s, ih = oh * s, id = od * s, oc = ic * s * s * s;

                            float[] xval = (new float[iw * ih * id * ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map3D x = new Map3D((int)ic, (int)iw, (int)ih, (int)id, (int)n, xval);

                            Map3D y = Reference(x, (int)s);

                            Array<float> x_tensor = xval;
                            Array<float> y_tensor = new(oc * ow * oh * od * n, zeroset: false);

                            PixelShuffle3D.SpaceToChannel(n, ic, iw, ih, id, s, x_tensor, y_tensor);

                            float[] y_expect = y.ToFloatArray();
                            float[] y_actual = y_tensor;

                            CollectionAssert.AreEqual(xval, (float[])x_tensor);

                            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG {ic},{oc},{s},{iw},{ih},{id},{n}");

                            Console.WriteLine($"OK: {ic},{oc},{s},{iw},{ih},{id},{n}");

                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x, int scale) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, inchannels = x.Channels, batch = x.Batch;
            if (inw % scale != 0 || inh % scale != 0 || ind % scale != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int outw = inw / scale, outh = inh / scale, outd = ind / scale;
            int outchannels = inchannels * scale * scale * scale;

            Map3D y = new Map3D(outchannels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ox, oy, oz = 0; oz < outd; oz++) {
                    for (oy = 0; oy < outh; oy++) {
                        for (ox = 0; ox < outw; ox++) {
                            for (int kx, ky, kz = 0; kz < scale; kz++) {
                                for (ky = 0; ky < scale; ky++) {
                                    for (kx = 0; kx < scale; kx++) {
                                        for (int inch = 0; inch < inchannels; inch++) {
                                            int outch = inch + (kx + ky * scale + kz * scale * scale) * inchannels;

                                            y[outch, ox, oy, oz, th] = x[inch, ox * scale + kx, oy * scale + ky, oz * scale + kz, th];

                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }
    }
}
