using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.PixelShuffle2DTest {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17 }) {
                    foreach (uint s in new int[] { 1, 2, 3, 4 }) {
                        foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 7, 11 }) {
                            foreach (uint ih in new int[] { 1, 2, 3, 4, 5, 7, 11 }) {
                                uint ow = iw * s, oh = ih * s, ic = oc * s * s;

                                float[] xval = (new float[iw * ih * ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();

                                Map2D x = new Map2D((int)ic, (int)iw, (int)ih, (int)n, xval);

                                Map2D y = Reference(x, (int)s);

                                Array<float> x_tensor = xval;
                                Array<float> y_tensor = new(oc * ow * oh * n, zeroset: false);

                                PixelShuffle2D.ChannelToSpace(n, ic, iw, ih, s, x_tensor, y_tensor);

                                float[] y_expect = y.ToFloatArray();
                                float[] y_actual = y_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);

                                AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG {ic},{oc},{s},{iw},{ih},{n}");

                                Console.WriteLine($"OK: {ic},{oc},{s},{iw},{ih},{n}");

                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, int scale) {
            int inchannels = x.Channels, batch = x.Batch;
            if (inchannels % (scale * scale) != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int inw = x.Width, inh = x.Height, outw = inw * scale, outh = inh * scale;
            int outchannels = inchannels / (scale * scale);

            Map2D y = new Map2D(outchannels, outw, outh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy = 0; iy < inh; iy++) {
                    for (ix = 0; ix < inw; ix++) {
                        for (int kx, ky = 0; ky < scale; ky++) {
                            for (kx = 0; kx < scale; kx++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    int inch = outch + kx * outchannels + ky * outchannels * scale;

                                    y[outch, ix * scale + kx, iy * scale + ky, th] = x[inch, ix, iy, th];

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
