using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.PixelShuffle3DTest {
    [TestClass]
    public class ChannelToSpaceTest {
        [TestMethod]
        public void ExecuteTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17 }) {
                    foreach (uint s in new int[] { 1, 2, 3, 4 }) {
                        foreach ((uint iw, uint ih, uint id) in new (uint, uint, uint)[] {
                            (1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (5, 2, 8), (3, 9, 6), (12, 16, 15) }) {

                            uint ow = iw * s, oh = ih * s, od = id * s, ic = oc * s * s * s;

                            float[] xval = (new float[iw * ih * id * ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();

                            Map3D x = new Map3D((int)ic, (int)iw, (int)ih, (int)id, (int)n, xval);

                            Map3D y = Reference(x, (int)s);

                            Array<float> x_tensor = xval;
                            Array<float> y_tensor = new(oc * ow * oh * od * n, zeroset: false);

                            PixelShuffle3D.ChannelToSpace(n, ic, iw, ih, id, s, x_tensor, y_tensor);

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
            int inchannels = x.Channels, batch = x.Batch;
            if (inchannels % (scale * scale * scale) != 0) {
                throw new ArgumentException(nameof(scale));
            }

            int inw = x.Width, inh = x.Height, ind = x.Depth, outw = inw * scale, outh = inh * scale, outd = ind * scale;
            int outchannels = inchannels / (scale * scale * scale);

            Map3D y = new Map3D(outchannels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            for (int kx, ky, kz = 0; kz < scale; kz++) {
                                for (ky = 0; ky < scale; ky++) {
                                    for (kx = 0; kx < scale; kx++) {
                                        for (int outch = 0; outch < outchannels; outch++) {
                                            int inch = outch + (kx + ky * scale + kz * scale * scale) * outchannels;

                                            y[outch, ix * scale + kx, iy * scale + ky, iz * scale + kz, th] = x[inch, ix, iy, iz, th];

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
