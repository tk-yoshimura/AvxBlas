using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection2DTest {
    [TestClass]
    public class BackwardFilterPadNoneTest {
        [TestMethod]
        public void SBackwardFilterPadNoneTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 2), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    foreach ((uint kw, uint kh) in new (uint, uint)[] { (1, 1), (1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (7, 7) }) {
                        if (iw < kw || ih < kh) {
                            continue;
                        }
                        uint ow = iw - kw + 1, oh = ih - kh + 1;

                        foreach ((uint ic, uint oc) in new (uint, uint)[] { (1, 1), (2, 3), (3, 2), (4, 5), (5, 4), (8, 10), (10, 8),
                                                                            (7, 16), (16, 7), (9, 24), (24, 9), (31, 32), (32, 31), (15, 64), (64, 15) }) {

                            float[] xval = (new float[iw * ih * ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] gyval = (new float[ow * oh * oc * n]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                            Map2D x = new((int)ic, (int)iw, (int)ih, (int)n, xval);
                            Map2D gy = new((int)oc, (int)ow, (int)oh, (int)n, gyval);

                            Filter2D gw = Reference(x, gy, (int)kw, (int)kh);

                            Array<float> x_tensor = xval;
                            Array<float> gy_tensor = gyval;

                            Array<float> gw_tensor = new(ic * oc * kw * kh, zeroset: false);

                            Convolution2D.BackwardFilter(n, ic, oc, iw, ih, kw, kh, PadMode.None, x_tensor, gy_tensor, gw_tensor);

                            float[] gw_expect = gw.ToFloatArray();
                            float[] gw_actual = gw_tensor;

                            CollectionAssert.AreEqual(xval, (float[])x_tensor);
                            CollectionAssert.AreEqual(gyval, (float[])gy_tensor);

                            AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"NG: {ic},{oc},{iw},{ih},{kw},{kh},{n}");

                            Console.WriteLine($"OK: {ic},{oc},{iw},{ih},{kw},{kh},{n}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Filter2D Reference(Map2D x, Map2D gy, int kw, int kh) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int iw = x.Width, ow = gy.Width, ih = x.Height, oh = gy.Height;

            if (ow != iw - kw + 1 || oh != ih - kh + 1) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new(inchannels, kw, kh, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int iy = ky, oy = 0; oy < oh; iy++, oy++) {
                            for (int ix = kx, ox = 0; ox < ow; ix++, ox++) {
                                for (int inch, outch = 0; outch < outchannels; outch++) {
                                    for (inch = 0; inch < inchannels; inch++) {
                                        w[inch, kx, ky, outch] += x[inch, ix, iy, th] * gy[outch, ox, oy, th];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11;
            int kwidth = 3, kheight = 5, inwidth = 9, inheight = 13;
            int outwidth = inwidth, outheight = inheight;

            float[] xval = (new float[inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outheight * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new(inchannels, inwidth, inheight, 1, xval);
            Map2D gy = new(outchannels, outwidth, outheight, 1, gyval);

            Filter2D gw = Reference(x, gy, kwidth, kheight);

            float[] gw_expect = {
                1.655500000e-02f, 1.727000000e-02f, 1.798500000e-02f, 1.870000000e-02f, 1.941500000e-02f, 2.013000000e-02f, 2.084500000e-02f,
                2.156000000e-02f, 2.227500000e-02f, 2.299000000e-02f, 2.370500000e-02f, 2.442000000e-02f, 2.513500000e-02f, 2.585000000e-02f,
                2.656500000e-02f, 2.728000000e-02f, 2.799500000e-02f, 2.871000000e-02f, 2.942500000e-02f, 3.014000000e-02f, 3.085500000e-02f,
                1.617000000e-02f, 1.687400000e-02f, 1.757800000e-02f, 1.828200000e-02f, 1.898600000e-02f, 1.969000000e-02f, 2.039400000e-02f,
                2.109800000e-02f, 2.180200000e-02f, 2.250600000e-02f, 2.321000000e-02f, 2.391400000e-02f, 2.461800000e-02f, 2.532200000e-02f,
                2.602600000e-02f, 2.673000000e-02f, 2.743400000e-02f, 2.813800000e-02f, 2.884200000e-02f, 2.954600000e-02f, 3.025000000e-02f,
                1.578500000e-02f, 1.647800000e-02f, 1.717100000e-02f, 1.786400000e-02f, 1.855700000e-02f, 1.925000000e-02f, 1.994300000e-02f,
                2.063600000e-02f, 2.132900000e-02f, 2.202200000e-02f, 2.271500000e-02f, 2.340800000e-02f, 2.410100000e-02f, 2.479400000e-02f,
                2.548700000e-02f, 2.618000000e-02f, 2.687300000e-02f, 2.756600000e-02f, 2.825900000e-02f, 2.895200000e-02f, 2.964500000e-02f,
                1.540000000e-02f, 1.608200000e-02f, 1.676400000e-02f, 1.744600000e-02f, 1.812800000e-02f, 1.881000000e-02f, 1.949200000e-02f,
                2.017400000e-02f, 2.085600000e-02f, 2.153800000e-02f, 2.222000000e-02f, 2.290200000e-02f, 2.358400000e-02f, 2.426600000e-02f,
                2.494800000e-02f, 2.563000000e-02f, 2.631200000e-02f, 2.699400000e-02f, 2.767600000e-02f, 2.835800000e-02f, 2.904000000e-02f,
                1.501500000e-02f, 1.568600000e-02f, 1.635700000e-02f, 1.702800000e-02f, 1.769900000e-02f, 1.837000000e-02f, 1.904100000e-02f,
                1.971200000e-02f, 2.038300000e-02f, 2.105400000e-02f, 2.172500000e-02f, 2.239600000e-02f, 2.306700000e-02f, 2.373800000e-02f,
                2.440900000e-02f, 2.508000000e-02f, 2.575100000e-02f, 2.642200000e-02f, 2.709300000e-02f, 2.776400000e-02f, 2.843500000e-02f,
                1.463000000e-02f, 1.529000000e-02f, 1.595000000e-02f, 1.661000000e-02f, 1.727000000e-02f, 1.793000000e-02f, 1.859000000e-02f,
                1.925000000e-02f, 1.991000000e-02f, 2.057000000e-02f, 2.123000000e-02f, 2.189000000e-02f, 2.255000000e-02f, 2.321000000e-02f,
                2.387000000e-02f, 2.453000000e-02f, 2.519000000e-02f, 2.585000000e-02f, 2.651000000e-02f, 2.717000000e-02f, 2.783000000e-02f,
                1.424500000e-02f, 1.489400000e-02f, 1.554300000e-02f, 1.619200000e-02f, 1.684100000e-02f, 1.749000000e-02f, 1.813900000e-02f,
                1.878800000e-02f, 1.943700000e-02f, 2.008600000e-02f, 2.073500000e-02f, 2.138400000e-02f, 2.203300000e-02f, 2.268200000e-02f,
                2.333100000e-02f, 2.398000000e-02f, 2.462900000e-02f, 2.527800000e-02f, 2.592700000e-02f, 2.657600000e-02f, 2.722500000e-02f,
                1.386000000e-02f, 1.449800000e-02f, 1.513600000e-02f, 1.577400000e-02f, 1.641200000e-02f, 1.705000000e-02f, 1.768800000e-02f,
                1.832600000e-02f, 1.896400000e-02f, 1.960200000e-02f, 2.024000000e-02f, 2.087800000e-02f, 2.151600000e-02f, 2.215400000e-02f,
                2.279200000e-02f, 2.343000000e-02f, 2.406800000e-02f, 2.470600000e-02f, 2.534400000e-02f, 2.598200000e-02f, 2.662000000e-02f,
                1.347500000e-02f, 1.410200000e-02f, 1.472900000e-02f, 1.535600000e-02f, 1.598300000e-02f, 1.661000000e-02f, 1.723700000e-02f,
                1.786400000e-02f, 1.849100000e-02f, 1.911800000e-02f, 1.974500000e-02f, 2.037200000e-02f, 2.099900000e-02f, 2.162600000e-02f,
                2.225300000e-02f, 2.288000000e-02f, 2.350700000e-02f, 2.413400000e-02f, 2.476100000e-02f, 2.538800000e-02f, 2.601500000e-02f,
                1.309000000e-02f, 1.370600000e-02f, 1.432200000e-02f, 1.493800000e-02f, 1.555400000e-02f, 1.617000000e-02f, 1.678600000e-02f,
                1.740200000e-02f, 1.801800000e-02f, 1.863400000e-02f, 1.925000000e-02f, 1.986600000e-02f, 2.048200000e-02f, 2.109800000e-02f,
                2.171400000e-02f, 2.233000000e-02f, 2.294600000e-02f, 2.356200000e-02f, 2.417800000e-02f, 2.479400000e-02f, 2.541000000e-02f,
                1.270500000e-02f, 1.331000000e-02f, 1.391500000e-02f, 1.452000000e-02f, 1.512500000e-02f, 1.573000000e-02f, 1.633500000e-02f,
                1.694000000e-02f, 1.754500000e-02f, 1.815000000e-02f, 1.875500000e-02f, 1.936000000e-02f, 1.996500000e-02f, 2.057000000e-02f,
                2.117500000e-02f, 2.178000000e-02f, 2.238500000e-02f, 2.299000000e-02f, 2.359500000e-02f, 2.420000000e-02f, 2.480500000e-02f,
            };

            float[] gw_actual = gw.ToFloatArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth}");
        }
    }
}
