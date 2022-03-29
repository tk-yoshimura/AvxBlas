using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection2DTest {
    [TestClass]
    public class BackwardFilterPadEdgeTest {
        [TestMethod]
        public void SBackwardFilterPadEdgeTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 2), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    foreach ((uint kw, uint kh) in new (uint, uint)[] { (1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (7, 7) }) {
                        uint ow = iw, oh = ih;

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

                            Convolution2D.BackwardFilter(n, ic, oc, iw, ih, kw, kh, PadMode.Edge, x_tensor, gy_tensor, gw_tensor);

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

            if (ow != iw || oh != ih) {
                throw new ArgumentException("mismatch shape");
            }

            Filter2D w = new(inchannels, kw, kh, outchannels);

            for (int th = 0; th < batch; th++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int oy = 0; oy < oh; oy++) {
                            int iy = Math.Min(ih - 1, Math.Max(0, ky + oy - kh / 2));

                            for (int ox = 0; ox < ow; ox++) {
                                int ix = Math.Min(iw - 1, Math.Max(0, kx + ox - kw / 2));

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
                2.156000000e-02f, 2.254800000e-02f, 2.353600000e-02f, 2.452400000e-02f, 2.551200000e-02f, 2.650000000e-02f, 2.748800000e-02f,
                2.748200000e-02f, 2.847000000e-02f, 2.945800000e-02f, 3.044600000e-02f, 3.143400000e-02f, 3.242200000e-02f, 3.341000000e-02f,
                3.432800000e-02f, 3.531600000e-02f, 3.630400000e-02f, 3.729200000e-02f, 3.828000000e-02f, 3.926800000e-02f, 4.025600000e-02f,
                2.109800000e-02f, 2.207300000e-02f, 2.304800000e-02f, 2.402300000e-02f, 2.499800000e-02f, 2.597300000e-02f, 2.694800000e-02f,
                2.693600000e-02f, 2.791100000e-02f, 2.888600000e-02f, 2.986100000e-02f, 3.083600000e-02f, 3.181100000e-02f, 3.278600000e-02f,
                3.369800000e-02f, 3.467300000e-02f, 3.564800000e-02f, 3.662300000e-02f, 3.759800000e-02f, 3.857300000e-02f, 3.954800000e-02f,
                2.063600000e-02f, 2.159800000e-02f, 2.256000000e-02f, 2.352200000e-02f, 2.448400000e-02f, 2.544600000e-02f, 2.640800000e-02f,
                2.639000000e-02f, 2.735200000e-02f, 2.831400000e-02f, 2.927600000e-02f, 3.023800000e-02f, 3.120000000e-02f, 3.216200000e-02f,
                3.306800000e-02f, 3.403000000e-02f, 3.499200000e-02f, 3.595400000e-02f, 3.691600000e-02f, 3.787800000e-02f, 3.884000000e-02f,
                2.017400000e-02f, 2.112300000e-02f, 2.207200000e-02f, 2.302100000e-02f, 2.397000000e-02f, 2.491900000e-02f, 2.586800000e-02f,
                2.584400000e-02f, 2.679300000e-02f, 2.774200000e-02f, 2.869100000e-02f, 2.964000000e-02f, 3.058900000e-02f, 3.153800000e-02f,
                3.243800000e-02f, 3.338700000e-02f, 3.433600000e-02f, 3.528500000e-02f, 3.623400000e-02f, 3.718300000e-02f, 3.813200000e-02f,
                1.971200000e-02f, 2.064800000e-02f, 2.158400000e-02f, 2.252000000e-02f, 2.345600000e-02f, 2.439200000e-02f, 2.532800000e-02f,
                2.529800000e-02f, 2.623400000e-02f, 2.717000000e-02f, 2.810600000e-02f, 2.904200000e-02f, 2.997800000e-02f, 3.091400000e-02f,
                3.180800000e-02f, 3.274400000e-02f, 3.368000000e-02f, 3.461600000e-02f, 3.555200000e-02f, 3.648800000e-02f, 3.742400000e-02f,
                1.925000000e-02f, 2.017300000e-02f, 2.109600000e-02f, 2.201900000e-02f, 2.294200000e-02f, 2.386500000e-02f, 2.478800000e-02f,
                2.475200000e-02f, 2.567500000e-02f, 2.659800000e-02f, 2.752100000e-02f, 2.844400000e-02f, 2.936700000e-02f, 3.029000000e-02f,
                3.117800000e-02f, 3.210100000e-02f, 3.302400000e-02f, 3.394700000e-02f, 3.487000000e-02f, 3.579300000e-02f, 3.671600000e-02f,
                1.878800000e-02f, 1.969800000e-02f, 2.060800000e-02f, 2.151800000e-02f, 2.242800000e-02f, 2.333800000e-02f, 2.424800000e-02f,
                2.420600000e-02f, 2.511600000e-02f, 2.602600000e-02f, 2.693600000e-02f, 2.784600000e-02f, 2.875600000e-02f, 2.966600000e-02f,
                3.054800000e-02f, 3.145800000e-02f, 3.236800000e-02f, 3.327800000e-02f, 3.418800000e-02f, 3.509800000e-02f, 3.600800000e-02f,
                1.832600000e-02f, 1.922300000e-02f, 2.012000000e-02f, 2.101700000e-02f, 2.191400000e-02f, 2.281100000e-02f, 2.370800000e-02f,
                2.366000000e-02f, 2.455700000e-02f, 2.545400000e-02f, 2.635100000e-02f, 2.724800000e-02f, 2.814500000e-02f, 2.904200000e-02f,
                2.991800000e-02f, 3.081500000e-02f, 3.171200000e-02f, 3.260900000e-02f, 3.350600000e-02f, 3.440300000e-02f, 3.530000000e-02f,
                1.786400000e-02f, 1.874800000e-02f, 1.963200000e-02f, 2.051600000e-02f, 2.140000000e-02f, 2.228400000e-02f, 2.316800000e-02f,
                2.311400000e-02f, 2.399800000e-02f, 2.488200000e-02f, 2.576600000e-02f, 2.665000000e-02f, 2.753400000e-02f, 2.841800000e-02f,
                2.928800000e-02f, 3.017200000e-02f, 3.105600000e-02f, 3.194000000e-02f, 3.282400000e-02f, 3.370800000e-02f, 3.459200000e-02f,
                1.740200000e-02f, 1.827300000e-02f, 1.914400000e-02f, 2.001500000e-02f, 2.088600000e-02f, 2.175700000e-02f, 2.262800000e-02f,
                2.256800000e-02f, 2.343900000e-02f, 2.431000000e-02f, 2.518100000e-02f, 2.605200000e-02f, 2.692300000e-02f, 2.779400000e-02f,
                2.865800000e-02f, 2.952900000e-02f, 3.040000000e-02f, 3.127100000e-02f, 3.214200000e-02f, 3.301300000e-02f, 3.388400000e-02f,
                1.694000000e-02f, 1.779800000e-02f, 1.865600000e-02f, 1.951400000e-02f, 2.037200000e-02f, 2.123000000e-02f, 2.208800000e-02f,
                2.202200000e-02f, 2.288000000e-02f, 2.373800000e-02f, 2.459600000e-02f, 2.545400000e-02f, 2.631200000e-02f, 2.717000000e-02f,
                2.802800000e-02f, 2.888600000e-02f, 2.974400000e-02f, 3.060200000e-02f, 3.146000000e-02f, 3.231800000e-02f, 3.317600000e-02f,
            };

            float[] gw_actual = gw.ToFloatArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth}");
        }
    }
}
