using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection2DTest {
    [TestClass]
    public class BackwardDataPadNoneTest {
        [TestMethod]
        public void SBackwardDataPadNoneTest() {
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

                            float[] yval = (new float[ow * oh * oc * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                            float[] wval = (new float[kw * kh * ic * oc]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map2D y = new((int)oc, (int)ow, (int)oh, (int)n, yval);
                            Filter2D w = new((int)ic, (int)kw, (int)kh, (int)oc, wval);

                            Map2D x = Reference(y, w, (int)iw, (int)kw, (int)ih, (int)kh);

                            Array<float> y_tensor = yval;
                            Array<float> w_tensor = wval;
                            Array<float> x_tensor = new(ic * iw * ih * n, zeroset: false);

                            Convolution2D.BackwardData(n, ic, oc, iw, ih, kw, kh, PadMode.None, y_tensor, w_tensor, x_tensor);

                            float[] x_expect = x.ToFloatArray();
                            float[] x_actual = x_tensor;

                            CollectionAssert.AreEqual(yval, (float[])y_tensor);
                            CollectionAssert.AreEqual(wval, (float[])w_tensor);

                            AssertError.Tolerance(x_expect, x_actual, 1e-8f, 1e-6f, ref max_err, $"NG: {ic},{oc},{iw},{ih},{kw},{kh},{n}");

                            Console.WriteLine($"OK: {ic},{oc},{iw},{ih},{kw},{kh},{n}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D y, Filter2D w, int iw, int kw, int ih, int kh) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int ow = iw - kw + 1, oh = ih - kh + 1;

            if (y.Width != ow || y.Height != oh) {
                throw new ArgumentException("mismatch shape");
            }

            Map2D x = new(inchannels, iw, ih, batch);

            for (int th = 0; th < batch; th++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int oy = 0; oy < oh; oy++) {
                            for (int ox = 0; ox < ow; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    double v = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        x[inch, kx + ox, ky + oy, th] += v * w[inch, kx, ky, outch];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11;
            int kwidth = 3, kheight = 5, inwidth = 9, inheight = 13, batch = 2;
            int outwidth = inwidth, outheight = inheight;

            float[] yval = (new float[outwidth * outheight * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D y = new(outchannels, outwidth, outheight, batch, yval);
            Filter2D w = new(inchannels, kwidth, kheight, outchannels, wval);

            Map2D x = Reference(y, w, inwidth, kwidth, inheight, kheight);

            float[] x_expect = {
                4.565000000e-03f, 4.510000000e-03f, 4.455000000e-03f, 4.400000000e-03f, 4.345000000e-03f, 4.290000000e-03f, 4.235000000e-03f,
                2.387000000e-02f, 2.363900000e-02f, 2.340800000e-02f, 2.317700000e-02f, 2.294600000e-02f, 2.271500000e-02f, 2.248400000e-02f,
                5.706800000e-02f, 5.654000000e-02f, 5.601200000e-02f, 5.548400000e-02f, 5.495600000e-02f, 5.442800000e-02f, 5.390000000e-02f,
                9.990200000e-02f, 9.901100000e-02f, 9.812000000e-02f, 9.722900000e-02f, 9.633800000e-02f, 9.544700000e-02f, 9.455600000e-02f,
                1.427360000e-01f, 1.414820000e-01f, 1.402280000e-01f, 1.389740000e-01f, 1.377200000e-01f, 1.364660000e-01f, 1.352120000e-01f,
                1.855700000e-01f, 1.839530000e-01f, 1.823360000e-01f, 1.807190000e-01f, 1.791020000e-01f, 1.774850000e-01f, 1.758680000e-01f,
                2.284040000e-01f, 2.264240000e-01f, 2.244440000e-01f, 2.224640000e-01f, 2.204840000e-01f, 2.185040000e-01f, 2.165240000e-01f,
                2.712380000e-01f, 2.688950000e-01f, 2.665520000e-01f, 2.642090000e-01f, 2.618660000e-01f, 2.595230000e-01f, 2.571800000e-01f,
                3.140720000e-01f, 3.113660000e-01f, 3.086600000e-01f, 3.059540000e-01f, 3.032480000e-01f, 3.005420000e-01f, 2.978360000e-01f,
                3.569060000e-01f, 3.538370000e-01f, 3.507680000e-01f, 3.476990000e-01f, 3.446300000e-01f, 3.415610000e-01f, 3.384920000e-01f,
                3.997400000e-01f, 3.963080000e-01f, 3.928760000e-01f, 3.894440000e-01f, 3.860120000e-01f, 3.825800000e-01f, 3.791480000e-01f,
                2.716340000e-01f, 2.692250000e-01f, 2.668160000e-01f, 2.644070000e-01f, 2.619980000e-01f, 2.595890000e-01f, 2.571800000e-01f,
                1.381050000e-01f, 1.368400000e-01f, 1.355750000e-01f, 1.343100000e-01f, 1.330450000e-01f, 1.317800000e-01f, 1.305150000e-01f,
                1.709400000e-01f, 1.695540000e-01f, 1.681680000e-01f, 1.667820000e-01f, 1.653960000e-01f, 1.640100000e-01f, 1.626240000e-01f,
                3.473030000e-01f, 3.444100000e-01f, 3.415170000e-01f, 3.386240000e-01f, 3.357310000e-01f, 3.328380000e-01f, 3.299450000e-01f,
                5.282420000e-01f, 5.237210000e-01f, 5.192000000e-01f, 5.146790000e-01f, 5.101580000e-01f, 5.056370000e-01f, 5.011160000e-01f,
                5.710760000e-01f, 5.661920000e-01f, 5.613080000e-01f, 5.564240000e-01f, 5.515400000e-01f, 5.466560000e-01f, 5.417720000e-01f,
                6.139100000e-01f, 6.086630000e-01f, 6.034160000e-01f, 5.981690000e-01f, 5.929220000e-01f, 5.876750000e-01f, 5.824280000e-01f,
                6.567440000e-01f, 6.511340000e-01f, 6.455240000e-01f, 6.399140000e-01f, 6.343040000e-01f, 6.286940000e-01f, 6.230840000e-01f,
                6.995780000e-01f, 6.936050000e-01f, 6.876320000e-01f, 6.816590000e-01f, 6.756860000e-01f, 6.697130000e-01f, 6.637400000e-01f,
                7.424120000e-01f, 7.360760000e-01f, 7.297400000e-01f, 7.234040000e-01f, 7.170680000e-01f, 7.107320000e-01f, 7.043960000e-01f,
                7.852460000e-01f, 7.785470000e-01f, 7.718480000e-01f, 7.651490000e-01f, 7.584500000e-01f, 7.517510000e-01f, 7.450520000e-01f,
                8.280800000e-01f, 8.210180000e-01f, 8.139560000e-01f, 8.068940000e-01f, 7.998320000e-01f, 7.927700000e-01f, 7.857080000e-01f,
                8.709140000e-01f, 8.634890000e-01f, 8.560640000e-01f, 8.486390000e-01f, 8.412140000e-01f, 8.337890000e-01f, 8.263640000e-01f,
                5.764330000e-01f, 5.713620000e-01f, 5.662910000e-01f, 5.612200000e-01f, 5.561490000e-01f, 5.510780000e-01f, 5.460070000e-01f,
                2.858460000e-01f, 2.832500000e-01f, 2.806540000e-01f, 2.780580000e-01f, 2.754620000e-01f, 2.728660000e-01f, 2.702700000e-01f,
            };

            float[] x_actual = x.ToFloatArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
