using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection2DTest {
    [TestClass]
    public class ForwardPadNoneTest {
        [TestMethod]
        public void SForwardTest() {
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

                            float[] xval = (new float[ic * iw * ih * n]).Select((_, idx) => (idx + 1) * 1e-3f).ToArray();
                            float[] wval = (new float[ic * oc * kw * kh]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                            Map2D x = new((int)ic, (int)iw, (int)ih, (int)n, xval);
                            Filter2D w = new((int)ic, (int)kw, (int)kh, (int)oc, wval);

                            Map2D y = Reference(x, w, (int)kw, (int)kh);

                            Array<float> x_tensor = xval;
                            Array<float> w_tensor = wval;
                            Array<float> y_tensor = new(oc * ow * oh * n, zeroset: false);

                            Convolution2D.Forward(n, ic, oc, iw, ih, kw, kh, PadMode.None, x_tensor, w_tensor, y_tensor);

                            float[] y_expect = y.ToFloatArray();
                            float[] y_actual = y_tensor;

                            CollectionAssert.AreEqual(xval, (float[])x_tensor);
                            CollectionAssert.AreEqual(wval, (float[])w_tensor);

                            AssertError.Tolerance(y_expect, y_actual, 1e-8f, 1e-6f, ref max_err, $"NG: {ic},{oc},{iw},{ih},{kw},{kh},{n}");

                            Console.WriteLine($"OK: {ic},{oc},{iw},{ih},{kw},{kh},{n}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map2D Reference(Map2D x, Filter2D w, int kw, int kh) {
            int inchannels = x.Channels, outchannels = w.OutChannels, batch = x.Batch;
            int iw = x.Width, ow = iw - kw + 1, ih = x.Height, oh = ih - kh + 1;

            Map2D y = new(outchannels, ow, oh, batch);

            for (int th = 0; th < batch; th++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int oy = 0; oy < oh; oy++) {
                            for (int ox = 0; ox < ow; ox++) {
                                for (int outch = 0; outch < outchannels; outch++) {
                                    double sum = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        sum += x[inch, kx + ox, ky + oy, th] * w[inch, kx, ky, outch];
                                    }

                                    y[outch, ox, oy, th] = sum;
                                }
                            }
                        }
                    }
                }
            }

            return y;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11;
            int kwidth = 3, kheight = 5, inwidth = 9, inheight = 13, batch = 2;

            float[] xval = (new float[batch * inwidth * inheight * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * kheight * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map2D x = new(inchannels, inwidth, inheight, batch, xval);
            Filter2D w = new(inchannels, kwidth, kheight, outchannels, wval);

            Map2D y = Reference(x, w, kwidth, kheight);

            float[] y_expect = {
                4.543000000e-02f, 4.102000000e-02f, 3.661000000e-02f, 3.220000000e-02f, 2.779000000e-02f, 2.338000000e-02f, 1.897000000e-02f, 1.456000000e-02f, 1.015000000e-02f, 5.740000000e-03f, 1.330000000e-03f,
                7.777000000e-02f, 7.027300000e-02f, 6.277600000e-02f, 5.527900000e-02f, 4.778200000e-02f, 4.028500000e-02f, 3.278800000e-02f, 2.529100000e-02f, 1.779400000e-02f, 1.029700000e-02f, 2.800000000e-03f,
                1.101100000e-01f, 9.952600000e-02f, 8.894200000e-02f, 7.835800000e-02f, 6.777400000e-02f, 5.719000000e-02f, 4.660600000e-02f, 3.602200000e-02f, 2.543800000e-02f, 1.485400000e-02f, 4.270000000e-03f,
                1.424500000e-01f, 1.287790000e-01f, 1.151080000e-01f, 1.014370000e-01f, 8.776600000e-02f, 7.409500000e-02f, 6.042400000e-02f, 4.675300000e-02f, 3.308200000e-02f, 1.941100000e-02f, 5.740000000e-03f,
                1.747900000e-01f, 1.580320000e-01f, 1.412740000e-01f, 1.245160000e-01f, 1.077580000e-01f, 9.100000000e-02f, 7.424200000e-02f, 5.748400000e-02f, 4.072600000e-02f, 2.396800000e-02f, 7.210000000e-03f,
                2.071300000e-01f, 1.872850000e-01f, 1.674400000e-01f, 1.475950000e-01f, 1.277500000e-01f, 1.079050000e-01f, 8.806000000e-02f, 6.821500000e-02f, 4.837000000e-02f, 2.852500000e-02f, 8.680000000e-03f,
                2.394700000e-01f, 2.165380000e-01f, 1.936060000e-01f, 1.706740000e-01f, 1.477420000e-01f, 1.248100000e-01f, 1.018780000e-01f, 7.894600000e-02f, 5.601400000e-02f, 3.308200000e-02f, 1.015000000e-02f,
                2.718100000e-01f, 2.457910000e-01f, 2.197720000e-01f, 1.937530000e-01f, 1.677340000e-01f, 1.417150000e-01f, 1.156960000e-01f, 8.967700000e-02f, 6.365800000e-02f, 3.763900000e-02f, 1.162000000e-02f,
                3.041500000e-01f, 2.750440000e-01f, 2.459380000e-01f, 2.168320000e-01f, 1.877260000e-01f, 1.586200000e-01f, 1.295140000e-01f, 1.004080000e-01f, 7.130200000e-02f, 4.219600000e-02f, 1.309000000e-02f,
                3.364900000e-01f, 3.042970000e-01f, 2.721040000e-01f, 2.399110000e-01f, 2.077180000e-01f, 1.755250000e-01f, 1.433320000e-01f, 1.111390000e-01f, 7.894600000e-02f, 4.675300000e-02f, 1.456000000e-02f,
                3.688300000e-01f, 3.335500000e-01f, 2.982700000e-01f, 2.629900000e-01f, 2.277100000e-01f, 1.924300000e-01f, 1.571500000e-01f, 1.218700000e-01f, 8.659000000e-02f, 5.131000000e-02f, 1.603000000e-02f,
                4.658500000e-01f, 4.213090000e-01f, 3.767680000e-01f, 3.322270000e-01f, 2.876860000e-01f, 2.431450000e-01f, 1.986040000e-01f, 1.540630000e-01f, 1.095220000e-01f, 6.498100000e-02f, 2.044000000e-02f,
                4.981900000e-01f, 4.505620000e-01f, 4.029340000e-01f, 3.553060000e-01f, 3.076780000e-01f, 2.600500000e-01f, 2.124220000e-01f, 1.647940000e-01f, 1.171660000e-01f, 6.953800000e-02f, 2.191000000e-02f,
                5.305300000e-01f, 4.798150000e-01f, 4.291000000e-01f, 3.783850000e-01f, 3.276700000e-01f, 2.769550000e-01f, 2.262400000e-01f, 1.755250000e-01f, 1.248100000e-01f, 7.409500000e-02f, 2.338000000e-02f,
                5.628700000e-01f, 5.090680000e-01f, 4.552660000e-01f, 4.014640000e-01f, 3.476620000e-01f, 2.938600000e-01f, 2.400580000e-01f, 1.862560000e-01f, 1.324540000e-01f, 7.865200000e-02f, 2.485000000e-02f,
                5.952100000e-01f, 5.383210000e-01f, 4.814320000e-01f, 4.245430000e-01f, 3.676540000e-01f, 3.107650000e-01f, 2.538760000e-01f, 1.969870000e-01f, 1.400980000e-01f, 8.320900000e-02f, 2.632000000e-02f,
                6.275500000e-01f, 5.675740000e-01f, 5.075980000e-01f, 4.476220000e-01f, 3.876460000e-01f, 3.276700000e-01f, 2.676940000e-01f, 2.077180000e-01f, 1.477420000e-01f, 8.776600000e-02f, 2.779000000e-02f,
                6.598900000e-01f, 5.968270000e-01f, 5.337640000e-01f, 4.707010000e-01f, 4.076380000e-01f, 3.445750000e-01f, 2.815120000e-01f, 2.184490000e-01f, 1.553860000e-01f, 9.232300000e-02f, 2.926000000e-02f,
                6.922300000e-01f, 6.260800000e-01f, 5.599300000e-01f, 4.937800000e-01f, 4.276300000e-01f, 3.614800000e-01f, 2.953300000e-01f, 2.291800000e-01f, 1.630300000e-01f, 9.688000000e-02f, 3.073000000e-02f,
                7.245700000e-01f, 6.553330000e-01f, 5.860960000e-01f, 5.168590000e-01f, 4.476220000e-01f, 3.783850000e-01f, 3.091480000e-01f, 2.399110000e-01f, 1.706740000e-01f, 1.014370000e-01f, 3.220000000e-02f,
                7.569100000e-01f, 6.845860000e-01f, 6.122620000e-01f, 5.399380000e-01f, 4.676140000e-01f, 3.952900000e-01f, 3.229660000e-01f, 2.506420000e-01f, 1.783180000e-01f, 1.059940000e-01f, 3.367000000e-02f,
                7.892500000e-01f, 7.138390000e-01f, 6.384280000e-01f, 5.630170000e-01f, 4.876060000e-01f, 4.121950000e-01f, 3.367840000e-01f, 2.613730000e-01f, 1.859620000e-01f, 1.105510000e-01f, 3.514000000e-02f,
            };

            float[] y_actual = y.ToFloatArray();

            AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
