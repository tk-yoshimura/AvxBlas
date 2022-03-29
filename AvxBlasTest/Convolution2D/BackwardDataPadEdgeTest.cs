using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection2DTest {
    [TestClass]
    public class BackwardDataPadEdgeTest {
        [TestMethod]
        public void SBackwardDataPadEdgeTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih) in new (uint, uint)[] { (1, 1), (1, 2), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    foreach ((uint kw, uint kh) in new (uint, uint)[] { (1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (7, 7) }) {
                        uint ow = iw, oh = ih;

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

                            Convolution2D.BackwardData(n, ic, oc, iw, ih, kw, kh, PadMode.Edge, y_tensor, w_tensor, x_tensor);

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
            int outw = iw, outh = ih;

            if (y.Width != outw || y.Height != outh) {
                throw new ArgumentException("mismatch shape");
            }

            Map2D x = new(inchannels, iw, ih, batch);

            for (int th = 0; th < batch; th++) {
                for (int ky = 0; ky < kh; ky++) {
                    for (int kx = 0; kx < kw; kx++) {
                        for (int oy = 0; oy < outh; oy++) {
                            int iy = Math.Min(ih - 1, Math.Max(0, ky + oy - kh / 2));

                            for (int ox = 0; ox < outw; ox++) {
                                int ix = Math.Min(iw - 1, Math.Max(0, kx + ox - kw / 2));

                                for (int outch = 0; outch < outchannels; outch++) {
                                    double v = y[outch, ox, oy, th];

                                    for (int inch = 0; inch < inchannels; inch++) {
                                        x[inch, ix, iy, th] += v * w[inch, kx, ky, outch];
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
                2.843500000e-02f, 2.814900000e-02f, 2.786300000e-02f, 2.757700000e-02f, 2.729100000e-02f, 2.700500000e-02f, 2.671900000e-02f,
                5.706800000e-02f, 5.654000000e-02f, 5.601200000e-02f, 5.548400000e-02f, 5.495600000e-02f, 5.442800000e-02f, 5.390000000e-02f,
                9.990200000e-02f, 9.901100000e-02f, 9.812000000e-02f, 9.722900000e-02f, 9.633800000e-02f, 9.544700000e-02f, 9.455600000e-02f,
                1.427360000e-01f, 1.414820000e-01f, 1.402280000e-01f, 1.389740000e-01f, 1.377200000e-01f, 1.364660000e-01f, 1.352120000e-01f,
                1.855700000e-01f, 1.839530000e-01f, 1.823360000e-01f, 1.807190000e-01f, 1.791020000e-01f, 1.774850000e-01f, 1.758680000e-01f,
                2.284040000e-01f, 2.264240000e-01f, 2.244440000e-01f, 2.224640000e-01f, 2.204840000e-01f, 2.185040000e-01f, 2.165240000e-01f,
                2.712380000e-01f, 2.688950000e-01f, 2.665520000e-01f, 2.642090000e-01f, 2.618660000e-01f, 2.595230000e-01f, 2.571800000e-01f,
                3.140720000e-01f, 3.113660000e-01f, 3.086600000e-01f, 3.059540000e-01f, 3.032480000e-01f, 3.005420000e-01f, 2.978360000e-01f,
                3.569060000e-01f, 3.538370000e-01f, 3.507680000e-01f, 3.476990000e-01f, 3.446300000e-01f, 3.415610000e-01f, 3.384920000e-01f,
                3.997400000e-01f, 3.963080000e-01f, 3.928760000e-01f, 3.894440000e-01f, 3.860120000e-01f, 3.825800000e-01f, 3.791480000e-01f,
                4.425740000e-01f, 4.387790000e-01f, 4.349840000e-01f, 4.311890000e-01f, 4.273940000e-01f, 4.235990000e-01f, 4.198040000e-01f,
                4.854080000e-01f, 4.812500000e-01f, 4.770920000e-01f, 4.729340000e-01f, 4.687760000e-01f, 4.646180000e-01f, 4.604600000e-01f,
                4.920190000e-01f, 4.876190000e-01f, 4.832190000e-01f, 4.788190000e-01f, 4.744190000e-01f, 4.700190000e-01f, 4.656190000e-01f,
                6.072990000e-01f, 6.022940000e-01f, 5.972890000e-01f, 5.922840000e-01f, 5.872790000e-01f, 5.822740000e-01f, 5.772690000e-01f,
                6.139100000e-01f, 6.086630000e-01f, 6.034160000e-01f, 5.981690000e-01f, 5.929220000e-01f, 5.876750000e-01f, 5.824280000e-01f,
                6.567440000e-01f, 6.511340000e-01f, 6.455240000e-01f, 6.399140000e-01f, 6.343040000e-01f, 6.286940000e-01f, 6.230840000e-01f,
                6.995780000e-01f, 6.936050000e-01f, 6.876320000e-01f, 6.816590000e-01f, 6.756860000e-01f, 6.697130000e-01f, 6.637400000e-01f,
                7.424120000e-01f, 7.360760000e-01f, 7.297400000e-01f, 7.234040000e-01f, 7.170680000e-01f, 7.107320000e-01f, 7.043960000e-01f,
                7.852460000e-01f, 7.785470000e-01f, 7.718480000e-01f, 7.651490000e-01f, 7.584500000e-01f, 7.517510000e-01f, 7.450520000e-01f,
                8.280800000e-01f, 8.210180000e-01f, 8.139560000e-01f, 8.068940000e-01f, 7.998320000e-01f, 7.927700000e-01f, 7.857080000e-01f,
                8.709140000e-01f, 8.634890000e-01f, 8.560640000e-01f, 8.486390000e-01f, 8.412140000e-01f, 8.337890000e-01f, 8.263640000e-01f,
                9.137480000e-01f, 9.059600000e-01f, 8.981720000e-01f, 8.903840000e-01f, 8.825960000e-01f, 8.748080000e-01f, 8.670200000e-01f,
                9.565820000e-01f, 9.484310000e-01f, 9.402800000e-01f, 9.321290000e-01f, 9.239780000e-01f, 9.158270000e-01f, 9.076760000e-01f,
                9.994160000e-01f, 9.909020000e-01f, 9.823880000e-01f, 9.738740000e-01f, 9.653600000e-01f, 9.568460000e-01f, 9.483320000e-01f,
                1.042250000e+00f, 1.033373000e+00f, 1.024496000e+00f, 1.015619000e+00f, 1.006742000e+00f, 9.978650000e-01f, 9.889880000e-01f,
                1.026839000e+00f, 1.017720000e+00f, 1.008601000e+00f, 9.994820000e-01f, 9.903630000e-01f, 9.812440000e-01f, 9.721250000e-01f,
            };

            float[] x_actual = x.ToFloatArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
