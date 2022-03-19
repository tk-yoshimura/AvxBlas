using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection1DTest {
    [TestClass]
    public class BackwardDataPadEdgeTest {
        [TestMethod]
        public void SForwardTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    foreach (uint kw in new int[] { 3, 5, 7 }) {
                        uint ow = iw;

                        foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                            foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                                float[] yval = (new float[ow * oc * n]).Select((_, idx) => (idx + 1) * 1e-3f).ToArray();
                                float[] wval = (new float[kw * ic * oc]).Select((_, idx) => (idx + 1) * 1e-3f).Reverse().ToArray();

                                Map1D y = new((int)oc, (int)ow, (int)n, yval);
                                Filter1D w = new((int)ic, (int)oc, (int)kw, wval);

                                Map1D x = Reference(y, w, (int)iw, (int)kw);

                                Array<float> y_tensor = yval;
                                Array<float> w_tensor = wval;
                                Array<float> x_tensor = new(ic * iw * n);

                                Convolution1D.BackwardData(n, ic, oc, iw, kw, PadMode.Edge, y_tensor, w_tensor, x_tensor);
                                
                                float[] x_expect = x.ToFloatArray();
                                float[] x_actual = x_tensor;

                                CollectionAssert.AreEqual(yval, (float[])y_tensor);
                                CollectionAssert.AreEqual(wval, (float[])w_tensor);

                                AssertError.Tolerance(x_expect, x_actual, 1e-8f, 1e-6f, ref max_err, $"NG: {ic},{oc},{kw},{iw},{n}");

                                Console.WriteLine($"OK: {ic},{oc},{kw},{iw},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map1D Reference(Map1D y, Filter1D w, int inw, int kwidth) {
            int inchannels = w.InChannels, outchannels = w.OutChannels, batch = y.Batch;
            int outw = inw;

            if (y.Width != outw) {
                throw new ArgumentException("mismatch shape");
            }

            Map1D x = new(inchannels, inw, batch);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        int ix = Math.Min(inw - 1, Math.Max(0, kx + ox - kwidth / 2));

                        for (int outch = 0; outch < outchannels; outch++) {
                            double v = y[outch, ox, th];

                            for (int inch = 0; inch < inchannels; inch++) {
                                x[inch, ix, th] += v * w[inch, outch, kx];
                            }
                        }
                    }
                }
            }

            return x;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13, batch = 2;
            int ow = inwidth;

            float[] yval = (new float[ow * outchannels * batch]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] wval = (new float[kwidth * outchannels * inchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D y = new(outchannels, ow, batch, yval);
            Filter1D w = new(inchannels, outchannels, kwidth, wval);

            Map1D x = Reference(y, w, inwidth, kwidth);

            float[] x_expect = {
                4.922500000e-02f, 4.893900000e-02f, 4.865300000e-02f, 4.836700000e-02f, 4.808100000e-02f, 4.779500000e-02f, 4.750900000e-02f,
                7.862800000e-02f, 7.810000000e-02f, 7.757200000e-02f, 7.704400000e-02f, 7.651600000e-02f, 7.598800000e-02f, 7.546000000e-02f,
                1.214620000e-01f, 1.205710000e-01f, 1.196800000e-01f, 1.187890000e-01f, 1.178980000e-01f, 1.170070000e-01f, 1.161160000e-01f,
                1.642960000e-01f, 1.630420000e-01f, 1.617880000e-01f, 1.605340000e-01f, 1.592800000e-01f, 1.580260000e-01f, 1.567720000e-01f,
                2.071300000e-01f, 2.055130000e-01f, 2.038960000e-01f, 2.022790000e-01f, 2.006620000e-01f, 1.990450000e-01f, 1.974280000e-01f,
                2.499640000e-01f, 2.479840000e-01f, 2.460040000e-01f, 2.440240000e-01f, 2.420440000e-01f, 2.400640000e-01f, 2.380840000e-01f,
                2.927980000e-01f, 2.904550000e-01f, 2.881120000e-01f, 2.857690000e-01f, 2.834260000e-01f, 2.810830000e-01f, 2.787400000e-01f,
                3.356320000e-01f, 3.329260000e-01f, 3.302200000e-01f, 3.275140000e-01f, 3.248080000e-01f, 3.221020000e-01f, 3.193960000e-01f,
                3.784660000e-01f, 3.753970000e-01f, 3.723280000e-01f, 3.692590000e-01f, 3.661900000e-01f, 3.631210000e-01f, 3.600520000e-01f,
                4.213000000e-01f, 4.178680000e-01f, 4.144360000e-01f, 4.110040000e-01f, 4.075720000e-01f, 4.041400000e-01f, 4.007080000e-01f,
                4.641340000e-01f, 4.603390000e-01f, 4.565440000e-01f, 4.527490000e-01f, 4.489540000e-01f, 4.451590000e-01f, 4.413640000e-01f,
                5.069680000e-01f, 5.028100000e-01f, 4.986520000e-01f, 4.944940000e-01f, 4.903360000e-01f, 4.861780000e-01f, 4.820200000e-01f,
                2.941290000e-01f, 2.897290000e-01f, 2.853290000e-01f, 2.809290000e-01f, 2.765290000e-01f, 2.721290000e-01f, 2.677290000e-01f,
                8.483090000e-01f, 8.433040000e-01f, 8.382990000e-01f, 8.332940000e-01f, 8.282890000e-01f, 8.232840000e-01f, 8.182790000e-01f,
                6.354700000e-01f, 6.302230000e-01f, 6.249760000e-01f, 6.197290000e-01f, 6.144820000e-01f, 6.092350000e-01f, 6.039880000e-01f,
                6.783040000e-01f, 6.726940000e-01f, 6.670840000e-01f, 6.614740000e-01f, 6.558640000e-01f, 6.502540000e-01f, 6.446440000e-01f,
                7.211380000e-01f, 7.151650000e-01f, 7.091920000e-01f, 7.032190000e-01f, 6.972460000e-01f, 6.912730000e-01f, 6.853000000e-01f,
                7.639720000e-01f, 7.576360000e-01f, 7.513000000e-01f, 7.449640000e-01f, 7.386280000e-01f, 7.322920000e-01f, 7.259560000e-01f,
                8.068060000e-01f, 8.001070000e-01f, 7.934080000e-01f, 7.867090000e-01f, 7.800100000e-01f, 7.733110000e-01f, 7.666120000e-01f,
                8.496400000e-01f, 8.425780000e-01f, 8.355160000e-01f, 8.284540000e-01f, 8.213920000e-01f, 8.143300000e-01f, 8.072680000e-01f,
                8.924740000e-01f, 8.850490000e-01f, 8.776240000e-01f, 8.701990000e-01f, 8.627740000e-01f, 8.553490000e-01f, 8.479240000e-01f,
                9.353080000e-01f, 9.275200000e-01f, 9.197320000e-01f, 9.119440000e-01f, 9.041560000e-01f, 8.963680000e-01f, 8.885800000e-01f,
                9.781420000e-01f, 9.699910000e-01f, 9.618400000e-01f, 9.536890000e-01f, 9.455380000e-01f, 9.373870000e-01f, 9.292360000e-01f,
                1.020976000e+00f, 1.012462000e+00f, 1.003948000e+00f, 9.954340000e-01f, 9.869200000e-01f, 9.784060000e-01f, 9.698920000e-01f,
                1.063810000e+00f, 1.054933000e+00f, 1.046056000e+00f, 1.037179000e+00f, 1.028302000e+00f, 1.019425000e+00f, 1.010548000e+00f,
                6.087290000e-01f, 5.996100000e-01f, 5.904910000e-01f, 5.813720000e-01f, 5.722530000e-01f, 5.631340000e-01f, 5.540150000e-01f,
            };

            float[] x_actual = x.ToFloatArray();

            AssertError.Tolerance(x_expect, x_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{kwidth},{inwidth},{batch}");
        }
    }
}
