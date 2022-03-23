using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Connection1DTest {
    [TestClass]
    public class BackwardFilterPadZeroTest {
        [TestMethod]
        public void SBackwardFilterPadZeroTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach (uint iw in new int[] { 1, 2, 3, 4, 5, 8, 15, 16, 17, 28, 30, 32 }) {
                    foreach (uint kw in new int[] { 3, 5, 7 }) {
                        uint ow = iw;

                        foreach (uint ic in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                            foreach (uint oc in new int[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                                float[] xval = (new float[iw * ic * n]).Select((_, idx) => idx * 1e-3f).ToArray();
                                float[] gyval = (new float[ow * oc * n]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

                                Map1D x = new((int)ic, (int)iw, (int)n, xval);
                                Map1D gy = new((int)oc, (int)ow, (int)n, gyval);

                                Filter1D gw = Reference(x, gy, (int)kw);

                                Array<float> x_tensor = xval;
                                Array<float> gy_tensor = gyval;

                                Array<float> gw_tensor = new(ic * oc * kw);

                                Convolution1D.BackwardFilter(n, ic, oc, iw, kw, PadMode.Zero, x_tensor, gy_tensor, gw_tensor);

                                float[] gw_expect = gw.ToFloatArray();
                                float[] gw_actual = gw_tensor;

                                CollectionAssert.AreEqual(xval, (float[])x_tensor);
                                CollectionAssert.AreEqual(gyval, (float[])gy_tensor);

                                AssertError.Tolerance(gw_expect, gw_actual, 1e-6f, 1e-4f, ref max_err, $"NG {ic},{oc},{kw},{iw},{n}");

                                Console.WriteLine($"OK: {ic},{oc},{kw},{iw},{n}");
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Filter1D Reference(Map1D x, Map1D gy, int kwidth) {
            int inchannels = x.Channels, outchannels = gy.Channels, batch = x.Batch;
            int inw = x.Width, outw = gy.Width;

            if (outw != inw) {
                throw new ArgumentException("mismatch shape");
            }

            Filter1D w = new(inchannels, kwidth, outchannels);

            for (int kx = 0; kx < kwidth; kx++) {
                for (int th = 0; th < batch; th++) {
                    for (int ox = 0; ox < outw; ox++) {
                        int ix = kx + ox - kwidth / 2;
                        if (ix < 0 || ix >= inw) {
                            continue;
                        }

                        for (int inch, outch = 0; outch < outchannels; outch++) {
                            for (inch = 0; inch < inchannels; inch++) {
                                w[inch, kx, outch] += x[inch, ix, th] * gy[outch, ox, th];
                            }
                        }
                    }
                }
            }

            return w;
        }

        [TestMethod]
        public void ReferenceTest() {
            int inchannels = 7, outchannels = 11, kwidth = 3, inwidth = 13;
            int outwidth = inwidth;

            float[] xval = (new float[inwidth * inchannels]).Select((_, idx) => idx * 1e-3f).ToArray();
            float[] gyval = (new float[outwidth * outchannels]).Select((_, idx) => idx * 1e-3f).Reverse().ToArray();

            Map1D x = new(inchannels, inwidth, 1, xval);
            Map1D gy = new(outchannels, outwidth, 1, gyval);

            Filter1D gw = Reference(x, gy, kwidth);

            float[] gw_expect = {
                2.156000000e-02f, 2.240600000e-02f, 2.325200000e-02f, 2.409800000e-02f, 2.494400000e-02f, 2.579000000e-02f, 2.663600000e-02f,
                2.748200000e-02f, 2.847000000e-02f, 2.945800000e-02f, 3.044600000e-02f, 3.143400000e-02f, 3.242200000e-02f, 3.341000000e-02f,
                3.348800000e-02f, 3.446600000e-02f, 3.544400000e-02f, 3.642200000e-02f, 3.740000000e-02f, 3.837800000e-02f, 3.935600000e-02f,
                2.109800000e-02f, 2.193200000e-02f, 2.276600000e-02f, 2.360000000e-02f, 2.443400000e-02f, 2.526800000e-02f, 2.610200000e-02f,
                2.693600000e-02f, 2.791100000e-02f, 2.888600000e-02f, 2.986100000e-02f, 3.083600000e-02f, 3.181100000e-02f, 3.278600000e-02f,
                3.294200000e-02f, 3.390800000e-02f, 3.487400000e-02f, 3.584000000e-02f, 3.680600000e-02f, 3.777200000e-02f, 3.873800000e-02f,
                2.063600000e-02f, 2.145800000e-02f, 2.228000000e-02f, 2.310200000e-02f, 2.392400000e-02f, 2.474600000e-02f, 2.556800000e-02f,
                2.639000000e-02f, 2.735200000e-02f, 2.831400000e-02f, 2.927600000e-02f, 3.023800000e-02f, 3.120000000e-02f, 3.216200000e-02f,
                3.239600000e-02f, 3.335000000e-02f, 3.430400000e-02f, 3.525800000e-02f, 3.621200000e-02f, 3.716600000e-02f, 3.812000000e-02f,
                2.017400000e-02f, 2.098400000e-02f, 2.179400000e-02f, 2.260400000e-02f, 2.341400000e-02f, 2.422400000e-02f, 2.503400000e-02f,
                2.584400000e-02f, 2.679300000e-02f, 2.774200000e-02f, 2.869100000e-02f, 2.964000000e-02f, 3.058900000e-02f, 3.153800000e-02f,
                3.185000000e-02f, 3.279200000e-02f, 3.373400000e-02f, 3.467600000e-02f, 3.561800000e-02f, 3.656000000e-02f, 3.750200000e-02f,
                1.971200000e-02f, 2.051000000e-02f, 2.130800000e-02f, 2.210600000e-02f, 2.290400000e-02f, 2.370200000e-02f, 2.450000000e-02f,
                2.529800000e-02f, 2.623400000e-02f, 2.717000000e-02f, 2.810600000e-02f, 2.904200000e-02f, 2.997800000e-02f, 3.091400000e-02f,
                3.130400000e-02f, 3.223400000e-02f, 3.316400000e-02f, 3.409400000e-02f, 3.502400000e-02f, 3.595400000e-02f, 3.688400000e-02f,
                1.925000000e-02f, 2.003600000e-02f, 2.082200000e-02f, 2.160800000e-02f, 2.239400000e-02f, 2.318000000e-02f, 2.396600000e-02f,
                2.475200000e-02f, 2.567500000e-02f, 2.659800000e-02f, 2.752100000e-02f, 2.844400000e-02f, 2.936700000e-02f, 3.029000000e-02f,
                3.075800000e-02f, 3.167600000e-02f, 3.259400000e-02f, 3.351200000e-02f, 3.443000000e-02f, 3.534800000e-02f, 3.626600000e-02f,
                1.878800000e-02f, 1.956200000e-02f, 2.033600000e-02f, 2.111000000e-02f, 2.188400000e-02f, 2.265800000e-02f, 2.343200000e-02f,
                2.420600000e-02f, 2.511600000e-02f, 2.602600000e-02f, 2.693600000e-02f, 2.784600000e-02f, 2.875600000e-02f, 2.966600000e-02f,
                3.021200000e-02f, 3.111800000e-02f, 3.202400000e-02f, 3.293000000e-02f, 3.383600000e-02f, 3.474200000e-02f, 3.564800000e-02f,
                1.832600000e-02f, 1.908800000e-02f, 1.985000000e-02f, 2.061200000e-02f, 2.137400000e-02f, 2.213600000e-02f, 2.289800000e-02f,
                2.366000000e-02f, 2.455700000e-02f, 2.545400000e-02f, 2.635100000e-02f, 2.724800000e-02f, 2.814500000e-02f, 2.904200000e-02f,
                2.966600000e-02f, 3.056000000e-02f, 3.145400000e-02f, 3.234800000e-02f, 3.324200000e-02f, 3.413600000e-02f, 3.503000000e-02f,
                1.786400000e-02f, 1.861400000e-02f, 1.936400000e-02f, 2.011400000e-02f, 2.086400000e-02f, 2.161400000e-02f, 2.236400000e-02f,
                2.311400000e-02f, 2.399800000e-02f, 2.488200000e-02f, 2.576600000e-02f, 2.665000000e-02f, 2.753400000e-02f, 2.841800000e-02f,
                2.912000000e-02f, 3.000200000e-02f, 3.088400000e-02f, 3.176600000e-02f, 3.264800000e-02f, 3.353000000e-02f, 3.441200000e-02f,
                1.740200000e-02f, 1.814000000e-02f, 1.887800000e-02f, 1.961600000e-02f, 2.035400000e-02f, 2.109200000e-02f, 2.183000000e-02f,
                2.256800000e-02f, 2.343900000e-02f, 2.431000000e-02f, 2.518100000e-02f, 2.605200000e-02f, 2.692300000e-02f, 2.779400000e-02f,
                2.857400000e-02f, 2.944400000e-02f, 3.031400000e-02f, 3.118400000e-02f, 3.205400000e-02f, 3.292400000e-02f, 3.379400000e-02f,
                1.694000000e-02f, 1.766600000e-02f, 1.839200000e-02f, 1.911800000e-02f, 1.984400000e-02f, 2.057000000e-02f, 2.129600000e-02f,
                2.202200000e-02f, 2.288000000e-02f, 2.373800000e-02f, 2.459600000e-02f, 2.545400000e-02f, 2.631200000e-02f, 2.717000000e-02f,
                2.802800000e-02f, 2.888600000e-02f, 2.974400000e-02f, 3.060200000e-02f, 3.146000000e-02f, 3.231800000e-02f, 3.317600000e-02f,
            };

            float[] gw_actual = gw.ToFloatArray();

            AssertError.Tolerance(gw_expect, gw_actual, 1e-7f, 1e-5f, $"mismatch value {inchannels},{outchannels},{kwidth},{inwidth}");
        }
    }
}
