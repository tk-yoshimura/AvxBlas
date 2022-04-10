using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Upsample3DTest {
    [TestClass]
    public class LinearTest {
        [TestMethod]
        public void SLinearTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih, uint id) in new (uint, uint, uint)[] {
                    (1, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1), (5, 2, 8), (3, 9, 4), (12, 16, 15) }) {
            
                    uint ow = iw * 2, oh = ih * 2, od = id * 2;
                    foreach (uint c in new uint[] { 1, 2, 3, 4, 5, 8, 10, 15, 16, 20, 31, 32, 33, 39, 40, 41, 47, 48, 49, 55, 56, 57, 63, 64, 65 }) {
                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();
            
                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);
            
                        Map3D y = Reference(x);
            
                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);
            
                        Upsample3D.LinearX2(n, c, iw, ih, id, x_tensor, y_tensor);
            
                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;
            
                        CollectionAssert.AreEqual(xval, (float[])x_tensor);
            
                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{n}");
            
                        Console.WriteLine($"OK: {c},{iw},{ih},{id},{n}");
            
                    }
                }
            }

            foreach (uint n in new int[] { 1, 2, 3, 4 }) {
                foreach ((uint ih, uint id) in new (uint, uint)[] { (1, 1), (1, 4), (4, 1), (4, 3), (5, 8), (16, 15), (17, 28), (32, 30) }) {
                    for (uint iw = 1; iw <= 65; iw++) {
                        const uint c = 1;

                        uint ow = iw * 2, oh = ih * 2, od = id * 2;

                        float[] xval = (new float[c * iw * ih * id * n]).Select((_, idx) => (float)(idx * 4547 % 17 + idx * 631 % 23)).ToArray();

                        Map3D x = new((int)c, (int)iw, (int)ih, (int)id, (int)n, xval);

                        Map3D y = Reference(x);

                        Array<float> x_tensor = xval;
                        Array<float> y_tensor = new(c * ow * oh * od * n, zeroset: false);

                        Upsample3D.LinearX2(n, c, iw, ih, id, x_tensor, y_tensor);

                        float[] y_expect = y.ToFloatArray();
                        float[] y_actual = y_tensor;

                        CollectionAssert.AreEqual(xval, (float[])x_tensor);

                        AssertError.Tolerance(y_expect, y_actual, 1e-7f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{n}");

                        Console.WriteLine($"OK: {c},{iw},{ih},{id},{n}");
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D x) {
            int inw = x.Width, inh = x.Height, ind = x.Depth, channels = x.Channels, batch = x.Batch;

            int outw = inw * 2, outh = inh * 2, outd = ind * 2;
            Map3D y = new Map3D(channels, outw, outh, outd, batch);

            for (int th = 0; th < batch; th++) {
                for (int ix, iy, iz = 0; iz < ind; iz++) {
                    for (iy = 0; iy < inh; iy++) {
                        for (ix = 0; ix < inw; ix++) {
                            int x0 = ix, xm = Math.Max(0, ix - 1), xp = Math.Min(inw - 1, ix + 1);
                            int y0 = iy, ym = Math.Max(0, iy - 1), yp = Math.Min(inh - 1, iy + 1);
                            int z0 = iz, zm = Math.Max(0, iz - 1), zp = Math.Min(ind - 1, iz + 1);

                            for (int f = 0; f < channels; f++) {
                                double vc = x[f, x0, y0, z0, th] * 8;

                                double vxm = x[f, xm, y0, z0, th] * 4;
                                double vxp = x[f, xp, y0, z0, th] * 4;
                                double vym = x[f, x0, ym, z0, th] * 4;
                                double vyp = x[f, x0, yp, z0, th] * 4;
                                double vzm = x[f, x0, y0, zm, th] * 4;
                                double vzp = x[f, x0, y0, zp, th] * 4;

                                double vxmym = x[f, xm, ym, z0, th] * 2;
                                double vxpym = x[f, xp, ym, z0, th] * 2;
                                double vxmyp = x[f, xm, yp, z0, th] * 2;
                                double vxpyp = x[f, xp, yp, z0, th] * 2;
                                double vymzm = x[f, x0, ym, zm, th] * 2;
                                double vypzm = x[f, x0, yp, zm, th] * 2;
                                double vymzp = x[f, x0, ym, zp, th] * 2;
                                double vypzp = x[f, x0, yp, zp, th] * 2;
                                double vxmzm = x[f, xm, y0, zm, th] * 2;
                                double vxpzm = x[f, xp, y0, zm, th] * 2;
                                double vxmzp = x[f, xm, y0, zp, th] * 2;
                                double vxpzp = x[f, xp, y0, zp, th] * 2;

                                double vxmymzm = x[f, xm, ym, zm, th];
                                double vxpymzm = x[f, xp, ym, zm, th];
                                double vxmypzm = x[f, xm, yp, zm, th];
                                double vxpypzm = x[f, xp, yp, zm, th];
                                double vxmymzp = x[f, xm, ym, zp, th];
                                double vxpymzp = x[f, xp, ym, zp, th];
                                double vxmypzp = x[f, xm, yp, zp, th];
                                double vxpypzp = x[f, xp, yp, zp, th];

                                y[f, ix * 2, iy * 2, iz * 2, th] = (vc + vxm + vym + vzm + vxmym + vymzm + vxmzm + vxmymzm) / 27;
                                y[f, ix * 2 + 1, iy * 2, iz * 2, th] = (vc + vxp + vym + vzm + vxpym + vymzm + vxpzm + vxpymzm) / 27;
                                y[f, ix * 2, iy * 2 + 1, iz * 2, th] = (vc + vxm + vyp + vzm + vxmyp + vypzm + vxmzm + vxmypzm) / 27;
                                y[f, ix * 2 + 1, iy * 2 + 1, iz * 2, th] = (vc + vxp + vyp + vzm + vxpyp + vypzm + vxpzm + vxpypzm) / 27;
                                y[f, ix * 2, iy * 2, iz * 2 + 1, th] = (vc + vxm + vym + vzp + vxmym + vymzp + vxmzp + vxmymzp) / 27;
                                y[f, ix * 2 + 1, iy * 2, iz * 2 + 1, th] = (vc + vxp + vym + vzp + vxpym + vymzp + vxpzp + vxpymzp) / 27;
                                y[f, ix * 2, iy * 2 + 1, iz * 2 + 1, th] = (vc + vxm + vyp + vzp + vxmyp + vypzp + vxmzp + vxmypzp) / 27;
                                y[f, ix * 2 + 1, iy * 2 + 1, iz * 2 + 1, th] = (vc + vxp + vyp + vzp + vxpyp + vypzp + vxpzp + vxpypzp) / 27;

                            }
                        }
                    }
                }
            }

            return y;
        }
    }
}
