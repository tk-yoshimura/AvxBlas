using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.Pool3DTest {
    [TestClass]
    public class AverageUnpoolingTest {
        [TestMethod]
        public void SAverageUnpoolingTest() {
            float max_err = 0;

            foreach (uint n in new int[] { 1, 2 }) {
                foreach ((uint iw, uint ih, uint id, uint sx, uint sy, uint sz) in new (uint, uint, uint, uint, uint, uint)[] {
                    (1, 1, 1, 1, 1, 1), (1, 1, 4, 1, 1, 2), (1, 4, 1, 1, 2, 1), (4, 1, 1, 2, 1, 1), (4, 3, 2, 2, 2, 3), (5, 8, 4, 2, 2, 2), (7, 16, 15, 2, 2, 3), (17, 6, 28, 4, 2, 2), (32, 31, 30, 2, 2, 3),
                    (1, 1, 1, 1, 1, 2), (1, 1, 4, 1, 2, 1), (1, 4, 1, 1, 1, 1), (4, 1, 1, 2, 2, 2), (4, 3, 2, 2, 1, 1), (5, 8, 4, 4, 2, 2), (7, 16, 15, 2, 2, 2), (17, 6, 28, 2, 2, 3), (32, 31, 30, 1, 1, 1)}) {

                    uint ow = (iw - 1) / sx + 1;
                    uint oh = (ih - 1) / sy + 1;
                    uint od = (id - 1) / sz + 1;

                    foreach ((uint kw, uint kh, uint kd) in new (uint, uint, uint)[] {
                        (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2), (2, 2, 4), (2, 4, 3), (4, 4, 4), (3, 2, 5), (2, 5, 3) }) {

                        foreach (uint c in new uint[] { 1, 2, 4, 5, 8, 10, 15, 16, 20, 32, 33 }) {

                            float[] dyval = (new float[c * ow * oh * od * n]).Select((_, idx) => (float)((idx + 1) * 4547 % 17)).Reverse().ToArray();

                            Map3D dy = new((int)c, (int)ow, (int)oh, (int)od, (int)n, dyval);
                            Map3D dx = Reference(dy, (int)iw, (int)sx, (int)kw, (int)ih, (int)sy, (int)kh, (int)id, (int)sz, (int)kd);

                            Array<float> dy_tensor = dyval;
                            Array<float> dx_tensor = new(c * iw * ih * id * n, zeroset: false);

                            Pool3D.AverageUnpooling(n, c, iw, ih, id, sx, sy, sz, kw, kh, kd, dy_tensor, dx_tensor);

                            float[] dx_expect = dx.ToFloatArray();
                            float[] dx_actual = dx_tensor;

                            CollectionAssert.AreEqual(dyval, (float[])dy_tensor);

                            AssertError.Tolerance(dx_expect, dx_actual, 1e-10f, 1e-5f, ref max_err, $"NG: {c},{iw},{ih},{id},{sx},{sy},{sz},{kw},{kh},{kd},{n}");

                            Console.WriteLine($"OK: {c},{iw},{ih},{id},{sx},{sy},{sz},{kw},{kh},{kd},{n}");
                        }
                    }
                }
            }

            Console.WriteLine($"maxerr:{max_err}");
        }

        public static Map3D Reference(Map3D dy, int iw, int sx, int kw, int ih, int sy, int kh, int id, int sz, int kd) {
            int channels = dy.Channels, batch = dy.Batch;
            int ow = (iw - 1) / sx + 1, oh = (ih - 1) / sy + 1, od = (id - 1) / sz + 1;

            if (dy.Width != ow || dy.Height != oh || dy.Depth != od) {
                throw new ArgumentException("mismatch shape");
            }

            Map3D dx = new(channels, iw, ih, id, batch);

            for (int th = 0; th < batch; th++) {
                for (int isz = 0, oz = 0; oz < od; isz += sz, oz++) {
                    for (int isy = 0, oy = 0; oy < oh; isy += sy, oy++) {
                        for (int isx = 0, ox = 0; ox < ow; isx += sx, ox++) {
                            for (int c = 0; c < channels; c++) {
                                for (int kz = 0, iz = isz + kz; kz < kd && iz < id; kz++, iz = isz + kz) {
                                    for (int ky = 0, iy = isy + ky; ky < kh && iy < ih; ky++, iy = isy + ky) {
                                        for (int kx = 0, ix = isx + kx; kx < kw && ix < iw; kx++, ix = isx + kx) {
                                            dx[c, ix, iy, iz, th] += dy[c, ox, oy, oz, th];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int th = 0; th < batch; th++) {
                for (int iz = 0; iz < id; iz++) {
                    for (int iy = 0; iy < ih; iy++) {
                        for (int ix = 0; ix < iw; ix++) {
                            for (int c = 0; c < channels; c++) {
                                dx[c, ix, iy, iz, th] /= kw * kh * kd;
                            }
                        }
                    }
                }
            }

            return dx;
        }
    }
}
