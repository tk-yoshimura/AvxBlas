using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.AffineTest {
    [TestClass]
    public class DotmulTest {
        [TestMethod]
        public void SDotmulTest() {
            Random random = new(1234);

            foreach (uint na in new uint[] {
                0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u }) {

                foreach (uint nb in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 20u, 23u, 24u, 25u, 28u, 
                        31u, 32u, 33u, 39u, 40u, 41u, 47u, 48u, 49u, 55u, 56u, 57u,
                        63u, 64u, 65u, 71u, 72u, 73u, 79u, 80u, 81u, 87u, 88u, 89u,
                        127u, 128u, 129u, 255u, 256u, 257u }) {

                        float[] a = (new float[na * stride]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                        float[] b = (new float[nb * stride]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                        float[] t = new float[na * nb + 4];

                        for (int i = 0; i < na; i++) {
                            for (int j = 0; j < nb; j++) {
                                double s = 0;

                                for (int k = 0; k < stride; k++) {
                                    s += (double)a[k + stride * i] * (double)b[k + stride * j];
                                }

                                t[j + nb * i] = (float)s;
                            }
                        }

                        Array<float> y = new(na * nb + 4);

                        Affine.Dotmul(na, nb, stride, a, b, y);

                        CollectionAssert.AreEqual(t, (float[])y, $"NG: na{na} nb{nb} stride{stride}");

                        Console.WriteLine($"OK: na{na} nb{nb} stride{stride}");
                    }
                }
            }
        }

        [TestMethod]
        public void DDotmulTest() {
            Random random = new(1234);

            foreach (uint na in new uint[] {
                0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u }) {

                foreach (uint nb in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 20u, 23u, 24u, 25u, 28u, 
                        31u, 32u, 33u, 39u, 40u, 41u, 47u, 48u, 49u, 55u, 56u, 57u,
                        63u, 64u, 65u, 71u, 72u, 73u, 79u, 80u, 81u, 87u, 88u, 89u,
                        127u, 128u, 129u, 255u, 256u, 257u }) {

                        double[] a = (new double[na * stride]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                        double[] b = (new double[nb * stride]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();
                        double[] t = new double[na * nb + 4];

                        for (int i = 0; i < na; i++) {
                            for (int j = 0; j < nb; j++) {
                                double s = 0;

                                for (int k = 0; k < stride; k++) {
                                    s += a[k + stride * i] * b[k + stride * j];
                                }

                                t[j + nb * i] = s;
                            }
                        }

                        Array<double> y = new(na * nb + 4);

                        Affine.Dotmul(na, nb, stride, a, b, y);

                        CollectionAssert.AreEqual(t, (double[])y, $"NG: na{na} nb{nb} stride{stride}");

                        Console.WriteLine($"OK: na{na} nb{nb} stride{stride}");
                    }
                }
            }
        }
    }
}
