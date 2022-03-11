using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.TransformTest {
    [TestClass]
    public class TransposeTest {
        [TestMethod]
        public void STransposeTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                0u, 1u, 2u, 3u }) {

                foreach (uint r in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 8u, 9u, 15u, 16u, 17u }) {

                    foreach (uint s in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 8u, 9u, 15u, 16u, 17u }) {

                        foreach (uint stride in new uint[] {
                            0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 11u, 12u, 13u, 14u,
                            15u, 16u, 17u, 20u, 23u, 24u, 25u, 28u,
                            31u, 32u, 33u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                            float[] x = (new float[checked(n * r * s * stride)]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                            float[] t = new float[checked(n * r * s * stride) + 4];

                            for (uint th = 0; th < n; th++) { 
                                for (uint k = 0; k < r; k++) { 
                                    for (uint j = 0; j < s; j++) { 
                                        for (uint i = 0; i < stride; i++) {
                                            t[i + stride * (k + r * (j + s * th))] = x[i + stride * (j + s * (k + r * th))];
                                        }
                                    }
                                }
                            }

                            Array<float> y = new(checked(n * r * s * stride) + 4);

                            Transform.Transpose(n, r, s, stride, x, y);

                            CollectionAssert.AreEqual(t, (float[])y, $"NG: na{n} r{r} s{s} stride{stride}");

                            Console.WriteLine($"OK: na{n} r{r} s{s} stride{stride}");
                        }
                    }
                }
            }
        }
    }
}
