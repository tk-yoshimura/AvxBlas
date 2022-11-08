using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.AggregateTest {
    [TestClass]
    public class ArgMinTest {
        [TestMethod]
        public void SArgMinTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u }) {

                foreach (uint samples in new uint[] {
                    1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                    15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u,
                    63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u}) {

                        uint inlength = n * samples * stride + 4;
                        uint outlength = n * stride + 4;

                        float[] x = (new float[inlength]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                        int[] t = new int[outlength];

                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < stride; j++) {
                                float s = float.PositiveInfinity;
                                int index = 0;

                                for (int k = 0; k < samples; k++) {
                                    if (s > x[j + stride * (k + samples * i)]) {
                                        index = k;
                                        s = x[j + stride * (k + samples * i)];
                                    }
                                }

                                t[j + stride * i] = index;
                            }
                        }

                        Array<int> y = new(outlength);

                        Aggregate.ArgMin(n, samples, stride, x, y);

                        CollectionAssert.AreEqual(t, (int[])y, $"NG: n{n} samples{samples} stride{stride}");

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }
    }
}
