using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.AggregateTest {
    [TestClass]
    public class SumTest {
        [TestMethod]
        public void SAddTest() {
            Random random = new(1234);

            for (uint n = 1; n <= 8; n++) {

                foreach (uint samples in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                        15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                        uint inlength = n * samples * stride + 4;
                        uint outlength = n * stride + 4;

                        float[] x = (new float[inlength]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                        float[] t = new float[outlength];

                        for (int i = 0; i < n; i++) { 
                            for (int j = 0; j < stride; j++) {
                                float s = 0;

                                for (int k = 0; k < samples; k++) {
                                    s += x[j + stride * (k + samples * i)];
                                }

                                t[j + stride * i] = s;
                            }
                        }

                        Array<float> y = new(outlength);

                        Aggregate.Sum(n, samples, stride, x, y);

                        CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} samples{samples} stride{stride}");

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }
    }
}
