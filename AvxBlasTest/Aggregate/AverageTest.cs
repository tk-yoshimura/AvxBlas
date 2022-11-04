using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.AggregateTest {
    [TestClass]
    public class AverageTest {
        [TestMethod]
        public void SAverageTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u }) {

                foreach (uint samples in new uint[] {
                    1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u,
                        63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u }) {

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

                                t[j + stride * i] = s / samples;
                            }
                        }

                        Array<float> y = new(outlength);

                        Aggregate.Average(n, samples, stride, x, y);

                        float[] ys = (float[])y;

                        for (int i = 0; i < t.Length; i++) {
                            Assert.AreEqual(t[i], ys[i], 1e-6, $"NG: n{n} samples{samples} stride{stride}");
                        }

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }

        [TestMethod]
        public void DAverageTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u }) {

                foreach (uint samples in new uint[] {
                    1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u,
                        63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u }) {

                        uint inlength = n * samples * stride + 4;
                        uint outlength = n * stride + 4;

                        double[] x = (new double[inlength]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                        double[] t = new double[outlength];

                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < stride; j++) {
                                double s = 0;

                                for (int k = 0; k < samples; k++) {
                                    s += x[j + stride * (k + samples * i)];
                                }

                                t[j + stride * i] = s / samples;
                            }
                        }

                        Array<double> y = new(outlength);

                        Aggregate.Average(n, samples, stride, x, y);

                        double[] ys = (double[])y;

                        for (int i = 0; i < t.Length; i++) {
                            Assert.AreEqual(t[i], ys[i], 1e-14, $"NG: n{n} samples{samples} stride{stride}");
                        }

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }
    }
}
