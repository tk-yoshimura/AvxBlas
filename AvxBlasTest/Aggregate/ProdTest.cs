using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.AggregateTest {
    [TestClass]
    public class ProdTest {
        [TestMethod]
        public void SProdTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u }) {

                foreach (uint samples in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u,
                        63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u }) {

                        uint inlength = n * samples * stride + 4;
                        uint outlength = n * stride + 4;

                        float[] x = (new float[inlength]).Select((_, idx) => (float)random.NextDouble() * 8 - 4).ToArray();

                        float[] ts = new float[outlength];

                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < stride; j++) {
                                float s = 1;

                                for (int k = 0; k < samples; k++) {
                                    s *= x[j + stride * (k + samples * i)];
                                }

                                ts[j + stride * i] = s;
                            }
                        }

                        Array<float> y = new(outlength);

                        Aggregate.Prod(n, samples, stride, x, y);

                        float[] ys = y;

                        for (int i = 0; i < ts.Length; i++) {
                            Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-5, $"NG: n{n} samples{samples} stride{stride}");
                        }

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }

        [TestMethod]
        public void DProdTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 15u, 16u, 17u }) {

                foreach (uint samples in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                    foreach (uint stride in new uint[] {
                        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u,
                        15u, 16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u,
                        63u, 64u, 65u, 127u, 128u, 129u, 255u, 256u, 257u }) {

                        uint inlength = n * samples * stride + 4;
                        uint outlength = n * stride + 4;

                        double[] x = (new double[inlength]).Select((_, idx) => (double)random.NextDouble() * 8 - 4).ToArray();

                        double[] ts = new double[outlength];

                        for (int i = 0; i < n; i++) {
                            for (int j = 0; j < stride; j++) {
                                double s = 1;

                                for (int k = 0; k < samples; k++) {
                                    s *= x[j + stride * (k + samples * i)];
                                }

                                ts[j + stride * i] = s;
                            }
                        }

                        Array<double> y = new(outlength);

                        Aggregate.Prod(n, samples, stride, x, y);

                        double[] ys = y;

                        for (int i = 0; i < ts.Length; i++) {
                            Assert.AreEqual(ts[i], ys[i], Math.Abs(ts[i]) * 1e-10, $"NG: n{n} samples{samples} stride{stride}");
                        }

                        Console.WriteLine($"OK: n{n} samples{samples} stride{stride}");
                    }
                }
            }
        }
    }
}
