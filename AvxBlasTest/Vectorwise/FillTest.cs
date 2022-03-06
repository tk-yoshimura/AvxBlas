using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.VectorwiseTest {
    [TestClass]
    public class FillTest {
        [TestMethod]
        public void SFillTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                foreach (uint stride in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u }) {

                    float[] v = (new float[stride]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? v[idx % stride] : 0)
                        .ToArray();

                    Array<float> y = new(n * stride + 4);

                    Vectorwise.Fill(n, stride, v, y);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} stride{stride}");

                    Console.WriteLine($"OK: n{n} stride{stride}");
                }
            }
        }

        [TestMethod]
        public void DFillTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                foreach (uint stride in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u }) {

                    double[] v = (new double[stride]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                    double[] t = (new double[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? v[idx % stride] : 0)
                        .ToArray();

                    Array<double> y = new(n * stride + 4);

                    Vectorwise.Fill(n, stride, v, y);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} stride{stride}");

                    Console.WriteLine($"OK: n{n} stride{stride}");
                }
            }
        }
    }
}
