using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.VectorwiseTest {
    [TestClass]
    public class AddTest {
        [TestMethod]
        public void SAddTest() {
            Random random = new Random(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                foreach (uint incx in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 255u, 256u, 257u, 1023u, 1024u, 1025u, 4095u, 4096u, 4097u }) {

                    Console.WriteLine($"\n{n} {incx}");

                    float[] x = (new float[n * incx]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();
                    float[] v = (new float[incx]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] t = (new float[n * incx + 4])
                        .Select((_, idx) => idx < n * incx ? x[idx] + v[idx % incx] : 0)
                        .ToArray();

                    Array<float> y = new float[n * incx + 4];

                    Vectorwise.Add(n, incx, x, v, y);

                    CollectionAssert.AreEqual(t, (float[])y);
                }
            }
        }
    }
}
