using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AvxBlasTest.FlipTest {
    [TestClass]
    public class FlipTest {
        [TestMethod]
        public void SFlipTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Reverse(t, (int)(i * s), (int)s);
                    }

                    Array<float> y = (float[])x.Clone();

                    Permutate.Flip(n, s, x, y);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s}");

                    Console.WriteLine($"OK: n{n} s{s}");
                }
            }
        }

        [TestMethod]
        public void DFlipTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select((_, idx) => random.NextDouble()).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Reverse(t, (int)(i * s), (int)s);
                    }

                    Array<double> y = (double[])x.Clone();

                    Permutate.Flip(n, s, x, y);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s}");

                    Console.WriteLine($"OK: n{n} s{s}");
                }
            }
        }
    }
}
