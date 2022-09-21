using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AvxBlasTest.SortTest {
    [TestClass]
    public class SortTest {

        private class SAscMinNaN : IComparer<float> {
            public int Compare(float x, float y) {
                if (x == y || float.IsNaN(x) && float.IsNaN(y)) {
                    return 0;
                }
                if (float.IsNaN(x)) {
                    return -1;
                }
                if (float.IsNaN(y)) {
                    return +1;
                }

                return (x < y) ? -1 : +1;
            }
        }

        private class SDscMinNaN : IComparer<float> {
            public int Compare(float x, float y) {
                if (x == y || float.IsNaN(x) && float.IsNaN(y)) {
                    return 0;
                }
                if (float.IsNaN(x)) {
                    return +1;
                }
                if (float.IsNaN(y)) {
                    return -1;
                }

                return (x < y) ? +1 : -1;
            }
        }

        private class SAscMaxNaN : IComparer<float> {
            public int Compare(float x, float y) {
                if (x == y || float.IsNaN(x) && float.IsNaN(y)) {
                    return 0;
                }
                if (float.IsNaN(x)) {
                    return +1;
                }
                if (float.IsNaN(y)) {
                    return -1;
                }

                return (x < y) ? -1 : +1;
            }
        }

        private class SDscMaxNaN : IComparer<float> {
            public int Compare(float x, float y) {
                if (x == y || float.IsNaN(x) && float.IsNaN(y)) {
                    return 0;
                }
                if (float.IsNaN(x)) {
                    return -1;
                }
                if (float.IsNaN(y)) {
                    return +1;
                }

                return (x < y) ? +1 : -1;
            }
        }

        private class DAscMinNaN : IComparer<double> {
            public int Compare(double x, double y) {
                if (x == y || double.IsNaN(x) && double.IsNaN(y)) {
                    return 0;
                }
                if (double.IsNaN(x)) {
                    return -1;
                }
                if (double.IsNaN(y)) {
                    return +1;
                }

                return (x < y) ? -1 : +1;
            }
        }

        private class DDscMinNaN : IComparer<double> {
            public int Compare(double x, double y) {
                if (x == y || double.IsNaN(x) && double.IsNaN(y)) {
                    return 0;
                }
                if (double.IsNaN(x)) {
                    return +1;
                }
                if (double.IsNaN(y)) {
                    return -1;
                }

                return (x < y) ? +1 : -1;
            }
        }

        private class DAscMaxNaN : IComparer<double> {
            public int Compare(double x, double y) {
                if (x == y || double.IsNaN(x) && double.IsNaN(y)) {
                    return 0;
                }
                if (double.IsNaN(x)) {
                    return +1;
                }
                if (double.IsNaN(y)) {
                    return -1;
                }

                return (x < y) ? -1 : +1;
            }
        }

        private class DDscMaxNaN : IComparer<double> {
            public int Compare(double x, double y) {
                if (x == y || double.IsNaN(x) && double.IsNaN(y)) {
                    return 0;
                }
                if (double.IsNaN(x)) {
                    return -1;
                }
                if (double.IsNaN(y)) {
                    return +1;
                }

                return (x < y) ? +1 : -1;
            }
        }

        [TestMethod]
        public void SSortAscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s);
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_none");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new SAscMinNaN());
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_minimize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new SAscMaxNaN());
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MaximizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_maximize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }
        }

        [TestMethod]
        public void SSortDscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s);
                        Array.Reverse(t, (int)(i * s), (int)s);
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_none");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new SDscMinNaN());
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_minimize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] x = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    float[] t = (float[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new SDscMaxNaN());
                    }

                    Array<float> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MaximizeNaN);

                    CollectionAssert.AreEqual(t, (float[])y, $"NG: n{n} s{s} nan_maximize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }
        }

        [TestMethod]
        public void DSortAscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select((_, idx) => (double)random.NextDouble()).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s);
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_none");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (double)random.NextDouble() : double.NaN).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new DAscMinNaN());
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_minimize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (double)random.NextDouble() : double.NaN).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new DAscMaxNaN());
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Ascending, SortNaNMode.MaximizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_maximize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }
        }

        [TestMethod]
        public void DSortDscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select((_, idx) => (double)random.NextDouble()).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s);
                        Array.Reverse(t, (int)(i * s), (int)s);
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_none");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (double)random.NextDouble() : double.NaN).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new DDscMinNaN());
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MinimizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_minimize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] x = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (double)random.NextDouble() : double.NaN).ToArray();
                    double[] t = (double[])x.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(t, (int)(i * s), (int)s, new DDscMaxNaN());
                    }

                    Array<double> y = x;

                    Permutate.Sort(n, s, y, SortOrder.Descending, SortNaNMode.MaximizeNaN);

                    CollectionAssert.AreEqual(t, (double[])y, $"NG: n{n} s{s} nan_maximize");

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }
        }
    }
}
