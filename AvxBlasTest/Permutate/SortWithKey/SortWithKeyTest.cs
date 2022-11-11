using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AvxBlasTest.SortTest {
    [TestClass]
    public class SortWithKeyTest {

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
        public void SSortWithKeyAscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s);
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_none");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_none");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SAscMinNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_minimize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_minimize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_minimize");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SAscMaxNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MaximizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_maximize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_maximize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_maximize");
                }
            }
        }

        [TestMethod]
        public void SSortWithKeyDscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s);
                        Array.Reverse(tk, (int)(i * s), (int)s);
                        Array.Reverse(tv, (int)(i * s), (int)s);
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_none");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_none");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SDscMinNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_minimize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_minimize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_minimize");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SDscMaxNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MaximizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_maximize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_maximize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_maximize");
                }
            }
        }

        [TestMethod]
        public void DSortWithKeyAscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s);
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_none");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_none");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SAscMinNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_minimize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_minimize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_minimize");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    float[] k = (new float[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? (float)random.NextDouble() : float.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    float[] tk = (float[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new SAscMaxNaN());
                    }

                    Array<float> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Ascending, SortNanMode.MaximizeNaN);

                    CollectionAssert.AreEqual(tk, (float[])yk, $"NG: n{n} s{s} nan_maximize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_maximize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_maximize");
                }
            }
        }

        [TestMethod]
        public void DSortWithKeyDscTest() {
            Random random = new(1234);

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] k = (new double[checked(n * s + 4)]).Select((_, idx) => random.NextDouble()).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    double[] tk = (double[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s);
                        Array.Reverse(tk, (int)(i * s), (int)s);
                        Array.Reverse(tv, (int)(i * s), (int)s);
                    }

                    Array<double> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (double[])yk, $"NG: n{n} s{s} nan_none");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_none");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_none");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] k = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? random.NextDouble() : double.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    double[] tk = (double[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new DDscMinNaN());
                    }

                    Array<double> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MinimizeNaN);

                    CollectionAssert.AreEqual(tk, (double[])yk, $"NG: n{n} s{s} nan_minimize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_minimize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_minimize");
                }
            }

            for (uint n = 0; n <= 64; n++) {
                for (uint s = 0; s <= 256; s++) {
                    double[] k = (new double[checked(n * s + 4)]).Select(
                        (_, idx) => random.Next(4) > 0 ? random.NextDouble() : double.NaN).ToArray();
                    int[] v = (new int[checked(n * s + 4)]).Select((_, idx) => idx).ToArray();

                    double[] tk = (double[])k.Clone();
                    int[] tv = (int[])v.Clone();

                    for (uint i = 0; i < n; i++) {
                        Array.Sort(tk, tv, (int)(i * s), (int)s, new DDscMaxNaN());
                    }

                    Array<double> yk = k;
                    Array<int> yv = v;

                    Permutate.SortWithKey(n, s, yk, yv, SortOrder.Descending, SortNanMode.MaximizeNaN);

                    CollectionAssert.AreEqual(tk, (double[])yk, $"NG: n{n} s{s} nan_maximize");

                    for (uint i = 0; i < s * n; i++) {
                        Assert.AreEqual(k[yv[i]], yk[i], $"NG: n{n} s{s} nan_maximize");
                    }

                    Console.WriteLine($"OK: n{n} s{s} nan_maximize");
                }
            }
        }
    }
}
