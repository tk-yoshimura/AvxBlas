using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.VectorwiseTest {
    [TestClass]
    public class SoftmaxTest {
        [TestMethod]
        public void SSoftmaxTest() {
            Random random = new(1234);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                for (uint stride = 0; stride <= 96; stride++) {

                    float[] x = (new float[n * stride]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    float[] v = new float[n];
                    for (uint i = 0; i < n; i++) {
                        v[i] = x.Skip(checked((int)(i * stride))).Take((int)stride).Select((v) => (float)Math.Exp(v)).Sum();
                    }

                    float[] t = (new float[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? (float)(Math.Exp(x[idx]) / v[idx / stride]) : 0)
                        .ToArray();

                    Array<float> y = new(n * stride + 4);

                    Vectorwise.Softmax(n, stride, x, y);

                    float[] ys = (float[])y;

                    for (int i = 0; i < t.Length; i++) {
                        Assert.AreEqual(t[i], ys[i], 1e-6, $"NG: n{n} stride{stride}");
                    }

                    Console.WriteLine($"OK: n{n} stride{stride}");
                }
            }

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                for (uint stride = 0; stride <= 96; stride++) {

                    float[] x = (new float[n * stride]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                    if (x.Length > 0) {
                        for (int i = 0; i < Math.Max(1, x.Length / 10); i++) {
                            x[random.Next(x.Length)] = float.NaN;
                        }
                    }

                    float[] v = new float[n];
                    for (uint i = 0; i < n; i++) {
                        v[i] = x.Skip(checked((int)(i * stride))).Take((int)stride).Select((v) => (float)Math.Exp(v)).Sum();
                    }

                    float[] t = (new float[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? (float)(Math.Exp(x[idx]) / v[idx / stride]) : 0)
                        .ToArray();

                    Array<float> y = new(n * stride + 4);

                    Vectorwise.Softmax(n, stride, x, y);

                    float[] ys = (float[])y;

                    for (int i = 0; i < t.Length; i++) {
                        if (!double.IsNaN(t[i])) {
                            Assert.AreEqual(t[i], ys[i], 1e-6, $"NG: n{n} stride{stride} contamin nan");
                        }
                        else {
                            Assert.IsTrue(double.IsNaN(ys[i]), $"NG: n{n} stride{stride} contamin nan");
                        }
                    }

                    Console.WriteLine($"OK: n{n} stride{stride} contamin nan");
                }
            }
        }

        [TestMethod]
        public void SSoftmaxInPlaceTest() {
            Random random = new(12345);

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                for (uint stride = 0; stride <= 96; stride++) {

                    float[] x = (new float[n * stride + 4]).Select(
                        (_, idx) => idx < n * stride ? (float)random.Next(32) - 16 : 0)
                        .ToArray();

                    float[] v = new float[n];
                    for (uint i = 0; i < n; i++) {
                        v[i] = x.Skip(checked((int)(i * stride))).Take((int)stride).Select((v) => (float)Math.Exp(v)).Sum();
                    }

                    float[] t = (new float[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? (float)(Math.Exp(x[idx]) / v[idx / stride]) : 0)
                        .ToArray();

                    Array<float> y = x;

                    Vectorwise.Softmax(n, stride, y, y);

                    float[] ys = (float[])y;

                    for (int i = 0; i < t.Length; i++) {
                        Assert.AreEqual(t[i], ys[i], 1e-6, $"NG: n{n} stride{stride}");
                    }

                    Console.WriteLine($"OK: n{n} stride{stride}");
                }
            }

            foreach (uint n in new uint[] {
                    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u,
                    15u, 16u, 17u, 63u, 64u, 65u, 255u, 256u, 257u }) {

                for (uint stride = 0; stride <= 96; stride++) {

                    float[] x = (new float[n * stride + 4]).Select(
                        (_, idx) => idx < n * stride ? (float)random.Next(32) - 16 : 0)
                        .ToArray();

                    if (n * stride > 0) {
                        for (int i = 0; i < Math.Max(1, n * stride / 10); i++) {
                            x[random.Next(checked((int)(n * stride)))] = float.NaN;
                        }
                    }

                    float[] v = new float[n];
                    for (uint i = 0; i < n; i++) {
                        v[i] = x.Skip(checked((int)(i * stride))).Take((int)stride).Select((v) => (float)Math.Exp(v)).Sum();
                    }

                    float[] t = (new float[n * stride + 4])
                        .Select((_, idx) => idx < n * stride ? (float)(Math.Exp(x[idx]) / v[idx / stride]) : 0)
                        .ToArray();

                    Array<float> y = x;

                    Vectorwise.Softmax(n, stride, y, y);

                    float[] ys = (float[])y;

                    for (int i = 0; i < t.Length; i++) {
                        if (!double.IsNaN(t[i])) {
                            Assert.AreEqual(t[i], ys[i], 1e-6, $"NG: n{n} stride{stride} contamin nan");
                        }
                        else {
                            Assert.IsTrue(double.IsNaN(ys[i]), $"NG: n{n} stride{stride} contamin nan");
                        }
                    }

                    Console.WriteLine($"OK: n{n} stride{stride} contamin nan");
                }
            }
        }
    }
}
