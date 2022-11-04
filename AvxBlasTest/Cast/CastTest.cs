using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;

namespace AvxBlasTest.CastTest {
    [TestClass]
    public class CastTest {
        [TestMethod]
        public void FloatDoubleTest() {
            Random random = new(1234);

            for (uint length = 0; length <= 64; length++) {
                float[] x = (new float[length]).Select((_, idx) => (float)random.Next(32) - 16).ToArray();

                double[] t = x.Select((v) => (double)v).Concat(new double[4]).ToArray();

                Array<double> y = new(length + 4);

                Elementwise.Cast(length, x, y);

                CollectionAssert.AreEqual(t, (double[])y);
            }
        }

        [TestMethod]
        public void DoubleFloatTest() {
            Random random = new(1234);

            for (uint length = 0; length <= 64; length++) {
                double[] x = (new double[length]).Select((_, idx) => (double)random.Next(32) - 16).ToArray();

                float[] t = x.Select((v) => (float)v).Concat(new float[4]).ToArray();

                Array<float> y = new(length + 4);

                Elementwise.Cast(length, x, y);

                CollectionAssert.AreEqual(t, (float[])y);
            }
        }
    }
}
