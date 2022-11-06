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

                Cast.AsType(length, x, y);

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

                Cast.AsType(length, x, y);

                CollectionAssert.AreEqual(t, (float[])y);
            }
        }

        [TestMethod]
        public void IntLongTest() {
            Random random = new(1234);

            for (uint length = 0; length <= 64; length++) {
                int[] x = (new int[length]).Select((_, idx) => random.Next(32) - 16).ToArray();

                long[] t = x.Select((v) => (long)v).Concat(new long[4]).ToArray();

                Array<long> y = new(length + 4);

                Cast.AsType(length, x, y);

                CollectionAssert.AreEqual(t, (long[])y);
            }
        }

        [TestMethod]
        public void LongIntTest() {
            Random random = new(1234);

            for (uint length = 0; length <= 64; length++) {
                long[] x = (new long[length]).Select((_, idx) => (long)(random.Next(65537) - 32768)).ToArray();
                
                int[] t = x.Select((v) => (int)v).Concat(new int[4]).ToArray();

                Array<int> y = new(length + 4);

                Cast.AsType(length, x, y);

                CollectionAssert.AreEqual(t, (int[])y);
            }

            for (uint length = 0; length <= 64; length++) {
                long[] x = (new long[length]).Select((_, idx) => (long)random.Next() * (random.Next(2) == 0 ? -1 : +1)).ToArray();
                
                int[] t = x.Select((v) => (int)v).Concat(new int[4]).ToArray();

                Array<int> y = new(length + 4);

                Cast.AsType(length, x, y);

                CollectionAssert.AreEqual(t, (int[])y);
            }

            {
                long[] x = new long[] { 
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                    int.MaxValue,
                    int.MinValue,
                };

                int[] t = x.Select((v) => (int)v).Concat(new int[4]).ToArray();

                Array<int> y = new(x.Length + 4);

                Cast.AsType((uint)x.Length, x, y);

                CollectionAssert.AreEqual(t, (int[])y);
            }
        }
    }
}
