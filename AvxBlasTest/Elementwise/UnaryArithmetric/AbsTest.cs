using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AvxBlasTest.ElementwiseTest {
    [TestClass]
    public class ClearTest {
        [TestMethod]
        public void SAbsTest() {
            Array<float> x = new float[] { 0, -1, -2, 3, -4, -5, 6, -7, 8, 9, -10 };
            Array<float> t = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0 };

            Array<float> y = new float[x.Length];

            Elementwise.Abs(9, x, y);

            CollectionAssert.AreEqual((float[])t, (float[])y);
        }

        [TestMethod]
        public void DAbsTest() {
            Array<double> x = new double[] { 0, -1, -2, 3, -4, -5, 6, -7, float.NegativeInfinity, 9, -10 };
            Array<double> t = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, float.PositiveInfinity, 0, 0 };

            Array<double> y = new double[x.Length];

            Elementwise.Abs(9, x, y);

            CollectionAssert.AreEqual((double[])t, (double[])y);
        }
    }
}
