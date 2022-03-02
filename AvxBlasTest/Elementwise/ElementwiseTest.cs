using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace AvxBlasTest {
    [TestClass]
    public class ElementwiseTest {
        [TestMethod]
        public void SAddTest() {
            Array<float> x1 = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            Array<float> x2 = new float[] { 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5 };
            Array<float> t = new float[] { 6, 8, 10, 12, 14, 5, 7, 9, 11, 0, 0 };

            Array<float> y = new float[x1.Length];

            Elementwise.Add(9, x1, x2, y);

            CollectionAssert.AreEqual((float[])t, (float[])y);
        }

        [TestMethod]
        public void DAddTest() {
            Array<double> x1 = new double[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            Array<double> x2 = new double[] { 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5 };
            Array<double> t = new double[] { 6, 8, 10, 12, 14, 5, 7, 9, 11, 0, 0 };

            Array<double> y = new double[x1.Length];

            Elementwise.Add(9, x1, x2, y);

            CollectionAssert.AreEqual((double[])t, (double[])y);
        }
    }
}
