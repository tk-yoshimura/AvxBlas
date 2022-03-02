using Microsoft.VisualStudio.TestTools.UnitTesting;
using AvxBlas;

namespace AvxBlasTest {
    [TestClass]
    public class ElementwiseTest {
        [TestMethod]
        public void AddTest() {
            float[] x1 = new float[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            float[] x2 = new float[] { 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5 };
            float[] t = new float[]  { 6, 8, 10, 12, 14, 5, 7, 9, 11, 0, 0 };

            float[] y = new float[x1.Length];

            Elementwise.Add(9, x1, x2, y);
            
            CollectionAssert.AreEqual(t, y);
        }
    }
}
