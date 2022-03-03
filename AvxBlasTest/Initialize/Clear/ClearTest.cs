using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace AvxBlasTest.InitializeTest {
    [TestClass]
    public class ClearTest {
        [TestMethod]
        public void SClearTest() {
            for (uint length = 1; length < 100; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        float[] v = (new float[length]).Select((_, idx) => (float)idx + 1).ToArray();
                        float[] v2 = (new float[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 3.14f : (float)idx + 1)
                            .ToArray();

                        Array<float> arr = new(v);

                        Initialize.Clear(index, count, 3.14f, arr);

                        float[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length < 100; length++) {
                for (uint count = 0; count <= length; count++) {
                    float[] v = (new float[length]).Select((_, idx) => (float)idx + 1).ToArray();
                    float[] v2 = (new float[length])
                        .Select((_, idx) => idx < count ? 3.14f : (float)idx + 1)
                        .ToArray();

                    Array<float> arr = new(v);

                    Initialize.Clear(count, 3.14f, arr);

                    float[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }

        [TestMethod]
        public void DClearTest() {
            for (uint length = 1; length < 100; length++) {
                for (uint index = 0; index < length; index++) {
                    for (uint count = 0; count <= length - index; count++) {
                        double[] v = (new double[length]).Select((_, idx) => (double)idx + 1).ToArray();
                        double[] v2 = (new double[length])
                            .Select((_, idx) => idx >= index && idx < (index + count) ? 3.14d : (double)idx + 1)
                            .ToArray();

                        Array<double> arr = new(v);

                        Initialize.Clear(index, count, 3.14d, arr);

                        double[] v3 = arr;

                        CollectionAssert.AreEqual(v2, v3);
                    }
                }
            }

            for (uint length = 1; length < 100; length++) {
                for (uint count = 0; count <= length; count++) {
                    double[] v = (new double[length]).Select((_, idx) => (double)idx + 1).ToArray();
                    double[] v2 = (new double[length])
                        .Select((_, idx) => idx < count ? 3.14d : (double)idx + 1)
                        .ToArray();

                    Array<double> arr = new(v);

                    Initialize.Clear(count, 3.14d, arr);

                    double[] v3 = arr;

                    CollectionAssert.AreEqual(v2, v3);
                }
            }
        }
    }
}
