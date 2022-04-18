using AvxBlas;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;

namespace AvxBlasTest.UtilTest {
    [TestClass]
    public class CheckTest {
        [TestMethod]
        public void CheckProdOverflowTest() {
            Util.CheckProdOverflow(255, 255, 255, 255);
            Util.CheckProdOverflow(256, 256, 256, 255);
            Util.CheckProdOverflow(254, 255, 256, 257);
            Util.CheckProdOverflow(65535, 65537);
            Util.CheckProdOverflow(65537, 65535);

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(255, 255, 255, 255, 255);
            });

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(256, 256, 256, 256);
            });

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(255, 256, 257, 258);
            });

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(65536, 65536);
            });

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(65536, 65537);
            });

            Assert.ThrowsException<OverflowException>(() => { 
                Util.CheckProdOverflow(65537, 65536);
            });
        }

        [TestMethod]
        public void CheckLengthTest() {
            Array<float> x = new(256);

            Util.CheckLength(256, x);

            Assert.ThrowsException<IndexOutOfRangeException>(() => { 
                Util.CheckLength(257, x);
            });
        }

        [TestMethod]
        public void CheckOutOfRangeTest() {
            Array<float> x = new(256);

            Util.CheckOutOfRange(32, 224, x);

            Assert.ThrowsException<IndexOutOfRangeException>(() => { 
                Util.CheckOutOfRange(32, 225, x);
            });

            Assert.ThrowsException<IndexOutOfRangeException>(() => { 
                Util.CheckOutOfRange(256, 0, x);
            });
        }
    }
}
