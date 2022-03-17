using System;
using System.Linq;

namespace AvxBlasTest.Connection1DTest {
    public class Map1D {
        private readonly double[] val;

        public int Channels { private set; get; }
        public int Width { private set; get; }
        public int Batch { private set; get; }
        public int Length => Channels * Width * Batch;

        public Map1D(int channels, int width, int batch, float[] val = null) {
            if (width < 1 || channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(width * channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(null, nameof(val));
            }

            this.val = (val is null) ? new double[length] : val.Select((v) => (double)v).ToArray();
            this.Width = width;
            this.Channels = channels;
            this.Batch = batch;
        }

        public Map1D(int channels, int width, int batch, double[] val) {
            if (width < 1 || channels < 1 || batch < 1) {
                throw new ArgumentException();
            }

            int length = checked(width * channels * batch);

            if (!(val is null) && val.Length != length) {
                throw new ArgumentException(null, nameof(val));
            }

            this.val = (val is null) ? new double[length] : (double[])val.Clone();
            this.Width = width;
            this.Channels = channels;
            this.Batch = batch;
        }

        public double this[int ch, int x, int th] {
            get {
                if (x < 0 || x >= Width || ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                return val[ch + Channels * (x + Width * th)];
            }
            set {
                if (x < 0 || x >= Width || ch < 0 || ch >= Channels || th < 0 || th >= Batch) {
                    throw new IndexOutOfRangeException();
                }

                val[ch + Channels * (x + Width * th)] = value;
            }
        }

        public double this[int idx] {
            get {
                return val[idx];
            }
            set {
                val[idx] = value;
            }
        }

        public static bool operator ==(Map1D map1, Map1D map2) {
            if (map1.Width != map2.Width) return false;
            if (map1.Channels != map2.Channels) return false;
            if (map1.Batch != map2.Batch) return false;

            return map1.val.SequenceEqual(map2.val);
        }

        public static bool operator !=(Map1D map1, Map1D map2) {
            return !(map1 == map2);
        }

        public override bool Equals(object obj) {
            return obj is Map1D map && this == map;
        }

        public override int GetHashCode() {
            return base.GetHashCode();
        }

        public float[] ToFloatArray() {
            return val.Select((v) => (float)v).ToArray();
        }
                
        public double[] ToDoubleArray() {
            return (double[])val.Clone();
        }
    }
}
