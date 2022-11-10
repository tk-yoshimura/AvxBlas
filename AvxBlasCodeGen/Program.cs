namespace AvxBlasCodeGen {
    public static class Program {
        static void Main() {
            string[] binaryfuncs = new[]{
                ""
            };
            string[] unaryfuncs = new[]{
                ""
            };

            //foreach (string binaryfunc in binaryfuncs) {
            //    Generator.GenerateCode(
            //        "../../../template/ew_binary_s.txt",
            //        $"../../../../AvxBlas/Elementwise/ew_{binaryfunc}_s.cpp",
            //        binaryfunc
            //    );
            //
            //    Generator.GenerateCode(
            //        "../../../template/ew_binary_d.txt",
            //        $"../../../../AvxBlas/Elementwise/ew_{binaryfunc}_d.cpp",
            //        binaryfunc
            //    );
            //}
            //
            //foreach (string unaryfunc in unaryfuncs) {
            //    Generator.GenerateCode(
            //        "../../../template/ew_unary_s.txt",
            //        $"../../../../AvxBlas/Elementwise/ew_{unaryfunc}_s.cpp",
            //        unaryfunc
            //    );
            //
            //    Generator.GenerateCode(
            //        "../../../template/ew_unary_d.txt",
            //        $"../../../../AvxBlas/Elementwise/ew_{unaryfunc}_d.cpp",
            //        unaryfunc
            //    );
            //}

            Console.WriteLine("END");
            Console.Read();
        }
    }
}
