namespace AvxBlasCodeGen {
    public static class Generator {
        public static void GenerateCode(string templatefilepath, string dstfilepath, string funcname) {
            using StreamReader sr = new(templatefilepath);
            string code = sr.ReadToEnd();

            code = code.Replace("#ope#", funcname).Replace("#Ope#", char.ToUpper(funcname[0]) + funcname[1..]);

            using StreamWriter sw = new(dstfilepath);
            sw.Write(code);
        }
    }
}
