#include <iostream>

#include <immintrin.h>
#include <chrono>

#include "avxblas_sandbox.h"
#include "../AvxBlas/Inline/inline_numeric.hpp"
#include "../AvxBlas/Inline/inline_loadstore_xn_s.hpp"
#include "../AvxBlas/Inline/inline_transpose_s.hpp"

__forceinline floatx8 float_linear3d(
    float xluf, float xuf, float xruf,
    float xlf, float xf, float xrf,
    float xldf, float xdf, float xrdf,
    float xlu, float xu, float xru,
    float xl, float xc, float xr,
    float xld, float xd, float xrd,
    float xlub, float xub, float xrub,
    float xlb, float xb, float xrb,
    float xldb, float xdb, float xrdb) {

    float wc2 = xc + xc;

    float wl2c = xl + xl + xc;
    float wr2c = xr + xr + xc;
    float wu2c = xu + xu + xc;
    float wd2c = xd + xd + xc;
    float wf2c = xf + xf + xc;
    float wb2c = xb + xb + xc;

    float wlu2l2u2c2 = (xlu + xlu) + (wl2c + wu2c);
    float wld2l2d2c2 = (xld + xld) + (wl2c + wd2c);
    float wlf2l2f2c2 = (xlf + xlf) + (wl2c + wf2c);
    float wlb2l2b2c2 = (xlb + xlb) + (wl2c + wb2c);
    float wru2r2u2c2 = (xru + xru) + (wr2c + wu2c);
    float wrd2r2d2c2 = (xrd + xrd) + (wr2c + wd2c);
    float wrf2r2f2c2 = (xrf + xrf) + (wr2c + wf2c);
    float wrb2r2b2c2 = (xrb + xrb) + (wr2c + wb2c);
    float wuf2u2f2c2 = (xuf + xuf) + (wu2c + wf2c);
    float wub2u2b2c2 = (xub + xub) + (wu2c + wb2c);
    float wdf2d2f2c2 = (xdf + xdf) + (wd2c + wf2c);
    float wdb2d2b2c2 = (xdb + xdb) + (wd2c + wb2c);

    float yluf = (xluf + wc2) + (wlu2l2u2c2 + wlf2l2f2c2 + wuf2u2f2c2);
    float yruf = (xruf + wc2) + (wru2r2u2c2 + wrf2r2f2c2 + wuf2u2f2c2);
    float yldf = (xldf + wc2) + (wld2l2d2c2 + wlf2l2f2c2 + wdf2d2f2c2);
    float yrdf = (xrdf + wc2) + (wrd2r2d2c2 + wrf2r2f2c2 + wdf2d2f2c2);
    float ylub = (xlub + wc2) + (wlu2l2u2c2 + wlb2l2b2c2 + wub2u2b2c2);
    float yrub = (xrub + wc2) + (wru2r2u2c2 + wrb2r2b2c2 + wub2u2b2c2);
    float yldb = (xldb + wc2) + (wld2l2d2c2 + wlb2l2b2c2 + wdb2d2b2c2);
    float yrdb = (xrdb + wc2) + (wrd2r2d2c2 + wrb2r2b2c2 + wdb2d2b2c2);

    return floatx8(yluf, yruf, yldf, yrdf, ylub, yrub, yldb, yrdb);
}

int main(){
    floatx8 y0 = float_linear3d(
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1);
    floatx8 y1 = float_linear3d(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0);
    floatx8 y2 = float_linear3d(
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0);

    printf("end");
    getchar();
}
