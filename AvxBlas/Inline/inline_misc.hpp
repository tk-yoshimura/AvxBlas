#include <intrin.h>

// bitscanforward
__forceinline unsigned int bsf(unsigned int n) {
    if (n == 0u) {
        return 0u;
    }

#ifdef _MSC_VER
    unsigned long index;

    _BitScanForward(&index, n);

    return (unsigned int)index;
#else
    return __builtin_ctz(n);
#endif
}