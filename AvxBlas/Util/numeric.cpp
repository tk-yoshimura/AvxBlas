#include "../AvxBlasUtil.h"
#include <exception>

unsigned int AvxBlas::gcd(unsigned int a, unsigned int b) {
    if (b == 0) {
        return a;
    }

    return gcd(b, a % b);
}

unsigned int AvxBlas::lcm(unsigned int a, unsigned int b) {
    unsigned int c = a / gcd(a, b);

    if (b > 0 && c > (~0u) / b) {
        throw std::exception("overflow lcm");
    }

    return c * b;
}

unsigned long AvxBlas::gcd(unsigned long a, unsigned long b) {
    if (b == 0) {
        return a;
    }

    return gcd(b, a % b);
}

unsigned long AvxBlas::lcm(unsigned long a, unsigned long b) {
    unsigned long c = a / gcd(a, b);

    if (b > 0 && c > (~0ul) / b) {
        throw std::exception("overflow lcm");
    }

    return c * b;
}