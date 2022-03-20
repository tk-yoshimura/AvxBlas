#pragma once

typedef unsigned int uint;

typedef const float* __restrict infloats;
typedef const double* __restrict indoubles;
typedef float* __restrict outfloats;
typedef double* __restrict outdoubles;

static_assert(sizeof(float) == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");
static_assert(sizeof(uint) == 4, "sizeof uint must be 4");