/*
 * util.c
 *
 * Memory, sizes, threads and small numerical helpers
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#endif

#include "potsynth_internal.h"

/*
potsynth_aligned_alloc()
  Allocate memory aligned to POTSYNTH_ALIGN bytes

Inputs: nbytes - number of bytes

Return: pointer to memory, free with potsynth_aligned_free()
*/

void *
potsynth_aligned_alloc(const size_t nbytes)
{
  size_t size = (nbytes + POTSYNTH_ALIGN - 1) / POTSYNTH_ALIGN * POTSYNTH_ALIGN;

  if (size == 0)
    size = POTSYNTH_ALIGN;

#ifdef _WIN32
  return _aligned_malloc(size, POTSYNTH_ALIGN);
#else
  return aligned_alloc(POTSYNTH_ALIGN, size);
#endif
}

/* aligned allocation initialized to zero */
void *
potsynth_aligned_calloc(const size_t nbytes)
{
  void * ptr = potsynth_aligned_alloc(nbytes);

  if (ptr)
    memset(ptr, 0, nbytes);

  return ptr;
}

void
potsynth_aligned_free(void * ptr)
{
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/*
potsynth_next_smooth()
  Return smallest n' >= n which is a multiple of mult and has no prime
factors other than 2, 3, 5 (fast FFT length)

Inputs: n    - minimum length
        mult - required multiple
*/

size_t
potsynth_next_smooth(const size_t n, const size_t mult)
{
  size_t k;

  for (k = (((n < 1) ? 1 : n) + mult - 1) / mult * mult; ; k += mult)
    {
      size_t r = k;

      while (r % 2 == 0)
        r /= 2;
      while (r % 3 == 0)
        r /= 3;
      while (r % 5 == 0)
        r /= 5;

      if (r == 1)
        return k;
    }
}

/* number of threads to use (requested > 0, or all available) */
int
potsynth_max_threads(const int requested)
{
#ifdef _OPENMP
  if (requested > 0)
    return requested;
  return omp_get_max_threads();
#else
  (void) requested;
  return 1;
#endif
}

/*
potsynth_pow_scaled()
  Compute s^k as a mantissa and exponent, s^k = mant * 2^expo,
by repeated squaring with renormalization, so that it never
underflows or overflows

Inputs: s    - base
        k    - power
        mant - (output) mantissa, 0.5 <= |mant| < 1 (mant = 1 for k = 0, 0 for s = 0)
        expo - (output) exponent

Notes:
1) The relative error is bounded by about k * eps, as for a running product
*/

void
potsynth_pow_scaled(const double s, const size_t k, double * mant, int * expo)
{
  const int negate = (s < 0.0) && (k & 1);
  size_t kk = k;
  double rm = 1.0, bm;
  int re = 0, be;

  bm = frexp(fabs(s), &be);

  if (bm == 0.0)
    {
      *mant = (k == 0) ? 1.0 : 0.0;
      *expo = 0;
      return;
    }

  while (kk > 0)
    {
      int e;

      if (kk & 1)
        {
          rm = frexp(rm * bm, &e);
          re += be + e;
        }

      kk >>= 1;

      if (kk > 0)
        {
          bm = frexp(bm * bm, &e);
          be = 2 * be + e;
        }
    }

  *mant = negate ? -rm : rm;
  *expo = re;
}

/*
potsynth_sincos_m()
  Compute cos(m phi) and sin(m phi) accurately for a double phi,
correcting for the rounding error of the product m*phi

Inputs: m   - order
        phi - angle (radians)
        c   - (output) cos(m phi)
        s   - (output) sin(m phi)
*/

void
potsynth_sincos_m(const size_t m, const double phi, double * c, double * s)
{
  const double dm = (double) m;
  const double hi = dm * phi;
  const double lo = fma(dm, phi, -hi); /* exact rounding error of m*phi */
  const double sh = sin(hi);
  const double ch = cos(hi);

  *s = sh + lo * ch;
  *c = ch - lo * sh;
}

/*
potsynth_cpu_avx2_fma()
  Report whether the CPU (and operating system) support AVX2 and FMA

Return: 1 if supported, 0 otherwise (always 0 on other architectures)

Notes:
1) Used by the Python package to choose between an AVX2/FMA build of the
library and a portable build; it is itself safe to call on any x86-64 CPU
*/

int
potsynth_cpu_avx2_fma(void)
{
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int r[4];

  /* leaf 1: FMA (ecx 12), OSXSAVE (ecx 27), AVX (ecx 28) */
  __cpuid(r, 1);
  if (!(r[2] & (1 << 12)) || !(r[2] & (1 << 27)) || !(r[2] & (1 << 28)))
    return 0;

  /* the OS saves the XMM and YMM registers */
  if ((_xgetbv(0) & 6) != 6)
    return 0;

  /* leaf 7: AVX2 (ebx 5) */
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 5)) != 0;
#else
  return 0;
#endif
}
