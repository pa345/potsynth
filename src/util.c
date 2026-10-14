/*
 * util.c -- memory, sizes, threads and small numerical helpers
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"

#ifdef _WIN32
#include <malloc.h>
#endif

const char *
potsynth_strerror(int status)
{
  switch (status)
    {
      case POTSYNTH_SUCCESS:
        return "success";
      case POTSYNTH_EINVAL:
        return "invalid argument";
      case POTSYNTH_ENOMEM:
        return "memory allocation failed";
      case POTSYNTH_EDOM:
        return "evaluation point outside the valid domain";
      case POTSYNTH_EUNSUP:
        return "unsupported option";
      default:
        return "unknown error";
    }
}

void
potsynth_params_init(potsynth_params *params)
{
  params->eps = 1.0e-10;
  params->method = POTSYNTH_METHOD_AUTO;
  params->sigma = 0.0;
  params->w = 0;
  params->rank_tol = 0.0;
  params->nthreads = 0;
}

size_t
potsynth_ncoef(size_t lmin, size_t lmax, size_t mmax)
{
  size_t n, sum = 0;

  for (n = lmin; n <= lmax; ++n)
    {
      size_t M = (n < mmax) ? n : mmax;
      sum += 2 * M + 1;
    }

  return sum;
}

void *
ps_malloc(size_t nbytes)
{
  size_t size = (nbytes + PS_ALIGN - 1) / PS_ALIGN * PS_ALIGN;

  if (size == 0)
    size = PS_ALIGN;

#ifdef _WIN32
  return _aligned_malloc(size, PS_ALIGN);
#else
  return aligned_alloc(PS_ALIGN, size);
#endif
}

void *
ps_calloc(size_t nbytes)
{
  void *ptr = ps_malloc(nbytes);

  if (ptr)
    memset(ptr, 0, nbytes);

  return ptr;
}

void
ps_free(void *ptr)
{
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

/* smallest n' >= n which is a multiple of mult and has no prime factors other than 2, 3, 5 */
size_t
ps_next_smooth(size_t n, size_t mult)
{
  size_t k;

  if (n < 1)
    n = 1;

  for (k = (n + mult - 1) / mult * mult; ; k += mult)
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

int
ps_max_threads(int requested)
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

void
ps_pow_scaled(double s, size_t k, double *mant, int *expo)
{
  const int negate = (s < 0.0) && (k & 1);
  double rm = 1.0, bm;
  int re = 0, be;

  bm = frexp(fabs(s), &be);

  if (bm == 0.0)
    {
      *mant = (k == 0) ? 1.0 : 0.0;
      *expo = 0;
      return;
    }

  while (k > 0)
    {
      int e;

      if (k & 1)
        {
          rm = frexp(rm * bm, &e);
          re += be + e;
        }

      k >>= 1;

      if (k > 0)
        {
          bm = frexp(bm * bm, &e);
          be = 2 * be + e;
        }
    }

  *mant = negate ? -rm : rm;
  *expo = re;
}

void
ps_sincos_m(size_t m, double phi, double *c, double *s)
{
  const double dm = (double) m;
  const double hi = dm * phi;
  const double lo = fma(dm, phi, -hi);  /* exact rounding error of m*phi */
  const double sh = sin(hi);
  const double ch = cos(hi);

  *s = sh + lo * ch;
  *c = ch - lo * sh;
}
