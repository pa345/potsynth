/*
 * window.c -- exponential-of-semicircle window, its Fourier transform, and the
 * choice of window width
 *
 *   phi(z) = exp(beta (sqrt(1 - z^2) - 1)),  |z| <= 1,   z = x / (w h / 2)
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>

#include "potsynth_internal.h"

/* Measured maximum error of the complete transform, relative to max |B|, against
 * a 19-digit reference (tests/gen_reference.py), worst case over L = 13, L = 185
 * (250-800 km shell) and L = 185 (r in [a, 2a]); indexed by w. The error is
 * essentially independent of L and of the radial range; ~5e-15 is the rounding
 * floor of double precision. */
static const double err_sigma2[PS_WMAX + 1] =
{
  0, 0, 0, 0, 8e-4, 2e-5, 9e-6, 7e-7, 3e-8, 5e-9, 6e-10, 3e-11, 6e-12, 4e-13, 3e-14, 1e-14, 7e-15
};

static const double err_sigma3[PS_WMAX + 1] =
{
  0, 0, 0, 0, 2e-4, 3e-5, 1e-6, 2e-7, 7e-9, 7e-10, 6e-11, 3e-12, 4e-13, 1e-14, 1e-14, 7e-15, 9e-15
};

int
ps_window_choose(double eps, double sigma)
{
  const double *table = (sigma >= 3.0) ? err_sigma3 : err_sigma2;
  int w;

  for (w = 4; w <= PS_WMAX; ++w)
    if (table[w] <= eps)
      return w;

  return PS_WMAX;
}

/* Gauss-Legendre nodes and weights on [-1,1] (Newton iteration on P_n) */
void
ps_gauss_legendre(size_t n, double *x, double *w)
{
  size_t i, k, it;

  for (i = 0; i < (n + 1) / 2; ++i)
    {
      double z = cos(M_PI * (i + 0.75) / (n + 0.5));
      double dp = 1.0;

      for (it = 0; it < 100; ++it)
        {
          double p0 = 1.0, p1 = z, dz;

          for (k = 2; k <= n; ++k)
            {
              const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
              p0 = p1;
              p1 = p2;
            }

          /* p1 = P_n(z), p0 = P_{n-1}(z) */
          dp = n * (z * p1 - p0) / (z * z - 1.0);
          dz = p1 / dp;
          z -= dz;
          if (fabs(dz) <= 1e-16 * fabs(z))
            break;
        }

      {
        double p0 = 1.0, p1 = z;

        for (k = 2; k <= n; ++k)
          {
            const double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = p2;
          }
        dp = n * (z * p1 - p0) / (z * z - 1.0);
      }

      x[i] = -z;
      x[n - 1 - i] = z;
      w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
}

int
ps_window_ft(size_t nk, int w, double beta, double h, double *out)
{
  const size_t nq = 8 * (size_t) w + 100;
  const double alpha = 0.5 * w * h;
  double *zq = malloc(nq * sizeof(double));
  double *wq = malloc(nq * sizeof(double));
  size_t k, q;

  if (!zq || !wq)
    {
      free(zq);
      free(wq);
      return POTSYNTH_ENOMEM;
    }

  ps_gauss_legendre(nq, zq, wq);

  for (q = 0; q < nq; ++q)
    wq[q] *= ps_window(zq[q], beta);

  for (k = 0; k < nk; ++k)
    {
      double sum = 0.0;

      for (q = 0; q < nq; ++q)
        sum += wq[q] * cos((double) k * alpha * zq[q]);

      out[k] = alpha * sum;
    }

  free(zq);
  free(wq);
  return POTSYNTH_SUCCESS;
}
