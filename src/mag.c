/*
 * mag.c -- magnetic front-end: B = -grad V for Schmidt semi-normalised Gauss
 * coefficients
 *
 *   B_r     = sum (n+1) (a/r)^(n+2) [g cos + h sin] P
 *   B_theta = -sum (a/r)^(n+2) [g cos + h sin] dP/dtheta
 *   B_phi   = -sum (a/r)^(n+2) d/dphi[g cos + h sin] P/sin(theta)
 *
 * Schmidt P = Pbar / sqrt(2n+1) (4-pi normalised), so Cbar = g / sqrt(2n+1).
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>

#include "potsynth_internal.h"
#include "potsynth_mag.h"

potsynth_workspace *
potsynth_mag_alloc(size_t lmin, size_t lmax, size_t mmax, const double *gh, double a,
                   double rmin, double rmax, const potsynth_params *params, int *status)
{
  const size_t L1 = lmax + 1;
  const size_t ncoef = potsynth_ncoef(lmin, lmax, mmax);
  potsynth_channel chan[3];
  potsynth_workspace *w;
  double *coef, *rfr, *rfm;
  size_t n, m;

  if (lmin < 1 || lmin > lmax || mmax > lmax || !gh)
    {
      if (status)
        *status = POTSYNTH_EINVAL;
      return NULL;
    }

  if (lmax > POTSYNTH_LMAX)
    {
      if (status)
        *status = POTSYNTH_EUNSUP;
      return NULL;
    }

  coef = malloc(ncoef * sizeof(double));
  rfr = malloc(L1 * sizeof(double));
  rfm = malloc(L1 * sizeof(double));
  if (!coef || !rfr || !rfm)
    {
      free(coef);
      free(rfr);
      free(rfm);
      if (status)
        *status = POTSYNTH_ENOMEM;
      return NULL;
    }

  for (n = lmin; n <= lmax; ++n)
    {
      const size_t M = (n < mmax) ? n : mmax;
      const double f = 1.0 / sqrt(2.0 * n + 1.0);

      for (m = 0; m <= M; ++m)
        {
          const size_t ip = POTSYNTH_IDX((long) n, (long) m, (long) lmin, (long) mmax);
          const size_t im = POTSYNTH_IDX((long) n, -(long) m, (long) lmin, (long) mmax);

          coef[ip] = f * gh[ip];
          if (m > 0)
            coef[im] = f * gh[im];
        }
    }

  for (n = 0; n <= lmax; ++n)
    {
      rfr[n] = n + 1.0;
      rfm[n] = -1.0;
    }

  chan[0].rpow = 2; chan[0].rfac = rfr; chan[0].theta = POTSYNTH_THETA_P;    chan[0].dphi = 0;
  chan[1].rpow = 2; chan[1].rfac = rfm; chan[1].theta = POTSYNTH_THETA_DP;   chan[1].dphi = 0;
  chan[2].rpow = 2; chan[2].rfac = rfm; chan[2].theta = POTSYNTH_THETA_PSIN; chan[2].dphi = 1;

  w = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, 3, chan, params, status);

  free(coef);
  free(rfr);
  free(rfm);
  return w;
}

void
potsynth_mag_rtp2nec(size_t npts, double *B)
{
  size_t i;

  for (i = 0; i < npts; ++i)
    {
      const double br = B[i], bt = B[npts + i], bp = B[2 * npts + i];

      B[i] = -bt;                 /* north  = -B_theta */
      B[npts + i] = bp;           /* east   =  B_phi   */
      B[2 * npts + i] = -br;      /* center = -B_r     */
    }
}
