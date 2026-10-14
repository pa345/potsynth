/*
 * mag.c
 *
 * Magnetic front-end: B = -grad V for Schmidt semi-normalized Gauss
 * coefficients,
 *
 * B_r     =  \sum (l+1) (a/r)^(l+2) [ g cos(m phi) + h sin(m phi) ] P_l^m
 * B_theta = -\sum (a/r)^(l+2) [ g cos(m phi) + h sin(m phi) ] dP_l^m/dtheta
 * B_phi   = -\sum (a/r)^(l+2) d/dphi [ g cos(m phi) + h sin(m phi) ] P_l^m / sin(theta)
 *
 * Schmidt P_l^m = Pbar_l^m / sqrt(2l+1) (4-pi normalized), so the core
 * coefficients are C_lm = g_l^m / sqrt(2l+1), S_lm = h_l^m / sqrt(2l+1).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_mag.h"

/*
potsynth_mag_alloc()
  Allocate a magnetic field model, with channels (B_r, B_theta, B_phi)

Inputs: lmin   - minimum SH degree (>= 1)
        lmax   - maximum SH degree (<= POTSYNTH_LMAX)
        mmax   - maximum SH order (<= lmax)
        gh     - Schmidt semi-normalized Gauss coefficients (nT), indexed by
                 potsynth_idx(l,m,lmin,mmax): g_l^m at m >= 0, h_l^m at -m;
                 length potsynth_nlm(lmin,lmax,mmax); copied
        a      - reference radius (km)
        rmin   - minimum radius of evaluation points (km), fast method
        rmax   - maximum radius of evaluation points (km), fast method
        params - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error

Notes:
1) Evaluate with potsynth_eval(); out[0*n + i] = B_r, out[1*n + i] = B_theta,
out[2*n + i] = B_phi, in nT. All three components are finite and correct at the
poles (B_theta and B_phi there refer to the local frame defined by phi)
*/

potsynth_workspace *
potsynth_mag_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * gh,
                   const double a, const double rmin, const double rmax,
                   const potsynth_parameters * params)
{
  const size_t L1 = lmax + 1;
  potsynth_channel chan[3];
  potsynth_workspace * w;
  double * coef, * rfac_r, * rfac_m;
  size_t l, m;

  if (lmin < 1)
    {
      POTSYNTH_ERROR_NULL ("lmin must be >= 1", POTSYNTH_EINVAL);
    }
  else if (lmin > lmax)
    {
      POTSYNTH_ERROR_NULL ("lmin must be <= lmax", POTSYNTH_EINVAL);
    }
  else if (mmax > lmax)
    {
      POTSYNTH_ERROR_NULL ("mmax must be <= lmax", POTSYNTH_EINVAL);
    }
  else if (lmax > POTSYNTH_LMAX)
    {
      POTSYNTH_ERROR_NULL ("lmax exceeds POTSYNTH_LMAX", POTSYNTH_EUNSUP);
    }
  else if (gh == NULL)
    {
      POTSYNTH_ERROR_NULL ("gh must be given", POTSYNTH_EINVAL);
    }

  coef = malloc(potsynth_nlm(lmin, lmax, mmax) * sizeof(double));
  rfac_r = malloc(L1 * sizeof(double));
  rfac_m = malloc(L1 * sizeof(double));
  if (!coef || !rfac_r || !rfac_m)
    {
      free(coef);
      free(rfac_r);
      free(rfac_m);
      POTSYNTH_ERROR_NULL ("unable to allocate coefficients", POTSYNTH_ENOMEM);
    }

  /* Schmidt -> 4-pi normalization */
  for (l = lmin; l <= lmax; ++l)
    {
      const size_t M = (l < mmax) ? l : mmax;
      const double f = 1.0 / sqrt(2.0 * l + 1.0);

      for (m = 0; m <= M; ++m)
        {
          const size_t ip = potsynth_idx((long) l, (long) m, (long) lmin, (long) mmax);
          const size_t im = potsynth_idx((long) l, -(long) m, (long) lmin, (long) mmax);

          coef[ip] = f * gh[ip];
          if (m > 0)
            coef[im] = f * gh[im];
        }
    }

  for (l = 0; l <= lmax; ++l)
    {
      rfac_r[l] = l + 1.0;
      rfac_m[l] = -1.0;
    }

  /* B_r */
  chan[0].rpow = 2;
  chan[0].rfac = rfac_r;
  chan[0].theta = POTSYNTH_THETA_P;
  chan[0].dphi = 0;

  /* B_theta */
  chan[1].rpow = 2;
  chan[1].rfac = rfac_m;
  chan[1].theta = POTSYNTH_THETA_DP;
  chan[1].dphi = 0;

  /* B_phi */
  chan[2].rpow = 2;
  chan[2].rfac = rfac_m;
  chan[2].theta = POTSYNTH_THETA_PSIN;
  chan[2].dphi = 1;

  w = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, 3, chan, params);

  free(coef);
  free(rfac_r);
  free(rfac_m);

  return w;
}

/*
potsynth_mag_rtp2nec()
  Convert field components in place from (B_r, B_theta, B_phi) to
(B_north, B_east, B_center) = (-B_theta, B_phi, -B_r)

Inputs: n - number of points
        B - field, channel-major as output by potsynth_eval(), length 3*n
*/

void
potsynth_mag_rtp2nec(const size_t n, double * B)
{
  size_t i;

  for (i = 0; i < n; ++i)
    {
      const double Br = B[i];
      const double Bt = B[n + i];
      const double Bp = B[2 * n + i];

      B[i] = -Bt;
      B[n + i] = Bp;
      B[2 * n + i] = -Br;
    }
}
