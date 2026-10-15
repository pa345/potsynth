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
 * The gradient tensor grad B = -grad grad V uses the five tensor channels of
 * potsynth_tensor_channels() with scale -1/a (V = a sum (a/r)^(l+1) ...), and
 * an output map giving its six components.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_mag.h"

/*
potsynth_mag_nout()
  Number of outputs per point for a set of quantities

Inputs: quantities - POTSYNTH_MAG_xxx flags

Return: number of outputs (0 if quantities is invalid)
*/

size_t
potsynth_mag_nout(const int quantities)
{
  if (quantities <= 0 || (quantities & ~(POTSYNTH_MAG_B | POTSYNTH_MAG_GRADB)))
    return 0;

  return ((quantities & POTSYNTH_MAG_B) ? 3 : 0) + ((quantities & POTSYNTH_MAG_GRADB) ? 6 : 0);
}

/*
potsynth_mag_alloc()
  Allocate a magnetic field model

Inputs: lmin       - minimum SH degree (>= 1)
        lmax       - maximum SH degree (<= POTSYNTH_LMAX)
        mmax       - maximum SH order (<= lmax)
        gh         - Schmidt semi-normalized Gauss coefficients (nT), indexed by
                     potsynth_idx(l,m,lmin,mmax): g_l^m at m >= 0, h_l^m at -m;
                     length potsynth_nlm(lmin,lmax,mmax); copied
        a          - reference radius (km)
        rmin       - minimum radius of evaluation points (km), fast method
        rmax       - maximum radius of evaluation points (km), fast method
        quantities - POTSYNTH_MAG_xxx flags (B, GRADB)
        params     - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error

Notes:
1) Evaluate with potsynth_eval(); the outputs are, in this order and as
selected by quantities: B_r, B_theta, B_phi (nT), then the gradient tensor
dB_rr, dB_rt, dB_rp, dB_tt, dB_tp, dB_pp (nT/km), out[k*n + i]. All are
finite and correct at the poles (the theta and phi components there refer to
the local frame defined by phi)

2) The tensor is symmetric and traceless; it is computed from five channels
*/

potsynth_workspace *
potsynth_mag_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * gh,
                   const double a, const double rmin, const double rmax,
                   const int quantities, const potsynth_parameters * params)
{
  const size_t L1 = lmax + 1;
  const size_t nout = potsynth_mag_nout(quantities);
  const int wantB = (quantities & POTSYNTH_MAG_B) != 0;
  const int wantT = (quantities & POTSYNTH_MAG_GRADB) != 0;
  potsynth_channel chan[8];
  potsynth_workspace * w;
  double * coef, * rfac;
  size_t l, m, nchan = 0;

  if (nout == 0)
    {
      POTSYNTH_ERROR_NULL ("invalid quantities", POTSYNTH_EINVAL);
    }
  else if (lmin < 1)
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
  else if (!(a > 0.0) || !isfinite(a))
    {
      POTSYNTH_ERROR_NULL ("reference radius must be positive", POTSYNTH_EINVAL);
    }

  coef = malloc(potsynth_nlm(lmin, lmax, mmax) * sizeof(double));
  rfac = malloc(7 * L1 * sizeof(double));
  if (!coef || !rfac)
    {
      free(coef);
      free(rfac);
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

  if (wantB)
    {
      double * rfac_r = rfac + 5 * L1;
      double * rfac_m = rfac + 6 * L1;

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

      nchan = 3;
    }

  /* grad B = -grad grad V, V = a sum (a/r)^(l+1) ...: scale -a/a^2 */
  if (wantT)
    {
      potsynth_tensor_channels(lmax, -1.0 / a, chan + nchan, rfac);
      nchan += 5;
    }

  w = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, params);

  free(coef);
  free(rfac);

  if (w != NULL && wantT)
    {
      double map[9 * 8] = { 0.0 };
      int status;

      if (wantB)
        {
          map[0 * nchan + 0] = 1.0;
          map[1 * nchan + 1] = 1.0;
          map[2 * nchan + 2] = 1.0;
        }

      potsynth_tensor_map(nchan, wantB ? 3 : 0, map + (wantB ? 3 : 0) * nchan);

      status = potsynth_set_outputs(w, nout, map);
      if (status)
        {
          potsynth_free(w);
          return NULL; /* reported by potsynth_set_outputs() */
        }
    }

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
