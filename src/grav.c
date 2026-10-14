/*
 * grav.c
 *
 * Gravity front-end: potential and gravitational acceleration g = grad V for
 * 4-pi fully normalized coefficients,
 *
 * V       =  GM/a \sum (a/r)^(l+1) [ C cos(m phi) + S sin(m phi) ] Pbar_l^m
 * g_r     = -GM/a^2 \sum (l+1) (a/r)^(l+2) [ C cos(m phi) + S sin(m phi) ] Pbar_l^m
 * g_theta =  GM/a^2 \sum (a/r)^(l+2) [ C cos(m phi) + S sin(m phi) ] dPbar_l^m/dtheta
 * g_phi   =  GM/a^2 \sum (a/r)^(l+2) d/dphi [ C cos(m phi) + S sin(m phi) ] Pbar_l^m / sin(theta)
 *
 * The coefficients are already in the core normalization.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_grav.h"

/*
potsynth_grav_nchan()
  Number of output channels for a set of quantities

Inputs: quantities - POTSYNTH_GRAV_xxx flags

Return: number of channels (0 if quantities is invalid)
*/

size_t
potsynth_grav_nchan(const int quantities)
{
  size_t n = 0;

  if (quantities <= 0 || (quantities & ~(POTSYNTH_GRAV_V | POTSYNTH_GRAV_GR | POTSYNTH_GRAV_G)))
    return 0;

  if (quantities & POTSYNTH_GRAV_V)
    ++n;

  if (quantities & POTSYNTH_GRAV_G)
    n += 3;
  else if (quantities & POTSYNTH_GRAV_GR)
    ++n;

  return n;
}

/*
potsynth_grav_alloc()
  Allocate a gravity field model

Inputs: lmin       - minimum SH degree (>= 0)
        lmax       - maximum SH degree (<= POTSYNTH_LMAX)
        mmax       - maximum SH order (<= lmax)
        coef       - 4-pi fully normalized coefficients, indexed by
                     potsynth_idx(l,m,lmin,mmax): C_lm at m >= 0, S_lm at -m;
                     length potsynth_nlm(lmin,lmax,mmax); copied
        GM         - gravitational constant times mass (m^3/s^2)
        a          - reference radius (m)
        rmin       - minimum radius of evaluation points (m), fast method
        rmax       - maximum radius of evaluation points (m), fast method
        quantities - POTSYNTH_GRAV_xxx flags (V, GR, G)
        params     - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error

Notes:
1) Evaluate with potsynth_eval(), with r in m. The output channels are, in
this order and as selected by quantities: V (m^2/s^2), g_r (m/s^2), and
g_theta, g_phi (m/s^2, with POTSYNTH_GRAV_G). g_r is negative (pointing
towards the center), g_theta is positive southwards

2) Each extra channel adds to the set-up time and grid memory of the fast
method; request only the quantities needed

3) The accuracy eps of the fast method is relative to the maximum of each
channel, which for a full model is dominated by the central (l = 0) term
*/

potsynth_workspace *
potsynth_grav_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * coef,
                    const double GM, const double a, const double rmin, const double rmax,
                    const int quantities, const potsynth_parameters * params)
{
  const size_t L1 = lmax + 1;
  const size_t nchan = potsynth_grav_nchan(quantities);
  potsynth_channel chan[4];
  potsynth_workspace * w;
  double * rfac_v, * rfac_r, * rfac_g;
  size_t c = 0, l;

  if (nchan == 0)
    {
      POTSYNTH_ERROR_NULL ("invalid quantities", POTSYNTH_EINVAL);
    }
  else if (!(GM > 0.0) || !isfinite(GM))
    {
      POTSYNTH_ERROR_NULL ("GM must be positive", POTSYNTH_EINVAL);
    }
  else if (!(a > 0.0) || !isfinite(a))
    {
      POTSYNTH_ERROR_NULL ("reference radius must be positive", POTSYNTH_EINVAL);
    }
  else if (lmax > POTSYNTH_LMAX)
    {
      POTSYNTH_ERROR_NULL ("lmax exceeds POTSYNTH_LMAX", POTSYNTH_EUNSUP);
    }

  rfac_v = malloc(L1 * sizeof(double));
  rfac_r = malloc(L1 * sizeof(double));
  rfac_g = malloc(L1 * sizeof(double));
  if (!rfac_v || !rfac_r || !rfac_g)
    {
      free(rfac_v);
      free(rfac_r);
      free(rfac_g);
      POTSYNTH_ERROR_NULL ("unable to allocate radial factors", POTSYNTH_ENOMEM);
    }

  for (l = 0; l <= lmax; ++l)
    {
      rfac_v[l] = GM / a;
      rfac_r[l] = -(GM / (a * a)) * (l + 1.0);
      rfac_g[l] = GM / (a * a);
    }

  /* V */
  if (quantities & POTSYNTH_GRAV_V)
    {
      chan[c].rpow = 1;
      chan[c].rfac = rfac_v;
      chan[c].theta = POTSYNTH_THETA_P;
      chan[c].dphi = 0;
      ++c;
    }

  /* g_r */
  if (quantities & (POTSYNTH_GRAV_GR | POTSYNTH_GRAV_G))
    {
      chan[c].rpow = 2;
      chan[c].rfac = rfac_r;
      chan[c].theta = POTSYNTH_THETA_P;
      chan[c].dphi = 0;
      ++c;
    }

  if (quantities & POTSYNTH_GRAV_G)
    {
      /* g_theta */
      chan[c].rpow = 2;
      chan[c].rfac = rfac_g;
      chan[c].theta = POTSYNTH_THETA_DP;
      chan[c].dphi = 0;
      ++c;

      /* g_phi */
      chan[c].rpow = 2;
      chan[c].rfac = rfac_g;
      chan[c].theta = POTSYNTH_THETA_PSIN;
      chan[c].dphi = 1;
      ++c;
    }

  w = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, params);

  free(rfac_v);
  free(rfac_r);
  free(rfac_g);

  return w;
}
