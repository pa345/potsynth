/*
 * direct.c -- point-by-point synthesis (reference method; fastest at low degree)
 *
 * For a block of PS_NB points, each order m is handled by one column recursion
 * (vectorised across the points), the radial powers u^n are applied to the column
 * values, and all channels are accumulated with the same kernel as the set-up.
 * No ALF array is stored. Pole-safe: P/sin(theta) and dP/dtheta come from the
 * scaled columns and are finite at theta = 0 and pi.
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"

/* precompute the weight rows for every order (called at allocation) */
int
ps_direct_init(potsynth_workspace *w)
{
  const size_t L = w->lmax, L1 = L + 1, nchan = w->nchan;
  size_t m, n, c, total = 0;

  w->doff = malloc(L1 * sizeof(size_t));
  w->dn0 = malloc(L1 * nchan * sizeof(size_t));
  w->dsrc = malloc(nchan);
  if (!w->doff || !w->dn0 || !w->dsrc)
    return POTSYNTH_ENOMEM;

  for (m = 0; m <= L; ++m)
    {
      w->doff[m] = total;
      if (m <= w->mmax)
        total += 2 * nchan * (L - m + 1);
    }

  w->dW = ps_calloc((total ? total : 1) * sizeof(double));
  if (!w->dW)
    return POTSYNTH_ENOMEM;

  for (c = 0; c < nchan; ++c)
    w->dsrc[c] = (w->theta_op[c] == POTSYNTH_THETA_DP);

  for (m = 0; m <= w->mmax; ++m)
    {
      const size_t nn = L - m + 1;
      const double dm = (double) m;

      for (c = 0; c < nchan; ++c)
        {
          double *wa = w->dW + w->doff[m] + 2 * c * nn, *wb = wa + nn;
          const double *rfac = w->rfac + c * L1;
          size_t nstart = (w->lmin > m) ? w->lmin : m;

          if (w->theta_op[c] == POTSYNTH_THETA_PSIN && m == 0)
            nstart = L + 1;
          w->dn0[m * nchan + c] = nstart - m;

          for (n = nstart; n <= L; ++n)
            {
              const size_t idx = ps_alf_idx(n, m, L);

              if (w->dphi[c] == 0)
                {
                  wa[n - m] = rfac[n] * w->C[idx];
                  wb[n - m] = rfac[n] * w->S[idx];
                }
              else
                {
                  wa[n - m] = rfac[n] * dm * w->S[idx];
                  wb[n - m] = -rfac[n] * dm * w->C[idx];
                }
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

void
ps_direct_free(potsynth_workspace *w)
{
  free(w->doff);
  free(w->dn0);
  free(w->dsrc);
  ps_free(w->dW);
  w->doff = NULL;
  w->dn0 = NULL;
  w->dsrc = NULL;
  w->dW = NULL;
}

size_t
ps_direct_scratch_size(const potsynth_workspace *w)
{
  const size_t L1 = w->lmax + 1;

  return 3 * L1 * PS_NB + 5 * w->nchan * PS_NB + 4 * PS_NB;
}

void
ps_direct_block(const potsynth_workspace *w, double *scratch, size_t nb,
                const double *r, const double *theta, const double *phi,
                double *out, size_t ldo)
{
  const size_t L = w->lmax, L1 = L + 1, nchan = w->nchan;
  double *Q = scratch;
  double *E = Q + L1 * PS_NB;
  double *U = E + L1 * PS_NB;
  double *acc = U + L1 * PS_NB;
  double *res = acc + 4 * nchan * PS_NB;
  double *x = res + nchan * PS_NB;
  double *s = x + PS_NB;
  double *u = s + PS_NB;
  double *ph = u + PS_NB;
  size_t j, n, m, c;

  for (j = 0; j < PS_NB; ++j)
    {
      const size_t jj = (j < nb) ? j : 0;    /* pad with a copy of the first point */
      double p = fmod(phi[jj], 2.0 * M_PI);

      if (p < 0.0)
        p += 2.0 * M_PI;

      x[j] = cos(theta[jj]);
      s[j] = sin(theta[jj]);
      u[j] = w->a / r[jj];
      ph[j] = p;
    }

  for (j = 0; j < PS_NB; ++j)
    U[j] = 1.0;
  for (n = 1; n <= L; ++n)
    for (j = 0; j < PS_NB; ++j)
      U[n * PS_NB + j] = U[(n - 1) * PS_NB + j] * u[j];

  memset(res, 0, nchan * PS_NB * sizeof(double));

  for (m = 0; m <= w->mmax; ++m)
    {
      const size_t nn = L - m + 1;
      const int scale = w->alf.scale[m];
      double cm[PS_NB], sm[PS_NB];

      ps_alf_column(&w->alf, m, x, s, Q, E);

      /* radial factors u^n */
      for (n = 0; n < nn; ++n)
        {
          const double *un = U + (m + n) * PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              Q[n * PS_NB + j] *= un[j];
              E[n * PS_NB + j] *= un[j];
            }
        }

      ps_alf_accum(nn, nchan, w->dW + w->doff[m], nn, w->dsrc, w->dn0 + m * nchan, Q, E, acc);

      for (j = 0; j < PS_NB; ++j)
        ps_sincos_m(m, ph[j], &cm[j], &sm[j]);

      for (c = 0; c < nchan; ++c)
        {
          const int op = w->theta_op[c];
          const size_t ps = (m == 0) ? 0 : ((op == POTSYNTH_THETA_P) ? m : m - 1);

          for (j = 0; j < PS_NB; ++j)
            {
              double vA = acc[(4 * c + 0) * PS_NB + j] + acc[(4 * c + 1) * PS_NB + j];
              double vB = acc[(4 * c + 2) * PS_NB + j] + acc[(4 * c + 3) * PS_NB + j];
              double v = vA * cm[j] + vB * sm[j];

              if (m > 0)
                {
                  double mant;
                  int expo;

                  ps_pow_scaled(s[j], ps, &mant, &expo);
                  v = ldexp(v * mant, expo + scale);
                }

              res[c * PS_NB + j] += v;
            }
        }
    }

  for (c = 0; c < nchan; ++c)
    {
      const int rp = w->rpow[c];

      for (j = 0; j < nb; ++j)
        {
          double f = 1.0;
          int k;

          for (k = 0; k < rp; ++k)
            f *= u[j];
          for (k = 0; k > rp; --k)
            f /= u[j];

          out[c * ldo + j] = res[c * PS_NB + j] * f;
        }
    }
}
