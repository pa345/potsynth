/*
 * alf.c -- 4-pi fully normalised associated Legendre functions
 *
 * The recursions are those of gsl_sf_alf_vsh_array() (GSL specfunc/alf_P.c,
 * written by Patrick Alken and relicensed here under BSD-3), which compute
 * P(n,m)/sin(theta) and dP(n,m)/dtheta without dividing by sin(theta), following
 *
 *   S. A. Holmes and W. E. Featherstone, A unified approach to the Clenshaw
 *   summation and the recursive computation of very high degree and order
 *   normalised associated Legendre functions, J. Geodesy 76, 279-299, 2002.
 *
 * Differences from GSL:
 *   - for m >= 1 the columns are computed for P/sin^m and (dP/dtheta)/sin^(m-1),
 *     which are polynomials in x = cos(theta); the powers of sin(theta) are applied
 *     afterwards by the caller, so the recursion itself never underflows;
 *   - instead of a fixed 1e-280 factor, each column is scaled by an exact power of
 *     two derived from the analytic maximum of P/sin^m (attained at the poles), so
 *     most columns are not scaled at all and none can overflow, even at the poles;
 *   - sin(theta) may be negative (doubled sphere); only s^2 enters for m >= 1;
 *   - the recursion runs over a block of PS_NB nodes at once (vectorised), and the
 *     values are consumed immediately by a register-blocked accumulation kernel.
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"

int
ps_alf_init(ps_alf *alf, size_t lmax)
{
  const size_t nlm = (lmax + 1) * (lmax + 2) / 2;
  const double L = (double) lmax;
  double prod = 1.0;
  size_t l, m;

  memset(alf, 0, sizeof(ps_alf));
  alf->lmax = lmax;
  alf->ab = ps_calloc(2 * nlm * sizeof(double));
  alf->cl = ps_calloc((lmax + 1) * sizeof(double));
  alf->qmm = ps_calloc((lmax + 1) * sizeof(double));
  alf->scale = calloc(lmax + 1, sizeof(int));
  if (!alf->ab || !alf->cl || !alf->qmm || !alf->scale)
    {
      ps_alf_free(alf);
      return POTSYNTH_ENOMEM;
    }

  /* recurrence factors (GSL_SF_ALF_FOURPI) */
  for (m = 0; m <= lmax; ++m)
    {
      for (l = m + 2; l <= lmax; ++l)
        {
          const size_t k = ps_alf_idx(l, m, lmax);
          const double dl = (double) l, dm = (double) m;

          alf->ab[2 * k] = (sqrt(2.0 * dl + 1.0) / sqrt(dl + dm)) *
                           (sqrt(2.0 * dl - 1.0) / sqrt(dl - dm));
          alf->ab[2 * k + 1] = -(sqrt(dl + dm - 1.0) / sqrt(dl + dm)) *
                                (sqrt(dl - dm - 1.0) / sqrt(dl - dm)) *
                                (sqrt(2.0 * dl + 1.0) / sqrt(2.0 * dl - 3.0));
        }

      alf->cl[m] = sqrt(2.0 * m + 3.0);
    }

  /* diagonal Q(m,m) = prod_{k=1}^m d_k, d_1 = sqrt(3), d_k = sqrt(1 + 1/(2k)),
   * with a power-of-2 scaling chosen from the maximum of Q(L,m) at x = 1:
   *   Q(l,m)(1) = sqrt(2(2l+1)) sqrt((l+m)!/(l-m)!) / (2^m m!)
   * E(l,m) is bounded by about (l+1) times that, hence the extra margin. */
  alf->qmm[0] = 1.0;
  alf->scale[0] = 0;
  for (m = 1; m <= lmax; ++m)
    {
      const double dm = (double) m;
      double log2q, need;

      prod *= (m == 1) ? sqrt(3.0) : sqrt(1.0 + 0.5 / dm);

      log2q = 0.5 * log2(2.0 * (2.0 * L + 1.0)) +
              (0.5 * (lgamma(L + dm + 1.0) - lgamma(L - dm + 1.0)) - lgamma(dm + 1.0)) / M_LN2 -
              dm;
      need = log2q + log2(L + 1.0) + 1.0 - PS_ALF_EXPMAX;

      alf->scale[m] = (need > 0.0) ? (int) ceil(need) : 0;
      if (alf->scale[m] > PS_ALF_SCALEMAX)
        {
          ps_alf_free(alf);
          return POTSYNTH_EUNSUP;
        }

      alf->qmm[m] = ldexp(prod, -alf->scale[m]);
    }

  return POTSYNTH_SUCCESS;
}

void
ps_alf_free(ps_alf *alf)
{
  ps_free(alf->ab);
  ps_free(alf->cl);
  ps_free(alf->qmm);
  free(alf->scale);
  memset(alf, 0, sizeof(ps_alf));
}

void
ps_alf_column(const ps_alf *alf, size_t m, const double *restrict x,
              const double *restrict s, double *restrict Q, double *restrict E)
{
  const size_t lmax = alf->lmax;
  const size_t nn = lmax - m + 1;
  const double *ab = alf->ab + 2 * ps_alf_idx(m, m, lmax);
  size_t k, j;

  if (m == 0)
    {
      /* P(0,0) = 1, dP(0,0) = 0; P(1,0) = sqrt(3) x, dP(1,0)/dtheta = -sqrt(3) s */
      for (j = 0; j < PS_NB; ++j)
        {
          Q[j] = 1.0;
          E[j] = 0.0;
        }

      if (nn == 1)
        return;

      for (j = 0; j < PS_NB; ++j)
        {
          Q[PS_NB + j] = alf->cl[0] * x[j];
          E[PS_NB + j] = -alf->cl[0] * s[j];
        }

      for (k = 2; k < nn; ++k)
        {
          const double a = ab[2 * k], b = ab[2 * k + 1];
          const double *restrict q1 = Q + (k - 1) * PS_NB, *restrict q2 = Q + (k - 2) * PS_NB;
          const double *restrict e1 = E + (k - 1) * PS_NB, *restrict e2 = E + (k - 2) * PS_NB;
          double *restrict q0 = Q + k * PS_NB, *restrict e0 = E + k * PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              q0[j] = a * x[j] * q1[j] + b * q2[j];
              e0[j] = a * (x[j] * e1[j] - s[j] * q1[j]) + b * e2[j];
            }
        }
    }
  else
    {
      const double qmm = alf->qmm[m];
      const double dm = (double) m;
      const double c = alf->cl[m];
      double s2[PS_NB];

      for (j = 0; j < PS_NB; ++j)
        {
          s2[j] = s[j] * s[j];
          Q[j] = qmm;
          E[j] = dm * x[j] * qmm;
        }

      if (nn == 1)
        return;

      for (j = 0; j < PS_NB; ++j)
        {
          Q[PS_NB + j] = c * x[j] * Q[j];
          E[PS_NB + j] = c * (x[j] * E[j] - s2[j] * Q[j]);
        }

      for (k = 2; k < nn; ++k)
        {
          const double a = ab[2 * k], b = ab[2 * k + 1];
          const double *restrict q1 = Q + (k - 1) * PS_NB, *restrict q2 = Q + (k - 2) * PS_NB;
          const double *restrict e1 = E + (k - 1) * PS_NB, *restrict e2 = E + (k - 2) * PS_NB;
          double *restrict q0 = Q + k * PS_NB, *restrict e0 = E + k * PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              q0[j] = a * x[j] * q1[j] + b * q2[j];
              e0[j] = a * (x[j] * e1[j] - s2[j] * q1[j]) + b * e2[j];
            }
        }
    }
}

void
ps_alf_accum(size_t nn, size_t npair, const double *restrict W, size_t ldw,
             const unsigned char *src, const size_t *n0,
             const double *restrict Q, const double *restrict E, double *restrict acc)
{
  size_t k;

  for (k = 0; k < npair; ++k)
    {
      const double *restrict V = src[k] ? E : Q;
      const double *restrict wa = W + 2 * k * ldw;
      const double *restrict wb = wa + ldw;
      double ae[PS_NB], ao[PS_NB], be[PS_NB], bo[PS_NB];
      size_t n = n0[k], j;

      for (j = 0; j < PS_NB; ++j)
        ae[j] = ao[j] = be[j] = bo[j] = 0.0;

      /* make n even so that (n, n+1) = (even, odd) */
      if ((n & 1) && n < nn)
        {
          const double *restrict v = V + n * PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              ao[j] += wa[n] * v[j];
              bo[j] += wb[n] * v[j];
            }
          ++n;
        }

      for (; n + 1 < nn; n += 2)
        {
          const double wa0 = wa[n], wa1 = wa[n + 1];
          const double wb0 = wb[n], wb1 = wb[n + 1];
          const double *restrict v0 = V + n * PS_NB;
          const double *restrict v1 = v0 + PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              ae[j] += wa0 * v0[j];
              ao[j] += wa1 * v1[j];
              be[j] += wb0 * v0[j];
              bo[j] += wb1 * v1[j];
            }
        }

      if (n < nn)
        {
          const double *restrict v = V + n * PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              ae[j] += wa[n] * v[j];
              be[j] += wb[n] * v[j];
            }
        }

      for (j = 0; j < PS_NB; ++j)
        {
          acc[(4 * k + 0) * PS_NB + j] = ae[j];
          acc[(4 * k + 1) * PS_NB + j] = ao[j];
          acc[(4 * k + 2) * PS_NB + j] = be[j];
          acc[(4 * k + 3) * PS_NB + j] = bo[j];
        }
    }
}
