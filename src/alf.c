/*
 * alf.c
 *
 * 4-pi fully normalized associated Legendre functions
 *
 * The recursions are those of gsl_sf_alf_vsh_array() (GSL specfunc/alf_P.c),
 * which compute P(l,m)/sin(theta) and dP(l,m)/dtheta without dividing by
 * sin(theta), following
 *
 * [1] S. A. Holmes and W. E. Featherstone, A unified approach to the Clenshaw
 *     summation and the recursive computation of very high degree and order
 *     normalised associated Legendre functions, J. Geodesy 76, 279-299, 2002.
 *
 * Differences from GSL:
 *
 * 1) for m >= 1 the columns are computed for P/sin^m and (dP/dtheta)/sin^(m-1),
 *    which are polynomials in x = cos(theta); the powers of sin(theta) are
 *    applied afterwards by the caller, so the recursion itself never underflows
 *
 * 2) instead of a fixed 1e-280 factor, each column is scaled by an exact power
 *    of 2 derived from the analytic maximum of P/sin^m (attained at the poles),
 *    so most columns are not scaled at all and none can overflow
 *
 * 3) sin(theta) may be negative (doubled sphere); only s^2 enters for m >= 1
 *
 * 4) the recursion runs over a block of POTSYNTH_NB nodes at once (vectorized),
 *    and the values are consumed immediately by a register-blocked accumulation
 *    kernel
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"

/*
potsynth_alf_alloc()
  Allocate ALF workspace and precompute recurrence factors

Inputs: lmax - maximum SH degree

Return: pointer to workspace

Notes:
1) The diagonal Q(m,m) = 2^-scale[m] prod_{k=1}^m d_k, with d_1 = sqrt(3),
d_k = sqrt(1 + 1/(2k)). The scaling is chosen from the maximum of Q(L,m)
at x = 1,

Q(l,m)(1) = sqrt(2(2l+1)) sqrt((l+m)!/(l-m)!) / (2^m m!)

E(l,m) is bounded by about (l+1) times that, hence an extra margin.

2) Returns NULL with POTSYNTH_EUNSUP if lmax is too large for the scaling
*/

potsynth_alf_workspace *
potsynth_alf_alloc(const size_t lmax)
{
  const size_t nlm = (lmax + 1) * (lmax + 2) / 2;
  const double L = (double) lmax;
  potsynth_alf_workspace * w;
  double prod = 1.0;
  size_t l, m;

  w = calloc(1, sizeof(potsynth_alf_workspace));
  if (w == NULL)
    {
      POTSYNTH_ERROR_NULL ("unable to allocate ALF workspace", POTSYNTH_ENOMEM);
    }

  w->lmax = lmax;
  w->ab = potsynth_aligned_calloc(2 * nlm * sizeof(double));
  w->cl = potsynth_aligned_calloc((lmax + 1) * sizeof(double));
  w->qmm = potsynth_aligned_calloc((lmax + 1) * sizeof(double));
  w->scale = calloc(lmax + 1, sizeof(int));
  if (!w->ab || !w->cl || !w->qmm || !w->scale)
    {
      potsynth_alf_free(w);
      POTSYNTH_ERROR_NULL ("unable to allocate ALF arrays", POTSYNTH_ENOMEM);
    }

  /* recurrence factors (GSL_SF_ALF_FOURPI) */
  for (m = 0; m <= lmax; ++m)
    {
      const double dm = (double) m;

      for (l = m + 2; l <= lmax; ++l)
        {
          const size_t idx = potsynth_alf_idx(l, m, lmax);
          const double dl = (double) l;

          w->ab[2 * idx] = (sqrt(2.0 * dl + 1.0) / sqrt(dl + dm)) *
                           (sqrt(2.0 * dl - 1.0) / sqrt(dl - dm));
          w->ab[2 * idx + 1] = -(sqrt(dl + dm - 1.0) / sqrt(dl + dm)) *
                                (sqrt(dl - dm - 1.0) / sqrt(dl - dm)) *
                                (sqrt(2.0 * dl + 1.0) / sqrt(2.0 * dl - 3.0));
        }

      w->cl[m] = sqrt(2.0 * dm + 3.0);
    }

  /* diagonal terms and column scaling */
  w->qmm[0] = 1.0;
  w->scale[0] = 0;
  for (m = 1; m <= lmax; ++m)
    {
      const double dm = (double) m;
      double log2q, need;

      prod *= (m == 1) ? sqrt(3.0) : sqrt(1.0 + 0.5 / dm);

      log2q = 0.5 * log2(2.0 * (2.0 * L + 1.0)) +
              (0.5 * (lgamma(L + dm + 1.0) - lgamma(L - dm + 1.0)) - lgamma(dm + 1.0)) / M_LN2 - dm;
      need = log2q + log2(L + 1.0) + 1.0 - POTSYNTH_ALF_EXPMAX;

      w->scale[m] = (need > 0.0) ? (int) ceil(need) : 0;
      if (w->scale[m] > POTSYNTH_ALF_SCALEMAX)
        {
          potsynth_alf_free(w);
          POTSYNTH_ERROR_NULL ("lmax too large for double precision ALFs", POTSYNTH_EUNSUP);
        }

      w->qmm[m] = ldexp(prod, -w->scale[m]);
    }

  return w;
}

void
potsynth_alf_free(potsynth_alf_workspace * w)
{
  if (w->ab)
    potsynth_aligned_free(w->ab);

  if (w->cl)
    potsynth_aligned_free(w->cl);

  if (w->qmm)
    potsynth_aligned_free(w->qmm);

  if (w->scale)
    free(w->scale);

  free(w);
}

/*
potsynth_alf_column()
  Compute one column (fixed m, l = m..lmax) of the ALF recursion at
POTSYNTH_NB nodes

Inputs: m - SH order
        x - cos(theta) at the nodes, length POTSYNTH_NB
        s - sin(theta) at the nodes, length POTSYNTH_NB (may be negative)
        Q - (output) Q[(l-m)*POTSYNTH_NB + j] = P(l,0) (m = 0) or scaled Q(l,m) (m > 0)
        E - (output) E[(l-m)*POTSYNTH_NB + j] = dP(l,0)/dtheta (m = 0) or scaled E(l,m) (m > 0)
        w - workspace

Notes:
1) Recurrences (m >= 1, s2 = s^2):

Q(m,m)   = qmm[m],               E(m,m)   = m x qmm[m]
Q(m+1,m) = c_m x Q(m,m),         E(m+1,m) = c_m (x E(m,m) - s2 Q(m,m))
Q(l,m)   = a x Q(l-1,m) + b Q(l-2,m)
E(l,m)   = a (x E(l-1,m) - s2 Q(l-1,m)) + b E(l-2,m)

For m = 0 the same recursions apply to P and dP/dtheta with s in place of s2.
*/

void
potsynth_alf_column(const size_t m, const double * restrict x, const double * restrict s,
                    double * restrict Q, double * restrict E, const potsynth_alf_workspace * w)
{
  const size_t lmax = w->lmax;
  const size_t nn = lmax - m + 1;
  const double * ab = w->ab + 2 * potsynth_alf_idx(m, m, lmax);
  double t[POTSYNTH_NB];   /* s (m = 0) or s^2 (m > 0) */
  size_t k, j;

  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      t[j] = (m == 0) ? s[j] : s[j] * s[j];
      Q[j] = w->qmm[m];
      E[j] = (double) m * x[j] * w->qmm[m];
    }

  if (nn == 1)
    return;

  /* l = m + 1 */
  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      Q[POTSYNTH_NB + j] = w->cl[m] * x[j] * Q[j];
      E[POTSYNTH_NB + j] = w->cl[m] * (x[j] * E[j] - t[j] * Q[j]);
    }

  /* l = m + 2 ... lmax */
  for (k = 2; k < nn; ++k)
    {
      const double a = ab[2 * k], b = ab[2 * k + 1];
      const double * restrict q1 = Q + (k - 1) * POTSYNTH_NB;
      const double * restrict q2 = Q + (k - 2) * POTSYNTH_NB;
      const double * restrict e1 = E + (k - 1) * POTSYNTH_NB;
      const double * restrict e2 = E + (k - 2) * POTSYNTH_NB;
      double * restrict q0 = Q + k * POTSYNTH_NB;
      double * restrict e0 = E + k * POTSYNTH_NB;

      for (j = 0; j < POTSYNTH_NB; ++j)
        {
          q0[j] = a * x[j] * q1[j] + b * q2[j];
          e0[j] = a * (x[j] * e1[j] - t[j] * q1[j]) + b * e2[j];
        }
    }
}

/*
potsynth_alf_accum()
  Accumulate weighted sums over l of column values, split by the parity
of l - m, for npair pairs of weight rows (A and B) sharing a source array

Inputs: nn    - column length, lmax - m + 1
        npair - number of pairs
        W     - weights, 2*npair rows of length ldw (row 2k = A, 2k+1 = B)
        ldw   - leading dimension of W
        src   - source of pair k: 0 = Q, 1 = E
        n0    - first index (l - m) of pair k
        Q     - column values from potsynth_alf_column()
        E     - column values from potsynth_alf_column()
        acc   - (output) sums, length 4*npair*POTSYNTH_NB:
                acc[(4k+0)*NB + j] = sum_{l-m even} WA_k[l-m] src_k[l-m][j]
                acc[(4k+1)*NB + j] = sum_{l-m odd}  WA_k[l-m] src_k[l-m][j]
                acc[(4k+2)*NB + j], acc[(4k+3)*NB + j] = same for WB_k
*/

void
potsynth_alf_accum(const size_t nn, const size_t npair, const double * restrict W, const size_t ldw,
                   const unsigned char * src, const size_t * n0,
                   const double * restrict Q, const double * restrict E, double * restrict acc)
{
  size_t k;

  for (k = 0; k < npair; ++k)
    {
      const double * restrict V = src[k] ? E : Q;
      const double * restrict wa = W + 2 * k * ldw;
      const double * restrict wb = wa + ldw;
      double ae[POTSYNTH_NB], ao[POTSYNTH_NB], be[POTSYNTH_NB], bo[POTSYNTH_NB];
      size_t l = n0[k], j;

      for (j = 0; j < POTSYNTH_NB; ++j)
        ae[j] = ao[j] = be[j] = bo[j] = 0.0;

      /* make l even so that (l, l+1) = (even, odd) */
      if ((l & 1) && l < nn)
        {
          const double * restrict v = V + l * POTSYNTH_NB;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              ao[j] += wa[l] * v[j];
              bo[j] += wb[l] * v[j];
            }

          ++l;
        }

      for (; l + 1 < nn; l += 2)
        {
          const double wa0 = wa[l], wa1 = wa[l + 1];
          const double wb0 = wb[l], wb1 = wb[l + 1];
          const double * restrict v0 = V + l * POTSYNTH_NB;
          const double * restrict v1 = v0 + POTSYNTH_NB;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              ae[j] += wa0 * v0[j];
              ao[j] += wa1 * v1[j];
              be[j] += wb0 * v0[j];
              bo[j] += wb1 * v1[j];
            }
        }

      if (l < nn)
        {
          const double * restrict v = V + l * POTSYNTH_NB;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              ae[j] += wa[l] * v[j];
              be[j] += wb[l] * v[j];
            }
        }

      for (j = 0; j < POTSYNTH_NB; ++j)
        {
          acc[(4 * k + 0) * POTSYNTH_NB + j] = ae[j];
          acc[(4 * k + 1) * POTSYNTH_NB + j] = ao[j];
          acc[(4 * k + 2) * POTSYNTH_NB + j] = be[j];
          acc[(4 * k + 3) * POTSYNTH_NB + j] = bo[j];
        }
    }
}
