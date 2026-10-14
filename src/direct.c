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
  return 3 * w->nchan * PS_NB;
}

/*
 * Sums over n for one order m:
 *
 *   acc[(2c)*NB + j]   = sum_n WA_c[n-m] u_j^n T_c(n,m)(x_j)
 *   acc[(2c+1)*NB + j] = sum_n WB_c[n-m] u_j^n T_c(n,m)(x_j)
 *
 * where T_c is the scaled column value Q or E selected by src[c]. One pass over n
 * runs the column recursion and accumulates every channel; no Legendre values are
 * stored. The first step uses the diagonal values q0, e0; t = s (m = 0) or s^2.
 */

/* generic version: any number of channels, accumulators in memory */
static void
direct_order_generic(size_t nchan, const double *Wm, size_t nn, const double *ab, double cl,
                     const unsigned char *src, const double *x, const double *t,
                     const double *u, const double *um, double q0, const double *e0,
                     double *restrict acc)
{
  double q1[PS_NB], q2[PS_NB] = { 0 }, e1[PS_NB], e2[PS_NB] = { 0 }, un[PS_NB];
  size_t j, k, c;

  memset(acc, 0, 2 * nchan * PS_NB * sizeof(double));

  for (j = 0; j < PS_NB; ++j)
    {
      q1[j] = q0;
      e1[j] = e0[j];
      un[j] = um[j];
    }

  for (k = 0; k < nn; ++k)
    {
      double qv[PS_NB], ev[PS_NB], rq[PS_NB], re[PS_NB];

      if (k == 0)
        {
          for (j = 0; j < PS_NB; ++j)
            {
              qv[j] = q1[j];
              ev[j] = e1[j];
            }
        }
      else
        {
          const double a = (k == 1) ? cl : ab[2 * k];
          const double b = (k == 1) ? 0.0 : ab[2 * k + 1];

          for (j = 0; j < PS_NB; ++j)
            {
              qv[j] = a * x[j] * q1[j] + b * q2[j];
              ev[j] = a * (x[j] * e1[j] - t[j] * q1[j]) + b * e2[j];
              q2[j] = q1[j];
              e2[j] = e1[j];
              q1[j] = qv[j];
              e1[j] = ev[j];
            }
        }

      for (j = 0; j < PS_NB; ++j)
        {
          rq[j] = un[j] * qv[j];
          re[j] = un[j] * ev[j];
          un[j] *= u[j];
        }

      for (c = 0; c < nchan; ++c)
        {
          const double wa = Wm[2 * c * nn + k], wb = Wm[(2 * c + 1) * nn + k];
          const double *restrict v = src[c] ? re : rq;
          double *restrict aa = acc + 2 * c * PS_NB, *restrict bb = aa + PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              aa[j] += wa * v[j];
              bb[j] += wb * v[j];
            }
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)

/* specialised versions for NCH channels: 4 points per vector, all accumulators
 * in vector registers (about 3x faster than the generic version) */
typedef double ps_d4 __attribute__((vector_size(32), aligned(8)));

#define PS_DIRECT_ORDER(NCH)                                                    \
static void                                                                     \
direct_order_##NCH(const double *Wm, size_t nn, const double *ab, double cl,    \
                   const unsigned char *src, ps_d4 x, ps_d4 t, ps_d4 u,          \
                   ps_d4 un, ps_d4 q1, ps_d4 e1, double *acc)                   \
{                                                                               \
  const ps_d4 zero = { 0.0, 0.0, 0.0, 0.0 };                                    \
  ps_d4 A[NCH], B[NCH], q2 = zero, e2 = zero, qv, ev, rq, re;                   \
  size_t k, c;                                                                  \
                                                                                \
  for (c = 0; c < NCH; ++c)                                                     \
    A[c] = B[c] = zero;                                                         \
                                                                                \
  for (k = 0; k < nn; ++k)                                                      \
    {                                                                           \
      if (k == 0)                                                               \
        {                                                                       \
          qv = q1;                                                              \
          ev = e1;                                                              \
        }                                                                       \
      else                                                                      \
        {                                                                       \
          const double a = (k == 1) ? cl : ab[2 * k];                           \
          const double b = (k == 1) ? 0.0 : ab[2 * k + 1];                      \
          qv = a * x * q1 + b * q2;                                             \
          ev = a * (x * e1 - t * q1) + b * e2;                                  \
          q2 = q1;                                                              \
          e2 = e1;                                                              \
          q1 = qv;                                                              \
          e1 = ev;                                                              \
        }                                                                       \
                                                                                \
      rq = un * qv;                                                             \
      re = un * ev;                                                             \
      un *= u;                                                                  \
                                                                                \
      for (c = 0; c < NCH; ++c)                                                 \
        {                                                                       \
          const ps_d4 v = src[c] ? re : rq;                                     \
          A[c] += Wm[2 * c * nn + k] * v;                                       \
          B[c] += Wm[(2 * c + 1) * nn + k] * v;                                 \
        }                                                                       \
    }                                                                           \
                                                                                \
  for (c = 0; c < NCH; ++c)                                                     \
    {                                                                           \
      *(ps_d4 *) (acc + 2 * c * PS_NB) = A[c];                                  \
      *(ps_d4 *) (acc + (2 * c + 1) * PS_NB) = B[c];                            \
    }                                                                           \
}

PS_DIRECT_ORDER(1)
PS_DIRECT_ORDER(2)
PS_DIRECT_ORDER(3)
PS_DIRECT_ORDER(4)
PS_DIRECT_ORDER(5)
PS_DIRECT_ORDER(6)

typedef void (*direct_order_fn)(const double *, size_t, const double *, double,
                                const unsigned char *, ps_d4, ps_d4, ps_d4, ps_d4, ps_d4,
                                ps_d4, double *);

static const direct_order_fn direct_order_vec[7] =
{
  NULL, direct_order_1, direct_order_2, direct_order_3, direct_order_4, direct_order_5,
  direct_order_6
};

#define PS_DIRECT_NVEC 6

#endif

/*
 * One block of PS_NB points. The diagonal scaling, s^(m-1), u^m and
 * cos/sin(m phi) are carried across orders.
 */
void
ps_direct_block(const potsynth_workspace *w, double *scratch, size_t nb,
                const double *r, const double *theta, const double *phi,
                double *out, size_t ldo)
{
  const size_t L = w->lmax, nchan = w->nchan;
  const ps_alf *alf = &w->alf;
  double *restrict acc = scratch;                     /* 2 * nchan * NB */
  double *restrict res = acc + 2 * nchan * PS_NB;     /* nchan * NB */
  double x[PS_NB], s[PS_NB], s2[PS_NB], u[PS_NB], um[PS_NB], ph[PS_NB], e0[PS_NB];
  double c1[PS_NB], s1[PS_NB], cm[PS_NB], sm[PS_NB], pm[PS_NB];
  int pe[PS_NB];
  size_t j, m, c;

  for (j = 0; j < PS_NB; ++j)
    {
      const size_t jj = (j < nb) ? j : 0;    /* pad with a copy of the first point */
      double p = fmod(phi[jj], 2.0 * M_PI);

      if (p < 0.0)
        p += 2.0 * M_PI;

      x[j] = cos(theta[jj]);
      s[j] = sin(theta[jj]);
      s2[j] = s[j] * s[j];
      u[j] = w->a / r[jj];
      um[j] = 1.0;
      ph[j] = p;
      c1[j] = cos(p);
      s1[j] = sin(p);
      cm[j] = 1.0;
      sm[j] = 0.0;
      pm[j] = 1.0;
      pe[j] = 0;
    }

  memset(res, 0, nchan * PS_NB * sizeof(double));

  for (m = 0; m <= w->mmax; ++m)
    {
      const size_t nn = L - m + 1;
      const int scale = alf->scale[m];
      const double *Wm = w->dW + w->doff[m];
      const double *ab = alf->ab + 2 * ps_alf_idx(m, m, L);
      const double *t = (m == 0) ? s : s2;    /* derivative recursion factor */
      const double q0 = (m == 0) ? 1.0 : alf->qmm[m];

      if (m > 0)
        {
          for (j = 0; j < PS_NB; ++j)
            {
              /* cos/sin(m phi) by rotation, exact every 16 orders */
              if ((m & 15) == 0)
                {
                  ps_sincos_m(m, ph[j], &cm[j], &sm[j]);
                }
              else
                {
                  const double ct = cm[j] * c1[j] - sm[j] * s1[j];
                  sm[j] = sm[j] * c1[j] + cm[j] * s1[j];
                  cm[j] = ct;
                }

              /* s^(m-1) as mantissa/exponent, renormalised only near underflow */
              if (m > 1)
                {
                  pm[j] *= s[j];
                  if (fabs(pm[j]) < 0x1p-500 && pm[j] != 0.0)
                    {
                      int e;
                      pm[j] = frexp(pm[j], &e);
                      pe[j] += e;
                    }
                }

              um[j] *= u[j];
            }
        }

      for (j = 0; j < PS_NB; ++j)
        e0[j] = (double) m * x[j] * q0;     /* E(m,m) = m x Q(m,m); 0 for m = 0 */

#ifdef PS_DIRECT_NVEC
      if (nchan <= PS_DIRECT_NVEC)
        {
          size_t h, q;

          for (h = 0; h < PS_NB; h += 4)
            {
              ps_d4 xv, tv, uv, unv, qv, ev;

              for (q = 0; q < 4; ++q)
                {
                  xv[q] = x[h + q];
                  tv[q] = t[h + q];
                  uv[q] = u[h + q];
                  unv[q] = um[h + q];
                  qv[q] = q0;
                  ev[q] = e0[h + q];
                }

              direct_order_vec[nchan](Wm, nn, ab, alf->cl[m], w->dsrc, xv, tv, uv, unv, qv, ev,
                                      acc + h);
            }
        }
      else
#endif
        {
          direct_order_generic(nchan, Wm, nn, ab, alf->cl[m], w->dsrc, x, t, u, um, q0, e0,
                               acc);
        }

      for (c = 0; c < nchan; ++c)
        {
          /* P carries s^m, P/sin and dP/dtheta carry s^(m-1) */
          const int sP = (m > 0 && w->theta_op[c] == POTSYNTH_THETA_P);
          const double *aa = acc + 2 * c * PS_NB, *bb = aa + PS_NB;

          for (j = 0; j < PS_NB; ++j)
            {
              double v = aa[j] * cm[j] + bb[j] * sm[j];

              if (m > 0)
                {
                  const double mant = sP ? pm[j] * s[j] : pm[j];
                  const int expo = pe[j] + scale;

                  v = (expo == 0) ? v * mant : ldexp(v * mant, expo);
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
