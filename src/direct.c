/*
 * direct.c
 *
 * Point-by-point synthesis (reference method; fastest at low degree)
 *
 * For a block of POTSYNTH_NB points and each order m, a single pass over l runs
 * the column recursion (vectorized across points), applies the running radial
 * factor u^l, and accumulates every channel; no Legendre values are stored.
 * P/sin(theta) and dP/dtheta come from the scaled columns and are finite at the
 * poles.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"

static void direct_order_generic(const size_t nchan, const double * Wm, const size_t nn,
                                 const double * ab, const double cl, const int * isE,
                                 const double * x, const double * t, const double * u,
                                 const double * um, const double q0, const double * e0,
                                 double * acc);

/*
potsynth_direct_init()
  Precompute the weight rows of the direct method for each order

Inputs: w - workspace

Return: success or error

Notes:
1) For order m and channel c, rows A (cos m phi) and B (sin m phi) of length
lmax-m+1 are stored at dW + doff[m] + 2c*(lmax-m+1); entries below the first
contributing degree are 0
*/

int
potsynth_direct_init(potsynth_workspace * w)
{
  const size_t L = w->lmax;
  const size_t L1 = L + 1;
  const size_t nchan = w->nchan;
  size_t m, l, c, total = 0;

  w->doff = malloc(L1 * sizeof(size_t));
  w->isE = malloc(nchan * sizeof(int));
  if (w->doff == NULL || w->isE == NULL)
    {
      POTSYNTH_ERROR ("unable to allocate direct method arrays", POTSYNTH_ENOMEM);
    }

  for (c = 0; c < nchan; ++c)
    w->isE[c] = (w->theta_op[c] == POTSYNTH_THETA_DP);

  for (m = 0; m <= L; ++m)
    {
      w->doff[m] = total;
      if (m <= w->mmax)
        total += 2 * nchan * (L - m + 1);
    }

  w->dW = potsynth_aligned_calloc((total > 0 ? total : 1) * sizeof(double));
  if (w->dW == NULL)
    {
      POTSYNTH_ERROR ("unable to allocate direct weights", POTSYNTH_ENOMEM);
    }

  for (m = 0; m <= w->mmax; ++m)
    {
      const size_t nn = L - m + 1;
      const double dm = (double) m;

      for (c = 0; c < nchan; ++c)
        {
          const double * rfac = w->rfac + c * L1;
          double * wa = w->dW + w->doff[m] + 2 * c * nn;
          double * wb = wa + nn;
          size_t lstart = (w->lmin > m) ? w->lmin : m;

          if (w->theta_op[c] == POTSYNTH_THETA_PSIN && m == 0)
            lstart = L + 1; /* P/sin is not defined for m = 0 */

          for (l = lstart; l <= L; ++l)
            {
              const size_t idx = potsynth_alf_idx(l, m, L);

              if (w->dphi[c] == 0)
                {
                  wa[l - m] = rfac[l] * w->C[idx];
                  wb[l - m] = rfac[l] * w->S[idx];
                }
              else
                {
                  wa[l - m] = rfac[l] * dm * w->S[idx];
                  wb[l - m] = -rfac[l] * dm * w->C[idx];
                }
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

void
potsynth_direct_free(potsynth_workspace * w)
{
  if (w->doff)
    free(w->doff);

  if (w->isE)
    free(w->isE);

  if (w->dW)
    potsynth_aligned_free(w->dW);
}

/* scratch space (doubles) per thread for potsynth_direct_block() */
size_t
potsynth_direct_scratch_size(const potsynth_workspace * w)
{
  return 3 * w->nchan * POTSYNTH_NB;
}

#ifdef POTSYNTH_HAVE_VECEXT

/* 4 doubles; aligned(8) permits unaligned access */
typedef double potsynth_d4 __attribute__((vector_size(32), aligned(8)));

/*
 * direct_order_NCH()
 *   Sums over l for one order m, 4 points and NCH channels, with all
 * accumulators in vector registers (about 3x faster than the generic version):
 *
 * acc[(2c)*NB + j]   = sum_l WA_c[l-m] u_j^l T_c(l,m)(x_j)
 * acc[(2c+1)*NB + j] = sum_l WB_c[l-m] u_j^l T_c(l,m)(x_j)
 *
 * where T_c is the scaled column value Q, or E if isE[c]
 */
#define POTSYNTH_DIRECT_ORDER(NCH)                                                   \
static void                                                                          \
direct_order_##NCH(const double * Wm, const size_t nn, const double * ab,            \
                   const double cl, const int * isE, potsynth_d4 x, potsynth_d4 t,   \
                   potsynth_d4 u, potsynth_d4 un, potsynth_d4 q1, potsynth_d4 e1,    \
                   double * acc)                                                     \
{                                                                                    \
  const potsynth_d4 zero = { 0.0, 0.0, 0.0, 0.0 };                                   \
  potsynth_d4 A[NCH], B[NCH], q2 = zero, e2 = zero, qv, ev, rq, re;                  \
  size_t k, c;                                                                       \
                                                                                     \
  for (c = 0; c < NCH; ++c)                                                          \
    A[c] = B[c] = zero;                                                              \
                                                                                     \
  for (k = 0; k < nn; ++k)                                                           \
    {                                                                                \
      if (k == 0)                                                                    \
        {                                                                            \
          qv = q1;                                                                   \
          ev = e1;                                                                   \
        }                                                                            \
      else                                                                           \
        {                                                                            \
          const double a = (k == 1) ? cl : ab[2 * k];                                \
          const double b = (k == 1) ? 0.0 : ab[2 * k + 1];                           \
                                                                                     \
          qv = a * x * q1 + b * q2;                                                  \
          ev = a * (x * e1 - t * q1) + b * e2;                                       \
          q2 = q1;                                                                   \
          e2 = e1;                                                                   \
          q1 = qv;                                                                   \
          e1 = ev;                                                                   \
        }                                                                            \
                                                                                     \
      rq = un * qv;                                                                  \
      re = un * ev;                                                                  \
      un *= u;                                                                       \
                                                                                     \
      for (c = 0; c < NCH; ++c)                                                      \
        {                                                                            \
          const potsynth_d4 v = isE[c] ? re : rq;                                    \
                                                                                     \
          A[c] += Wm[2 * c * nn + k] * v;                                            \
          B[c] += Wm[(2 * c + 1) * nn + k] * v;                                      \
        }                                                                            \
    }                                                                                \
                                                                                     \
  for (c = 0; c < NCH; ++c)                                                          \
    {                                                                                \
      *(potsynth_d4 *) (acc + 2 * c * POTSYNTH_NB) = A[c];                           \
      *(potsynth_d4 *) (acc + (2 * c + 1) * POTSYNTH_NB) = B[c];                     \
    }                                                                                \
}

POTSYNTH_DIRECT_ORDER(1)
POTSYNTH_DIRECT_ORDER(2)
POTSYNTH_DIRECT_ORDER(3)
POTSYNTH_DIRECT_ORDER(4)
POTSYNTH_DIRECT_ORDER(5)
POTSYNTH_DIRECT_ORDER(6)

typedef void (*direct_order_fn)(const double *, const size_t, const double *, const double,
                                const int *, potsynth_d4, potsynth_d4, potsynth_d4, potsynth_d4,
                                potsynth_d4, potsynth_d4, double *);

#define POTSYNTH_DIRECT_NVEC         6 /* channels with specialized kernels */

static const direct_order_fn direct_order_vec[POTSYNTH_DIRECT_NVEC + 1] =
{
  NULL, direct_order_1, direct_order_2, direct_order_3, direct_order_4, direct_order_5,
  direct_order_6
};

#endif /* POTSYNTH_HAVE_VECEXT */

/*
potsynth_direct_block()
  Evaluate all channels at a block of up to POTSYNTH_NB points

Inputs: nb      - number of points (<= POTSYNTH_NB)
        r       - radius, length nb
        theta   - colatitude (radians), length nb
        phi     - longitude (radians), length nb
        ldo     - leading dimension of out
        out     - (output) out[c*ldo + j]
        scratch - scratch space, length potsynth_direct_scratch_size()
        w       - workspace

Notes:
1) The diagonal scaling, s^(m-1), u^m and cos/sin(m phi) are carried across
orders; cos/sin(m phi) by rotation, recomputed exactly every 16 orders so that
rounding errors cannot accumulate

2) s^(m-1) is kept as mantissa and exponent, renormalized only near underflow
*/

void
potsynth_direct_block(const size_t nb, const double * r, const double * theta, const double * phi,
                      const size_t ldo, double * out, double * scratch, const potsynth_workspace * w)
{
  const size_t L = w->lmax;
  const size_t nchan = w->nchan;
  const potsynth_alf_workspace * alf = w->alf_workspace_p;
  double * restrict acc = scratch;                       /* 2*nchan*NB */
  double * restrict res = acc + 2 * nchan * POTSYNTH_NB; /* nchan*NB */
  double x[POTSYNTH_NB], s[POTSYNTH_NB], s2[POTSYNTH_NB], u[POTSYNTH_NB], um[POTSYNTH_NB];
  double ph[POTSYNTH_NB], e0[POTSYNTH_NB];
  double c1[POTSYNTH_NB], s1[POTSYNTH_NB], cm[POTSYNTH_NB], sm[POTSYNTH_NB], pm[POTSYNTH_NB];
  int pe[POTSYNTH_NB];
  size_t j, m, c;

  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      const size_t jj = (j < nb) ? j : 0; /* pad with a copy of the first point */
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

  memset(res, 0, nchan * POTSYNTH_NB * sizeof(double));

  for (m = 0; m <= w->mmax; ++m)
    {
      const size_t nn = L - m + 1;
      const int scale = alf->scale[m];
      const double * Wm = w->dW + w->doff[m];
      const double * ab = alf->ab + 2 * potsynth_alf_idx(m, m, L);
      const double * t = (m == 0) ? s : s2; /* derivative recursion factor */
      const double q0 = alf->qmm[m];

      if (m > 0)
        {
          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              /* cos/sin(m phi) */
              if ((m & 15) == 0)
                {
                  potsynth_sincos_m(m, ph[j], &cm[j], &sm[j]);
                }
              else
                {
                  const double ct = cm[j] * c1[j] - sm[j] * s1[j];

                  sm[j] = sm[j] * c1[j] + cm[j] * s1[j];
                  cm[j] = ct;
                }

              /* s^(m-1) */
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

      /* E(m,m) = m x Q(m,m) */
      for (j = 0; j < POTSYNTH_NB; ++j)
        e0[j] = (double) m * x[j] * q0;

#ifdef POTSYNTH_HAVE_VECEXT
      if (nchan <= POTSYNTH_DIRECT_NVEC)
        {
          size_t h, q;

          for (h = 0; h < POTSYNTH_NB; h += 4)
            {
              potsynth_d4 xv, tv, uv, unv, qv, ev;

              for (q = 0; q < 4; ++q)
                {
                  xv[q] = x[h + q];
                  tv[q] = t[h + q];
                  uv[q] = u[h + q];
                  unv[q] = um[h + q];
                  qv[q] = q0;
                  ev[q] = e0[h + q];
                }

              (direct_order_vec[nchan])(Wm, nn, ab, alf->cl[m], w->isE, xv, tv, uv, unv, qv, ev,
                                        acc + h);
            }
        }
      else
#endif
        {
          direct_order_generic(nchan, Wm, nn, ab, alf->cl[m], w->isE, x, t, u, um, q0, e0, acc);
        }

      for (c = 0; c < nchan; ++c)
        {
          /* P carries s^m, P/sin and dP/dtheta carry s^(m-1) */
          const int sP = (m > 0 && w->theta_op[c] == POTSYNTH_THETA_P);
          const double * aa = acc + 2 * c * POTSYNTH_NB;
          const double * bb = aa + POTSYNTH_NB;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              double v = aa[j] * cm[j] + bb[j] * sm[j];

              if (m > 0)
                {
                  const double mant = sP ? pm[j] * s[j] : pm[j];
                  const int expo = pe[j] + scale;

                  v = (expo == 0) ? v * mant : ldexp(v * mant, expo);
                }

              res[c * POTSYNTH_NB + j] += v;
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

          out[c * ldo + j] = res[c * POTSYNTH_NB + j] * f;
        }
    }
}

/*
direct_order_generic()
  Sums over l for one order m (as direct_order_NCH()), any number of
channels, accumulators in memory

Inputs: nchan - number of channels
        Wm    - weight rows of order m
        nn    - lmax - m + 1
        ab    - recurrence factors of column m
        cl    - c_m
        isE   - 1 if channel uses column E
        x     - cos(theta), length POTSYNTH_NB
        t     - s (m = 0) or s^2, length POTSYNTH_NB
        u     - a/r, length POTSYNTH_NB
        um    - u^m, length POTSYNTH_NB
        q0    - Q(m,m)
        e0    - E(m,m), length POTSYNTH_NB
        acc   - (output) sums, length 2*nchan*POTSYNTH_NB
*/

static void
direct_order_generic(const size_t nchan, const double * Wm, const size_t nn,
                     const double * ab, const double cl, const int * isE,
                     const double * x, const double * t, const double * u,
                     const double * um, const double q0, const double * e0,
                     double * restrict acc)
{
  double q1[POTSYNTH_NB], q2[POTSYNTH_NB], e1[POTSYNTH_NB], e2[POTSYNTH_NB], un[POTSYNTH_NB];
  size_t j, k, c;

  memset(acc, 0, 2 * nchan * POTSYNTH_NB * sizeof(double));

  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      q1[j] = q0;
      e1[j] = e0[j];
      q2[j] = 0.0;
      e2[j] = 0.0;
      un[j] = um[j];
    }

  for (k = 0; k < nn; ++k)
    {
      double qv[POTSYNTH_NB], ev[POTSYNTH_NB], rq[POTSYNTH_NB], re[POTSYNTH_NB];

      if (k == 0)
        {
          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              qv[j] = q1[j];
              ev[j] = e1[j];
            }
        }
      else
        {
          const double a = (k == 1) ? cl : ab[2 * k];
          const double b = (k == 1) ? 0.0 : ab[2 * k + 1];

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              qv[j] = a * x[j] * q1[j] + b * q2[j];
              ev[j] = a * (x[j] * e1[j] - t[j] * q1[j]) + b * e2[j];
              q2[j] = q1[j];
              e2[j] = e1[j];
              q1[j] = qv[j];
              e1[j] = ev[j];
            }
        }

      for (j = 0; j < POTSYNTH_NB; ++j)
        {
          rq[j] = un[j] * qv[j];
          re[j] = un[j] * ev[j];
          un[j] *= u[j];
        }

      for (c = 0; c < nchan; ++c)
        {
          const double wa = Wm[2 * c * nn + k];
          const double wb = Wm[(2 * c + 1) * nn + k];
          const double * restrict v = isE[c] ? re : rq;
          double * restrict aa = acc + 2 * c * POTSYNTH_NB;
          double * restrict bb = aa + POTSYNTH_NB;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              aa[j] += wa * v[j];
              bb[j] += wb * v[j];
            }
        }
    }
}
