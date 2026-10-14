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
 * Each channel is a sum of terms. A term is the sum over l of one scaled column
 * (Q, E or G = -dQ/dx) with a weight row, multiplied after the sum by a power
 * of sin(theta) (and possibly x = cos(theta)). P, P/sin and dP/dtheta are one
 * term each; the second-derivative operators (see the documentation, "The
 * gradient tensor") are
 *
 * S2A = 2m(m-1) s^(m-2) Q + s^m [ (2m - l(l+1)) Q - 2 x G ]   (3 terms)
 * S2B = (m-1) x s^(m-2) Q + s^m G                           (2 terms, m >= 1)
 *
 * where the s^(m-2) terms vanish for m < 2, so no division by sin(theta)
 * occurs at any order.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"

#ifndef POTSYNTH_HAVE_VECEXT
static void direct_order_generic(const size_t nterm, const double * Wm, const size_t nn,
                                 const double * ab, const double cl, const int * col,
                                 const double * x, const double * t, const double * u,
                                 const double * um, const double q0, const double * e0,
                                 double * acc);
#endif

/*
potsynth_direct_init()
  Expand the channels into terms and precompute the weight rows of the direct
method for each order

Inputs: w - workspace

Return: success or error

Notes:
1) For order m and term k, rows A (cos m phi) and B (sin m phi) of length
lmax-m+1 are stored at dW + doff[m] + 2k*(lmax-m+1); entries below the first
contributing degree are 0
*/

int
potsynth_direct_init(potsynth_workspace * w)
{
  const size_t L = w->lmax;
  const size_t L1 = L + 1;
  const size_t nchan = w->nchan;
  size_t m, l, c, k, total = 0;

  w->nterm = 0;
  for (c = 0; c < nchan; ++c)
    w->nterm += potsynth_op_nterm(w->theta_op[c]);

  w->doff = malloc(L1 * sizeof(size_t));
  w->tcol = malloc(w->nterm * sizeof(int));
  w->tpost = malloc(w->nterm * sizeof(int));
  w->tchan = malloc(w->nterm * sizeof(size_t));
  if (w->doff == NULL || w->tcol == NULL || w->tpost == NULL || w->tchan == NULL)
    {
      POTSYNTH_ERROR ("unable to allocate direct method arrays", POTSYNTH_ENOMEM);
    }

  w->needG = 0;
  for (c = 0, k = 0; c < nchan; ++c)
    {
      size_t sub;

      for (sub = 0; sub < potsynth_op_nterm(w->theta_op[c]); ++sub, ++k)
        {
          potsynth_op_term(w->theta_op[c], sub, &(w->tcol[k]), &(w->tpost[k]));
          w->tchan[k] = c;
          if (w->tcol[k] == POTSYNTH_COL_G)
            w->needG = 1;
        }
    }

  for (m = 0; m <= L; ++m)
    {
      w->doff[m] = total;
      if (m <= w->mmax)
        total += 2 * w->nterm * (L - m + 1);
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

      for (c = 0, k = 0; c < nchan; ++c)
        {
          const int op = w->theta_op[c];
          const double * rfac = w->rfac + c * L1;
          size_t sub;

          for (sub = 0; sub < potsynth_op_nterm(op); ++sub, ++k)
            {
              double * wa = w->dW + w->doff[m] + 2 * k * nn;
              double * wb = wa + nn;
              size_t lstart = (w->lmin > m) ? w->lmin : m;

              /* P/sin and S2B are not defined for m = 0 */
              if ((op == POTSYNTH_THETA_PSIN || op == POTSYNTH_THETA_S2B) && m == 0)
                lstart = L + 1;

              for (l = lstart; l <= L; ++l)
                {
                  const size_t idx = potsynth_alf_idx(l, m, L);
                  const double f = rfac[l] * potsynth_op_factor(op, sub, l, m);

                  if (w->dphi[c] == 0)
                    {
                      wa[l - m] = f * w->C[idx];
                      wb[l - m] = f * w->S[idx];
                    }
                  else
                    {
                      wa[l - m] = f * dm * w->S[idx];
                      wb[l - m] = -f * dm * w->C[idx];
                    }
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

  if (w->tcol)
    free(w->tcol);

  if (w->tpost)
    free(w->tpost);

  if (w->tchan)
    free(w->tchan);

  if (w->dW)
    potsynth_aligned_free(w->dW);
}

/* scratch space (doubles) per thread for potsynth_direct_block() */
size_t
potsynth_direct_scratch_size(const potsynth_workspace * w)
{
  return (2 * w->nterm + w->nchan) * POTSYNTH_NB;
}

#ifdef POTSYNTH_HAVE_VECEXT

/* 4 doubles; aligned(8) permits unaligned access */
typedef double potsynth_d4 __attribute__((vector_size(32), aligned(8)));

/*
 * direct_order_NCH(), direct_order_gNCH()
 *   Sums over l for one order m, 4 points and NCH terms, with all
 * accumulators in vector registers (about 3x faster than the generic version):
 *
 * acc[(2k)*NB + j]   = sum_l WA_k[l-m] u_j^l T_k(l,m)(x_j)
 * acc[(2k+1)*NB + j] = sum_l WB_k[l-m] u_j^l T_k(l,m)(x_j)
 *
 * where T_k is the scaled column Q, E or G selected by col[k]. The g variants
 * also run the recursion for G = -dQ/dx; the others never touch it.
 */
#define POTSYNTH_DIRECT_ORDER(NCH, WITHG, NAME)                                      \
static void                                                                          \
NAME(const double * Wm, const size_t nn, const double * ab, const double cl,         \
     const int * col, potsynth_d4 x, potsynth_d4 t, potsynth_d4 u, potsynth_d4 un,   \
     potsynth_d4 q1, potsynth_d4 e1, double * acc)                                   \
{                                                                                    \
  const potsynth_d4 zero = { 0.0, 0.0, 0.0, 0.0 };                                   \
  potsynth_d4 A[NCH], B[NCH], q2 = zero, e2 = zero, g1 = zero, g2 = zero;            \
  potsynth_d4 qv, ev, gv = zero, rq, re, rg = zero;                                  \
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
          gv = g1;                                                                   \
        }                                                                            \
      else                                                                           \
        {                                                                            \
          const double a = (k == 1) ? cl : ab[2 * k];                                \
          const double b = (k == 1) ? 0.0 : ab[2 * k + 1];                           \
                                                                                     \
          qv = a * x * q1 + b * q2;                                                  \
          ev = a * (x * e1 - t * q1) + b * e2;                                       \
          if (WITHG)                                                                 \
            {                                                                        \
              gv = a * (x * g1 - q1) + b * g2;                                       \
              g2 = g1;                                                               \
              g1 = gv;                                                               \
            }                                                                        \
          q2 = q1;                                                                   \
          e2 = e1;                                                                   \
          q1 = qv;                                                                   \
          e1 = ev;                                                                   \
        }                                                                            \
                                                                                     \
      rq = un * qv;                                                                  \
      re = un * ev;                                                                  \
      if (WITHG)                                                                     \
        rg = un * gv;                                                                \
      un *= u;                                                                       \
                                                                                     \
      for (c = 0; c < NCH; ++c)                                                      \
        {                                                                            \
          const potsynth_d4 v = (col[c] == POTSYNTH_COL_Q) ? rq :                    \
                                (col[c] == POTSYNTH_COL_E) ? re : rg;                \
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

POTSYNTH_DIRECT_ORDER(1, 0, direct_order_1)
POTSYNTH_DIRECT_ORDER(2, 0, direct_order_2)
POTSYNTH_DIRECT_ORDER(3, 0, direct_order_3)
POTSYNTH_DIRECT_ORDER(4, 0, direct_order_4)
POTSYNTH_DIRECT_ORDER(5, 0, direct_order_5)
POTSYNTH_DIRECT_ORDER(6, 0, direct_order_6)
POTSYNTH_DIRECT_ORDER(1, 1, direct_order_g1)
POTSYNTH_DIRECT_ORDER(2, 1, direct_order_g2)
POTSYNTH_DIRECT_ORDER(3, 1, direct_order_g3)
POTSYNTH_DIRECT_ORDER(4, 1, direct_order_g4)
POTSYNTH_DIRECT_ORDER(5, 1, direct_order_g5)
POTSYNTH_DIRECT_ORDER(6, 1, direct_order_g6)

typedef void (*direct_order_fn)(const double *, const size_t, const double *, const double,
                                const int *, potsynth_d4, potsynth_d4, potsynth_d4, potsynth_d4,
                                potsynth_d4, potsynth_d4, double *);

#define POTSYNTH_DIRECT_NVEC         6 /* terms per kernel call; more terms are done in groups */

static const direct_order_fn direct_order_vec[2][POTSYNTH_DIRECT_NVEC + 1] =
{
  { NULL, direct_order_1, direct_order_2, direct_order_3, direct_order_4, direct_order_5,
    direct_order_6 },
  { NULL, direct_order_g1, direct_order_g2, direct_order_g3, direct_order_g4, direct_order_g5,
    direct_order_g6 }
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
  const size_t nterm = w->nterm;
  const potsynth_alf_workspace * alf = w->alf_workspace_p;
  double * restrict acc = scratch;                       /* 2*nterm*NB */
  double * restrict res = acc + 2 * nterm * POTSYNTH_NB; /* nchan*NB */
  double x[POTSYNTH_NB], s[POTSYNTH_NB], s2[POTSYNTH_NB], u[POTSYNTH_NB], um[POTSYNTH_NB];
  double ph[POTSYNTH_NB], e0[POTSYNTH_NB];
  double c1[POTSYNTH_NB], s1[POTSYNTH_NB], cm[POTSYNTH_NB], sm[POTSYNTH_NB];
  double pm[POTSYNTH_NB], pmp[POTSYNTH_NB];
  int pe[POTSYNTH_NB], pep[POTSYNTH_NB];
  size_t j, m, c, k;

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
      pmp[j] = 1.0;
      pep[j] = 0;
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

              /* s^(m-1), keeping s^(m-2) */
              if (m > 1)
                {
                  pmp[j] = pm[j];
                  pep[j] = pe[j];

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
      {
        size_t h, q, k0;

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

            /* groups of up to NVEC terms, each with its own pass of the recursion */
            for (k0 = 0; k0 < nterm; k0 += POTSYNTH_DIRECT_NVEC)
              {
                const size_t nk = (nterm - k0 < POTSYNTH_DIRECT_NVEC) ? nterm - k0 : POTSYNTH_DIRECT_NVEC;
                int needG = 0;

                for (q = 0; q < nk; ++q)
                  needG |= (w->tcol[k0 + q] == POTSYNTH_COL_G);

                (direct_order_vec[needG][nk])(Wm + 2 * k0 * nn, nn, ab, alf->cl[m], w->tcol + k0,
                                              xv, tv, uv, unv, qv, ev,
                                              acc + 2 * k0 * POTSYNTH_NB + h);
              }
          }
      }
#else
      direct_order_generic(nterm, Wm, nn, ab, alf->cl[m], w->tcol, x, t, u, um, q0, e0, acc);
#endif

      for (k = 0; k < nterm; ++k)
        {
          const int post = w->tpost[k];
          const double * aa = acc + 2 * k * POTSYNTH_NB;
          const double * bb = aa + POTSYNTH_NB;
          double * rr = res + w->tchan[k] * POTSYNTH_NB;

          /* s^(m-2) terms have zero weights for m < 2 */
          if (m < 2 && (post == POTSYNTH_POST_SM2 || post == POTSYNTH_POST_XSM2))
            continue;

          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              double v = aa[j] * cm[j] + bb[j] * sm[j];

              if (m == 0)
                {
                  if (post == POTSYNTH_POST_XSM)
                    v *= x[j];
                }
              else
                {
                  double mant;
                  int expo;

                  switch (post)
                    {
                      case POTSYNTH_POST_SM:
                        mant = pm[j] * s[j];
                        expo = pe[j];
                        break;

                      case POTSYNTH_POST_SM1:
                        mant = pm[j];
                        expo = pe[j];
                        break;

                      case POTSYNTH_POST_XSM:
                        mant = pm[j] * s[j] * x[j];
                        expo = pe[j];
                        break;

                      case POTSYNTH_POST_SM2:
                        mant = pmp[j];
                        expo = pep[j];
                        break;

                      default: /* POTSYNTH_POST_XSM2 */
                        mant = pmp[j] * x[j];
                        expo = pep[j];
                        break;
                    }

                  expo += scale;
                  v = (expo == 0) ? v * mant : ldexp(v * mant, expo);
                }

              rr[j] += v;
            }
        }
    }

  for (c = 0; c < nchan; ++c)
    {
      const int rp = w->rpow[c];

      for (j = 0; j < nb; ++j)
        {
          double f = 1.0;
          int i;

          for (i = 0; i < rp; ++i)
            f *= u[j];
          for (i = 0; i > rp; --i)
            f /= u[j];

          out[c * ldo + j] = res[c * POTSYNTH_NB + j] * f;
        }
    }
}

#ifndef POTSYNTH_HAVE_VECEXT

/*
direct_order_generic()
  Sums over l for one order m (as direct_order_NCH()), any number of
terms, accumulators in memory

Inputs: nterm - number of terms
        Wm    - weight rows of order m
        nn    - lmax - m + 1
        ab    - recurrence factors of column m
        cl    - c_m
        col   - column of each term (POTSYNTH_COL_xxx)
        x     - cos(theta), length POTSYNTH_NB
        t     - s (m = 0) or s^2, length POTSYNTH_NB
        u     - a/r, length POTSYNTH_NB
        um    - u^m, length POTSYNTH_NB
        q0    - Q(m,m)
        e0    - E(m,m), length POTSYNTH_NB
        acc   - (output) sums, length 2*nterm*POTSYNTH_NB
*/

static void
direct_order_generic(const size_t nterm, const double * Wm, const size_t nn,
                     const double * ab, const double cl, const int * col,
                     const double * x, const double * t, const double * u,
                     const double * um, const double q0, const double * e0,
                     double * restrict acc)
{
  double q1[POTSYNTH_NB], q2[POTSYNTH_NB], e1[POTSYNTH_NB], e2[POTSYNTH_NB];
  double g1[POTSYNTH_NB], g2[POTSYNTH_NB], un[POTSYNTH_NB];
  size_t j, k, c;

  memset(acc, 0, 2 * nterm * POTSYNTH_NB * sizeof(double));

  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      q1[j] = q0;
      e1[j] = e0[j];
      g1[j] = 0.0;
      q2[j] = 0.0;
      e2[j] = 0.0;
      g2[j] = 0.0;
      un[j] = um[j];
    }

  for (k = 0; k < nn; ++k)
    {
      double qv[POTSYNTH_NB], ev[POTSYNTH_NB], gv[POTSYNTH_NB];
      double rq[POTSYNTH_NB], re[POTSYNTH_NB], rg[POTSYNTH_NB];

      if (k == 0)
        {
          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              qv[j] = q1[j];
              ev[j] = e1[j];
              gv[j] = g1[j];
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
              gv[j] = a * (x[j] * g1[j] - q1[j]) + b * g2[j];
              q2[j] = q1[j];
              e2[j] = e1[j];
              g2[j] = g1[j];
              q1[j] = qv[j];
              e1[j] = ev[j];
              g1[j] = gv[j];
            }
        }

      for (j = 0; j < POTSYNTH_NB; ++j)
        {
          rq[j] = un[j] * qv[j];
          re[j] = un[j] * ev[j];
          rg[j] = un[j] * gv[j];
          un[j] *= u[j];
        }

      for (c = 0; c < nterm; ++c)
        {
          const double wa = Wm[2 * c * nn + k];
          const double wb = Wm[(2 * c + 1) * nn + k];
          const double * restrict v = (col[c] == POTSYNTH_COL_Q) ? rq :
                                      (col[c] == POTSYNTH_COL_E) ? re : rg;
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

#endif /* !POTSYNTH_HAVE_VECEXT */
