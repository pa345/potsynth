/*
 * setup.c -- model set-up: coefficients, method selection, and construction of
 * the fast-transform grid (Steps 1-4 of the method)
 *
 * Grid construction, for each order m (in parallel):
 *   1. Column recursion in n at the coarse colatitude nodes of one quadrant,
 *      theta_j in (0, pi/2), fused with the sums over n of every channel, split
 *      by the parity of n - m (ps_alf_accum). The other three quadrants of the
 *      doubled sphere follow from parity and sign rules, so the recursion runs on
 *      a quarter of the nodes.
 *   2. Exact Fourier coefficients in theta by one FFT of length Nt per channel.
 *   3. Pre-sharpening (division by the window transform), zero padding, and an
 *      inverse FFT to the fine theta grid.
 *   4. The values of the stored rows are written into the grid in half-complex
 *      order along phi.
 * Then, for each stored row (in parallel), an inverse real FFT along phi turns the
 * half-complex spectra into grid values, and periodic padding columns are filled.
 * Peak memory is the grid itself plus small per-thread buffers.
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"
#include "pocketfft.h"

/* degree from which AUTO selects the fast method */
#define PS_AUTO_LMIN 24

static long
floor_mod(long a, long n)
{
  long r = a % n;
  return (r < 0) ? r + n : r;
}

/* rms size of the theta operator applied to a degree-n function (radial weights) */
static double
op_norm(int op, size_t n)
{
  if (op == POTSYNTH_THETA_P)
    return 1.0;
  return sqrt(0.5 * (double) n * ((double) n + 1.0));
}

/* ------------------------------------------------------------------------- */
/* fast method                                                               */
/* ------------------------------------------------------------------------- */

typedef struct
{
  double *W;            /* 2*npair x L1 weights */
  size_t *n0;           /* first nloc per pair */
  unsigned char *src;   /* 0 = Q, 1 = E */
  double *Q, *E;        /* column values, L1 x NB */
  double *acc;          /* 4*npair x NB */
  double *S;            /* 4*npair x nqp quadrant sums */
  double *H;            /* Nt complex */
  double *Z;            /* npair x nf1 complex */
  double *R;            /* nf2 reals: row transform buffer */
} setup_thread;

typedef struct
{
  size_t Nt, nq, nqp;    /* coarse theta nodes: total, per quadrant, padded */
  double *xq, *sq;       /* quadrant nodes */
  double *c1, *c2;       /* pre-sharpening factors */
  double *shift;         /* e^{-i p pi/Nt}/Nt, p = -L..L (complex) */
  cfft_plan pt, p1;
  rfft_plan p2;
  setup_thread *th;
  int nth;
} setup_data;

static void
setup_data_free(setup_data *d)
{
  int i;

  free(d->xq);
  free(d->sq);
  free(d->c1);
  free(d->c2);
  free(d->shift);
  if (d->pt)
    destroy_cfft_plan(d->pt);
  if (d->p1)
    destroy_cfft_plan(d->p1);
  if (d->p2)
    destroy_rfft_plan(d->p2);

  if (d->th)
    {
      for (i = 0; i < d->nth; ++i)
        {
          ps_free(d->th[i].W);
          free(d->th[i].n0);
          free(d->th[i].src);
          ps_free(d->th[i].Q);
          ps_free(d->th[i].E);
          ps_free(d->th[i].acc);
          ps_free(d->th[i].S);
          ps_free(d->th[i].H);
          ps_free(d->th[i].Z);
          ps_free(d->th[i].R);
        }
      free(d->th);
    }

  memset(d, 0, sizeof(setup_data));
}

/* one order m: fill the half-complex slots of all stored rows */
static int
setup_order(potsynth_workspace *w, const setup_data *d, setup_thread *t, size_t m)
{
  const size_t L = w->lmax, L1 = L + 1;
  const size_t nn = L - m + 1;
  const size_t rho = w->rad.rho;
  const size_t npair = w->nch;
  const size_t Nt = d->Nt, nq = d->nq, nqp = d->nqp, nf1 = w->nf1;
  const int scale = w->alf.scale[m];
  const double dm = (double) m;
  size_t k, b, j, n;
  long r;

  /* weights: pair k = (c, i), rows A (cos m phi) and B (sin m phi) */
  for (k = 0; k < npair; ++k)
    {
      const size_t c = k / rho, i = k % rho;
      const double *beta = w->rad.beta + (c * rho + i) * L1;
      const double *rfac = w->rfac + c * L1;
      double *wa = t->W + 2 * k * L1, *wb = wa + L1;
      size_t nstart = (w->lmin > m) ? w->lmin : m;

      t->src[k] = (w->theta_op[c] == POTSYNTH_THETA_DP);
      if (w->theta_op[c] == POTSYNTH_THETA_PSIN && m == 0)
        nstart = L + 1;                       /* P/sin is not defined for m = 0 */
      t->n0[k] = nstart - m;

      for (n = m; n <= L; ++n)
        {
          const size_t idx = ps_alf_idx(n, m, L);
          const double f = (n >= nstart) ? beta[n] * rfac[n] : 0.0;
          double A, B;

          if (w->dphi[c] == 0)
            {
              A = w->C[idx];
              B = w->S[idx];
            }
          else
            {
              A = dm * w->S[idx];
              B = -dm * w->C[idx];
            }

          wa[n - m] = f * A;
          wb[n - m] = f * B;
        }
    }

  /* recursion + accumulation over node blocks of the first quadrant */
  for (b = 0; b < nqp; b += PS_NB)
    {
      ps_alf_column(&w->alf, m, d->xq + b, d->sq + b, t->Q, t->E);
      ps_alf_accum(nn, npair, t->W, L1, t->src, t->n0, t->Q, t->E, t->acc);

      for (k = 0; k < 4 * npair; ++k)
        for (j = 0; j < PS_NB; ++j)
          t->S[k * nqp + b + j] = t->acc[k * PS_NB + j];
    }

  /* per pair: unscale, build the doubled-sphere function, FFT, pre-sharpen, IFFT */
  for (k = 0; k < npair; ++k)
    {
      const size_t c = k / rho;
      const int op = w->theta_op[c];
      const int typeE = (op == POTSYNTH_THETA_DP);
      /* power of sin(theta) restoring the unscaled function, and its parity */
      const size_t ps = (m == 0) ? 0 : ((op == POTSYNTH_THETA_P) ? m : m - 1);
      const double sgn = (op == POTSYNTH_THETA_P) ? ((m & 1) ? -1.0 : 1.0)
                                                  : ((m & 1) ? 1.0 : -1.0);
      const double *AE = t->S + (4 * k + 0) * nqp, *AO = t->S + (4 * k + 1) * nqp;
      const double *BE = t->S + (4 * k + 2) * nqp, *BO = t->S + (4 * k + 3) * nqp;
      const double pfac = (m == 0) ? 1.0 : 0.5;
      double *H = t->H, *Z = t->Z + 2 * k * nf1;
      long p;

      for (j = 0; j < nq; ++j)
        {
          double ae = AE[j], ao = AO[j], be = BE[j], bo = BO[j];
          double vA, vB, uA, uB;

          if (m > 0)
            {
              double mant;
              int expo;

              ps_pow_scaled(d->sq[j], ps, &mant, &expo);
              ae = ldexp(ae * mant, expo + scale);
              ao = ldexp(ao * mant, expo + scale);
              be = ldexp(be * mant, expo + scale);
              bo = ldexp(bo * mant, expo + scale);
            }

          /* theta_j and pi - theta_j */
          vA = ae + ao;
          vB = be + bo;
          uA = typeE ? (ao - ae) : (ae - ao);
          uB = typeE ? (bo - be) : (be - bo);

          /* H = vA - i vB */
          H[2 * j] = vA;
          H[2 * j + 1] = -vB;
          H[2 * (Nt / 2 - 1 - j)] = uA;
          H[2 * (Nt / 2 - 1 - j) + 1] = -uB;

          /* 2 pi - theta_j and pi + theta_j: sin(theta) changes sign */
          H[2 * (Nt - 1 - j)] = sgn * vA;
          H[2 * (Nt - 1 - j) + 1] = -sgn * vB;
          H[2 * (Nt / 2 + j)] = sgn * uA;
          H[2 * (Nt / 2 + j) + 1] = -sgn * uB;
        }

      if (cfft_forward(d->pt, H, 1.0) != 0)
        return POTSYNTH_ENOMEM;

      memset(Z, 0, 2 * nf1 * sizeof(double));
      for (p = -(long) L; p <= (long) L; ++p)
        {
          const size_t ip = (size_t) floor_mod(p, (long) Nt);
          const size_t iz = (size_t) floor_mod(p, (long) nf1);
          const double *sh = d->shift + 2 * (size_t) (p + (long) L);
          const double f = d->c1[labs(p)] * d->c2[m] * pfac;
          const double hr = H[2 * ip], hi = H[2 * ip + 1];

          Z[2 * iz] = f * (sh[0] * hr - sh[1] * hi);
          Z[2 * iz + 1] = f * (sh[0] * hi + sh[1] * hr);
        }

      if (cfft_backward(d->p1, Z, 1.0) != 0)
        return POTSYNTH_ENOMEM;
    }

  /* write the half-complex slots: m = 0 -> column 0 (real part),
   * m >= 1 -> columns 2m-1 (real) and 2m (imaginary) */
  for (r = w->row_lo; r <= w->row_hi; ++r)
    {
      const size_t l = (size_t) floor_mod(r, (long) nf1);
      double *row = w->grid + (size_t) (r - w->row_lo) * w->ncols * w->nchp;

      if (m == 0)
        {
          double *g = row + (size_t) (0 - w->col_lo) * w->nchp;

          for (k = 0; k < npair; ++k)
            g[k] = t->Z[2 * k * nf1 + 2 * l];
        }
      else
        {
          double *gr = row + (size_t) ((long) (2 * m - 1) - w->col_lo) * w->nchp;
          double *gi = gr + w->nchp;

          for (k = 0; k < npair; ++k)
            {
              gr[k] = t->Z[2 * k * nf1 + 2 * l];
              gi[k] = t->Z[2 * k * nf1 + 2 * l + 1];
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

int
ps_fast_setup(potsynth_workspace *w)
{
  const size_t L = w->lmax, L1 = L + 1;
  const size_t nband = 2 * L + 1;
  setup_data d;
  double *wn = NULL;
  double tol;
  size_t c, n, m, j, nrows, ncols;
  int status = POTSYNTH_SUCCESS;
  int i;

  memset(&d, 0, sizeof(setup_data));

  /* window */
  w->sigma = (w->params.sigma > 0.0) ? w->params.sigma : 2.0;
  w->w = (w->params.w > 0) ? w->params.w : ps_window_choose(w->params.eps, w->sigma);
  if (w->w < 2 || w->w > 32)
    return POTSYNTH_EINVAL;
  w->beta = M_PI * (1.0 - 1.0 / (2.0 * w->sigma)) * w->w;
  w->half = 0.5 * w->w;

  /* step 1: radial basis, weighted by the rms size of each degree in each channel */
  wn = calloc(w->nchan * L1, sizeof(double));
  if (!wn)
    return POTSYNTH_ENOMEM;

  for (n = w->lmin; n <= L; ++n)
    {
      double sum = 0.0;
      size_t M = (n < w->mmax) ? n : w->mmax;

      for (m = 0; m <= M; ++m)
        {
          const size_t idx = ps_alf_idx(n, m, L);
          sum += w->C[idx] * w->C[idx] + w->S[idx] * w->S[idx];
        }

      for (c = 0; c < w->nchan; ++c)
        wn[c * L1 + n] = fabs(w->rfac[c * L1 + n]) * op_norm(w->theta_op[c], n) * sqrt(sum);
    }

  tol = (w->params.rank_tol > 0.0) ? w->params.rank_tol : 0.1 * w->params.eps;
  if (tol < 1e-16)
    tol = 1e-16;

  status = ps_radial_init(&w->rad, w->a / w->rmax, w->a / w->rmin, w->lmin, L,
                          w->nchan, w->rpow, wn, tol);
  free(wn);
  if (status)
    return status;

  /* grid geometry */
  w->nf1 = ps_next_smooth((size_t) ceil(w->sigma * nband) > 2 * (size_t) w->w ?
                          (size_t) ceil(w->sigma * nband) : 2 * (size_t) w->w, 2);
  w->nf2 = w->nf1;
  w->h1 = 2.0 * M_PI / w->nf1;
  w->h2 = 2.0 * M_PI / w->nf2;
  w->inv_h1 = w->nf1 / (2.0 * M_PI);
  w->inv_h2 = w->nf2 / (2.0 * M_PI);

  /* rows that windows of points with theta in [0, pi] can touch, plus one */
  w->row_lo = (long) ceil(0.0 * w->inv_h1 - w->half) - 1;
  w->row_hi = (long) ceil(M_PI * w->inv_h1 - w->half) + w->w;
  /* columns for phi in [0, 2pi), with periodic padding so windows never wrap */
  w->col_lo = (long) ceil(-w->half) - 1;
  w->col_hi = (long) w->nf2 + w->w;

  nrows = (size_t) (w->row_hi - w->row_lo + 1);
  ncols = (size_t) (w->col_hi - w->col_lo + 1);
  w->nrows = nrows;
  w->ncols = ncols;
  w->nch = w->nchan * w->rad.rho;
  w->nchp = (w->nch + PS_CHAN_ALIGN - 1) / PS_CHAN_ALIGN * PS_CHAN_ALIGN;

  if ((double) nrows * ncols * w->nchp * sizeof(double) > (double) SIZE_MAX / 2)
    return POTSYNTH_ENOMEM;

  w->grid = ps_calloc(nrows * ncols * w->nchp * sizeof(double));
  if (!w->grid)
    return POTSYNTH_ENOMEM;

  /* coarse theta nodes on the doubled sphere: theta_j = (j + 1/2) 2 pi / Nt, Nt a
   * multiple of 4, >= 2L + 2 (exact for degree L); quadrant nodes j < Nt/4 */
  d.Nt = ps_next_smooth(2 * L + 2, 4);
  d.nq = d.Nt / 4;
  d.nqp = (d.nq + PS_NB - 1) / PS_NB * PS_NB;
  d.xq = malloc(d.nqp * sizeof(double));
  d.sq = malloc(d.nqp * sizeof(double));
  d.c1 = malloc(L1 * sizeof(double));
  d.c2 = malloc(L1 * sizeof(double));
  d.shift = malloc(2 * nband * sizeof(double));
  d.pt = make_cfft_plan(d.Nt);
  d.p1 = make_cfft_plan(w->nf1);
  d.p2 = make_rfft_plan(w->nf2);
  if (!d.xq || !d.sq || !d.c1 || !d.c2 || !d.shift || !d.pt || !d.p1 || !d.p2)
    {
      setup_data_free(&d);
      return POTSYNTH_ENOMEM;
    }

  for (j = 0; j < d.nqp; ++j)
    {
      const size_t jj = (j < d.nq) ? j : d.nq - 1;   /* padding repeats the last node */
      const double th = (jj + 0.5) * 2.0 * M_PI / d.Nt;
      d.xq[j] = cos(th);
      d.sq[j] = sin(th);
    }

  for (j = 0; j < nband; ++j)
    {
      const double p = (double) j - (double) L;
      d.shift[2 * j] = cos(p * M_PI / d.Nt) / d.Nt;
      d.shift[2 * j + 1] = -sin(p * M_PI / d.Nt) / d.Nt;
    }

  /* pre-sharpening factors h / phi_hat(k) */
  status = ps_window_ft(L1, w->w, w->beta, w->h1, d.c1);
  if (!status)
    status = ps_window_ft(L1, w->w, w->beta, w->h2, d.c2);
  if (status)
    {
      setup_data_free(&d);
      return status;
    }
  for (j = 0; j < L1; ++j)
    {
      d.c1[j] = w->h1 / d.c1[j];
      d.c2[j] = w->h2 / d.c2[j];
    }

  /* per-thread buffers */
  d.nth = w->nthreads;
  d.th = calloc((size_t) d.nth, sizeof(setup_thread));
  if (!d.th)
    {
      setup_data_free(&d);
      return POTSYNTH_ENOMEM;
    }

  for (i = 0; i < d.nth; ++i)
    {
      setup_thread *t = &d.th[i];

      t->W = ps_malloc(2 * w->nch * L1 * sizeof(double));
      t->n0 = malloc(w->nch * sizeof(size_t));
      t->src = malloc(w->nch);
      t->Q = ps_malloc(L1 * PS_NB * sizeof(double));
      t->E = ps_malloc(L1 * PS_NB * sizeof(double));
      t->acc = ps_malloc(4 * w->nch * PS_NB * sizeof(double));
      t->S = ps_malloc(4 * w->nch * d.nqp * sizeof(double));
      t->H = ps_malloc(2 * d.Nt * sizeof(double));
      t->Z = ps_malloc(2 * w->nch * w->nf1 * sizeof(double));
      t->R = ps_malloc(w->nf2 * sizeof(double));
      if (!t->W || !t->n0 || !t->src || !t->Q || !t->E || !t->acc || !t->S || !t->H || !t->Z ||
          !t->R)
        {
          setup_data_free(&d);
          return POTSYNTH_ENOMEM;
        }
    }

  /* phase 1: orders m (work decreases with m; dynamic schedule) */
  {
    const long mmax = (long) w->mmax;
    long mm;
    int err = POTSYNTH_SUCCESS;

#pragma omp parallel for schedule(dynamic, 1) num_threads(d.nth)
    for (mm = 0; mm <= mmax; ++mm)
      {
#ifdef _OPENMP
        setup_thread *t = &d.th[omp_get_thread_num()];
#else
        setup_thread *t = &d.th[0];
#endif
        int s = setup_order(w, &d, t, (size_t) mm);
        if (s)
          {
#pragma omp atomic write
            err = s;
          }
      }

    if (err)
      {
        setup_data_free(&d);
        return err;
      }
  }

  /* phase 2: inverse real FFT along phi for each stored row and channel,
   * then periodic padding columns */
  {
    const long nr = (long) nrows;
    const size_t nf2 = w->nf2, nchp = w->nchp, nch = w->nch;
    long rr;
    int err = POTSYNTH_SUCCESS;

#pragma omp parallel for schedule(static) num_threads(d.nth)
    for (rr = 0; rr < nr; ++rr)
      {
#ifdef _OPENMP
        double *rowfft = d.th[omp_get_thread_num()].R;
#else
        double *rowfft = d.th[0].R;
#endif
        double *row = w->grid + (size_t) rr * ncols * nchp;
        size_t q, k;

        for (q = 0; q < nch; ++q)
          {
            long col;

            for (k = 0; k < nf2; ++k)
              rowfft[k] = row[(size_t) ((long) k - w->col_lo) * nchp + q];

            if (rfft_backward(d.p2, rowfft, 1.0) != 0)
              {
#pragma omp atomic write
                err = POTSYNTH_ENOMEM;
                break;
              }

            for (col = w->col_lo; col <= w->col_hi; ++col)
              row[(size_t) (col - w->col_lo) * nchp + q] =
                rowfft[floor_mod(col, (long) nf2)];
          }
      }

    if (err)
      {
        setup_data_free(&d);
        return err;
      }
  }

  setup_data_free(&d);
  return POTSYNTH_SUCCESS;
}

/* ------------------------------------------------------------------------- */
/* allocation                                                                */
/* ------------------------------------------------------------------------- */

static int
check_args(size_t lmin, size_t lmax, size_t mmax, const double *coef, double a,
           size_t nchan, const potsynth_channel *chan, const potsynth_params *p)
{
  size_t c;

  if (lmin > lmax || mmax > lmax || !coef || !chan || nchan == 0)
    return POTSYNTH_EINVAL;
  if (lmax > POTSYNTH_LMAX)
    return POTSYNTH_EUNSUP;
  if (!(a > 0.0) || !isfinite(a))
    return POTSYNTH_EINVAL;
  if (!(p->eps > 0.0) || p->sigma < 0.0 || p->w < 0 || p->rank_tol < 0.0 || p->nthreads < 0)
    return POTSYNTH_EINVAL;
  if (p->method != POTSYNTH_METHOD_AUTO && p->method != POTSYNTH_METHOD_FAST &&
      p->method != POTSYNTH_METHOD_DIRECT)
    return POTSYNTH_EINVAL;

  for (c = 0; c < nchan; ++c)
    {
      if (chan[c].theta != POTSYNTH_THETA_P && chan[c].theta != POTSYNTH_THETA_DP &&
          chan[c].theta != POTSYNTH_THETA_PSIN)
        return POTSYNTH_EINVAL;
      if (chan[c].dphi != 0 && chan[c].dphi != 1)
        return POTSYNTH_EINVAL;
      if ((long) lmin + chan[c].rpow < 0)
        return POTSYNTH_EINVAL;
    }

  return POTSYNTH_SUCCESS;
}

potsynth_workspace *
potsynth_alloc(size_t lmin, size_t lmax, size_t mmax, const double *coef, double a,
               double rmin, double rmax, size_t nchan, const potsynth_channel *chan,
               const potsynth_params *params, int *status)
{
  const size_t L1 = lmax + 1;
  const size_t nlm = L1 * (lmax + 2) / 2;
  potsynth_params p;
  potsynth_workspace *w;
  size_t c, n, m;
  int s;

  if (params)
    p = *params;
  else
    potsynth_params_init(&p);

  s = check_args(lmin, lmax, mmax, coef, a, nchan, chan, &p);
  if (s)
    {
      if (status)
        *status = s;
      return NULL;
    }

  w = calloc(1, sizeof(potsynth_workspace));
  if (!w)
    {
      if (status)
        *status = POTSYNTH_ENOMEM;
      return NULL;
    }

  w->lmin = lmin;
  w->lmax = lmax;
  w->mmax = mmax;
  w->a = a;
  w->rmin = rmin;
  w->rmax = rmax;
  w->nchan = nchan;
  w->params = p;
  w->nthreads = ps_max_threads(p.nthreads);

  if (p.method == POTSYNTH_METHOD_AUTO)
    w->method = (lmax >= PS_AUTO_LMIN) ? POTSYNTH_METHOD_FAST : POTSYNTH_METHOD_DIRECT;
  else
    w->method = p.method;

  if (w->method == POTSYNTH_METHOD_FAST &&
      !(rmin > 0.0 && rmin <= rmax && isfinite(rmax)))
    {
      free(w);
      if (status)
        *status = POTSYNTH_EINVAL;
      return NULL;
    }

  w->rpow = malloc(nchan * sizeof(int));
  w->theta_op = malloc(nchan * sizeof(int));
  w->dphi = malloc(nchan * sizeof(int));
  w->rfac = ps_calloc(nchan * L1 * sizeof(double));
  w->C = ps_calloc(nlm * sizeof(double));
  w->S = ps_calloc(nlm * sizeof(double));
  if (!w->rpow || !w->theta_op || !w->dphi || !w->rfac || !w->C || !w->S)
    {
      potsynth_free(w);
      if (status)
        *status = POTSYNTH_ENOMEM;
      return NULL;
    }

  for (c = 0; c < nchan; ++c)
    {
      w->rpow[c] = chan[c].rpow;
      w->theta_op[c] = chan[c].theta;
      w->dphi[c] = chan[c].dphi;
      for (n = 0; n <= lmax; ++n)
        w->rfac[c * L1 + n] = chan[c].rfac ? chan[c].rfac[n] : 1.0;
    }

  /* public layout (green2 convention) -> internal M-major */
  for (n = lmin; n <= lmax; ++n)
    {
      const size_t M = (n < mmax) ? n : mmax;

      for (m = 0; m <= M; ++m)
        {
          const size_t idx = ps_alf_idx(n, m, lmax);
          const long lm = (long) m;

          w->C[idx] = coef[POTSYNTH_IDX((long) n, lm, (long) lmin, (long) mmax)];
          if (m > 0)
            w->S[idx] = coef[POTSYNTH_IDX((long) n, -lm, (long) lmin, (long) mmax)];
        }
    }

  s = ps_alf_init(&w->alf, lmax);
  if (!s && w->method == POTSYNTH_METHOD_FAST)
    s = ps_fast_setup(w);
  else if (!s)
    s = ps_direct_init(w);

  if (s)
    {
      potsynth_free(w);
      if (status)
        *status = s;
      return NULL;
    }

  if (status)
    *status = POTSYNTH_SUCCESS;
  return w;
}

void
potsynth_free(potsynth_workspace *w)
{
  if (!w)
    return;

  free(w->rpow);
  free(w->theta_op);
  free(w->dphi);
  ps_free(w->rfac);
  ps_free(w->C);
  ps_free(w->S);
  ps_alf_free(&w->alf);
  ps_free(w->grid);
  ps_radial_free(&w->rad);
  ps_direct_free(w);
  free(w);
}

int
potsynth_get_info(const potsynth_workspace *w, potsynth_info *info)
{
  if (!w || !info)
    return POTSYNTH_EINVAL;

  memset(info, 0, sizeof(potsynth_info));
  info->lmin = w->lmin;
  info->lmax = w->lmax;
  info->mmax = w->mmax;
  info->nchan = w->nchan;
  info->method = w->method;

  if (w->method == POTSYNTH_METHOD_FAST)
    {
      info->w = w->w;
      info->sigma = w->sigma;
      info->rho = w->rad.rho;
      info->ncheb = w->rad.ncheb;
      info->nf1 = w->nf1;
      info->nf2 = w->nf2;
      info->nrows = w->nrows;
      info->grid_bytes = w->nrows * w->ncols * w->nchp * sizeof(double);
    }

  return POTSYNTH_SUCCESS;
}
