/*
 * fast.c
 *
 * Construction of the fast-transform grid (steps 1-4 of the fast method)
 *
 * For each order m (in parallel):
 *
 * 1) Column recursion in l at the coarse colatitude nodes of one quadrant,
 *    theta_j in (0, pi/2), fused with the sums over l of every channel, split by
 *    the parity of l - m (potsynth_alf_accum). The other three quadrants of the
 *    doubled sphere follow from parity and sign rules, so the recursion runs on a
 *    quarter of the nodes.
 *
 * 2) Exact Fourier coefficients in theta by one FFT of length Nt per channel.
 *
 * 3) Pre-sharpening (division by the window transform), zero padding, and an
 *    inverse FFT to the fine theta grid.
 *
 * 4) The values of the stored rows are written into the grid in half-complex
 *    order along phi.
 *
 * Then, for each stored row (in parallel), an inverse real FFT along phi turns
 * the half-complex spectra into grid values, and the periodic padding columns
 * are filled. Peak memory is the grid itself plus small per-thread buffers.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"
#include "pocketfft.h"

/* per-thread buffers */
typedef struct
{
  double * W;           /* weights, 2*npair rows of length lmax+1 */
  size_t * n0;          /* first index (l - m) of each pair, length npair */
  unsigned char * src;  /* source of each pair (0 = Q, 1 = E), length npair */
  double * Q;           /* column values, (lmax+1)*POTSYNTH_NB */
  double * E;           /* column values, (lmax+1)*POTSYNTH_NB */
  double * acc;         /* sums of one node block, 4*npair*POTSYNTH_NB */
  double * S;           /* quadrant sums, 4*npair*nqp */
  double * H;           /* doubled-sphere function, Nt complex */
  double * Z;           /* fine theta grid of each pair, npair*nf1 complex */
  double * R;           /* row transform buffer, nf2 */
} fast_thread;

/* data shared by all threads during set-up */
typedef struct
{
  size_t Nt;            /* coarse theta nodes on the doubled sphere */
  size_t nq;            /* nodes per quadrant, Nt/4 */
  size_t nqp;           /* nq padded to a multiple of POTSYNTH_NB */
  double * xq;          /* cos(theta_j) of quadrant nodes, length nqp */
  double * sq;          /* sin(theta_j) of quadrant nodes, length nqp */
  double * c1;          /* pre-sharpening factors in theta, length lmax+1 */
  double * c2;          /* pre-sharpening factors in phi, length lmax+1 */
  double * shift;       /* e^{-i p pi/Nt}/Nt, p = -lmax..lmax (complex) */
  cfft_plan pt;         /* FFT of length Nt */
  cfft_plan p1;         /* FFT of length nf1 */
  rfft_plan p2;         /* real FFT of length nf2 */
  int nthreads;         /* number of threads */
  fast_thread * threads;
} fast_state_t;

static int fast_state_alloc(potsynth_workspace * w, fast_state_t * state);
static void fast_state_free(fast_state_t * state);
static int fast_order(const size_t m, const fast_state_t * state, fast_thread * thread,
                      potsynth_workspace * w);
static int fast_row(const size_t row, double * R, const fast_state_t * state, potsynth_workspace * w);
static double op_norm(const int op, const size_t l);
static long floor_mod(const long a, const long n);

/*
potsynth_fast_init()
  Construct the radial basis and the grid of the fast method

Inputs: w - workspace, with coefficients and channels set

Return: success or error
*/

int
potsynth_fast_init(potsynth_workspace * w)
{
  const size_t L = w->lmax;
  const size_t L1 = L + 1;
  const size_t nband = 2 * L + 1;
  fast_state_t state;
  double * wn;
  double tol;
  size_t c, l, m;
  long mm, rr;
  int status = POTSYNTH_SUCCESS;

  /* window */
  w->sigma = (w->params.sigma > 0.0) ? w->params.sigma : 2.0;
  w->w = (w->params.w > 0) ? w->params.w : potsynth_window_choose(w->params.eps, w->sigma);
  if (w->w < 2 || w->w > 32)
    {
      POTSYNTH_ERROR ("window width must be between 2 and 32", POTSYNTH_EINVAL);
    }

  w->beta = M_PI * (1.0 - 1.0 / (2.0 * w->sigma)) * w->w;
  w->half = 0.5 * w->w;

  /* step 1: radial basis, weighted by the rms size of each degree in each channel */
  wn = calloc(w->nchan * L1, sizeof(double));
  if (wn == NULL)
    {
      POTSYNTH_ERROR ("unable to allocate radial weights", POTSYNTH_ENOMEM);
    }

  for (l = w->lmin; l <= L; ++l)
    {
      const size_t M = (l < w->mmax) ? l : w->mmax;
      double sum = 0.0;

      for (m = 0; m <= M; ++m)
        {
          const size_t idx = potsynth_alf_idx(l, m, L);
          sum += w->C[idx] * w->C[idx] + w->S[idx] * w->S[idx];
        }

      for (c = 0; c < w->nchan; ++c)
        wn[c * L1 + l] = fabs(w->rfac[c * L1 + l]) * op_norm(w->theta_op[c], l) * sqrt(sum);
    }

  tol = (w->params.rank_tol > 0.0) ? w->params.rank_tol : 0.1 * w->params.eps;
  if (tol < 1.0e-16)
    tol = 1.0e-16;

  w->radial_workspace_p = potsynth_radial_alloc(w->a / w->rmax, w->a / w->rmin, w->lmin, L,
                                                w->nchan, w->rpow, wn, tol);
  free(wn);
  if (w->radial_workspace_p == NULL)
    return POTSYNTH_EINVAL; /* reported by potsynth_radial_alloc() */

  /* grid geometry: nf >= sigma (2 lmax + 1) and >= 2 w, even and 2,3,5-smooth */
  {
    const size_t nmin = (size_t) ceil(w->sigma * nband);
    w->nf1 = potsynth_next_smooth((nmin > 2 * (size_t) w->w) ? nmin : 2 * (size_t) w->w, 2);
  }
  w->nf2 = w->nf1;
  w->h1 = 2.0 * M_PI / w->nf1;
  w->h2 = 2.0 * M_PI / w->nf2;
  w->inv_h1 = w->nf1 / (2.0 * M_PI);
  w->inv_h2 = w->nf2 / (2.0 * M_PI);

  /* rows that windows of points with theta in [0,pi] can touch, plus one */
  w->row_lo = (long) ceil(-w->half) - 1;
  w->row_hi = (long) ceil(M_PI * w->inv_h1 - w->half) + w->w;

  /* columns for phi in [0,2pi), with periodic padding so windows never wrap */
  w->col_lo = (long) ceil(-w->half) - 1;
  w->col_hi = (long) w->nf2 + w->w;

  w->nrows = (size_t) (w->row_hi - w->row_lo + 1);
  w->ncols = (size_t) (w->col_hi - w->col_lo + 1);
  w->nch = w->nchan * w->radial_workspace_p->rho;
  w->nchp = (w->nch + POTSYNTH_CHAN_ALIGN - 1) / POTSYNTH_CHAN_ALIGN * POTSYNTH_CHAN_ALIGN;

  if ((double) w->nrows * w->ncols * w->nchp * sizeof(double) > (double) SIZE_MAX / 2)
    {
      POTSYNTH_ERROR ("grid too large", POTSYNTH_ENOMEM);
    }

  w->grid = potsynth_aligned_calloc(w->nrows * w->ncols * w->nchp * sizeof(double));
  if (w->grid == NULL)
    {
      POTSYNTH_ERROR ("unable to allocate grid", POTSYNTH_ENOMEM);
    }

  status = fast_state_alloc(w, &state);
  if (status)
    {
      fast_state_free(&state);
      return status;
    }

  /* steps 1-4 for each order m (work decreases with m: dynamic schedule) */
#pragma omp parallel for private(mm) schedule(dynamic, 1) num_threads(state.nthreads)
  for (mm = 0; mm <= (long) w->mmax; ++mm)
    {
#ifdef _OPENMP
      fast_thread * thread = &(state.threads[omp_get_thread_num()]);
#else
      fast_thread * thread = &(state.threads[0]);
#endif
      int s = fast_order((size_t) mm, &state, thread, w);

      if (s)
        {
#pragma omp atomic write
          status = s;
        }
    }

  /* inverse real FFT along phi of each stored row, and periodic padding */
  if (status == POTSYNTH_SUCCESS)
    {
#pragma omp parallel for private(rr) schedule(static) num_threads(state.nthreads)
      for (rr = 0; rr < (long) w->nrows; ++rr)
        {
#ifdef _OPENMP
          double * R = state.threads[omp_get_thread_num()].R;
#else
          double * R = state.threads[0].R;
#endif
          int s = fast_row((size_t) rr, R, &state, w);

          if (s)
            {
#pragma omp atomic write
              status = s;
            }
        }
    }

  fast_state_free(&state);

  if (status)
    {
      POTSYNTH_ERROR ("FFT failed (out of memory)", status);
    }

  return POTSYNTH_SUCCESS;
}

/*
fast_state_alloc()
  Allocate and initialize the shared set-up data and per-thread buffers

Inputs: w     - workspace
        state - (output) state

Notes:
1) The coarse theta nodes on the doubled sphere are theta_j = (j + 1/2) 2 pi / Nt,
with Nt a multiple of 4 and >= 2 lmax + 2 (exact for degree lmax); the quadrant
nodes are j < Nt/4
*/

static int
fast_state_alloc(potsynth_workspace * w, fast_state_t * state)
{
  const size_t L = w->lmax;
  const size_t L1 = L + 1;
  const size_t nband = 2 * L + 1;
  size_t j;
  int i, status;

  memset(state, 0, sizeof(fast_state_t));

  state->Nt = potsynth_next_smooth(2 * L + 2, 4);
  state->nq = state->Nt / 4;
  state->nqp = (state->nq + POTSYNTH_NB - 1) / POTSYNTH_NB * POTSYNTH_NB;
  state->nthreads = w->nthreads;

  state->xq = malloc(state->nqp * sizeof(double));
  state->sq = malloc(state->nqp * sizeof(double));
  state->c1 = malloc(L1 * sizeof(double));
  state->c2 = malloc(L1 * sizeof(double));
  state->shift = malloc(2 * nband * sizeof(double));
  state->pt = make_cfft_plan(state->Nt);
  state->p1 = make_cfft_plan(w->nf1);
  state->p2 = make_rfft_plan(w->nf2);
  state->threads = calloc((size_t) state->nthreads, sizeof(fast_thread));
  if (!state->xq || !state->sq || !state->c1 || !state->c2 || !state->shift ||
      !state->pt || !state->p1 || !state->p2 || !state->threads)
    {
      POTSYNTH_ERROR ("unable to allocate set-up data", POTSYNTH_ENOMEM);
    }

  for (j = 0; j < state->nqp; ++j)
    {
      const size_t jj = (j < state->nq) ? j : state->nq - 1; /* padding repeats the last node */
      const double theta = (jj + 0.5) * 2.0 * M_PI / state->Nt;

      state->xq[j] = cos(theta);
      state->sq[j] = sin(theta);
    }

  for (j = 0; j < nband; ++j)
    {
      const double p = (double) j - (double) L;

      state->shift[2 * j] = cos(p * M_PI / state->Nt) / state->Nt;
      state->shift[2 * j + 1] = -sin(p * M_PI / state->Nt) / state->Nt;
    }

  /* pre-sharpening factors h / phi_hat(k) */
  status = potsynth_window_ft(L1, w->w, w->beta, w->h1, state->c1);
  if (status)
    return status;

  status = potsynth_window_ft(L1, w->w, w->beta, w->h2, state->c2);
  if (status)
    return status;

  for (j = 0; j < L1; ++j)
    {
      state->c1[j] = w->h1 / state->c1[j];
      state->c2[j] = w->h2 / state->c2[j];
    }

  for (i = 0; i < state->nthreads; ++i)
    {
      fast_thread * thread = &(state->threads[i]);

      thread->W = potsynth_aligned_alloc(2 * w->nch * L1 * sizeof(double));
      thread->n0 = malloc(w->nch * sizeof(size_t));
      thread->src = malloc(w->nch);
      thread->Q = potsynth_aligned_alloc(L1 * POTSYNTH_NB * sizeof(double));
      thread->E = potsynth_aligned_alloc(L1 * POTSYNTH_NB * sizeof(double));
      thread->acc = potsynth_aligned_alloc(4 * w->nch * POTSYNTH_NB * sizeof(double));
      thread->S = potsynth_aligned_alloc(4 * w->nch * state->nqp * sizeof(double));
      thread->H = potsynth_aligned_alloc(2 * state->Nt * sizeof(double));
      thread->Z = potsynth_aligned_alloc(2 * w->nch * w->nf1 * sizeof(double));
      thread->R = potsynth_aligned_alloc(w->nf2 * sizeof(double));
      if (!thread->W || !thread->n0 || !thread->src || !thread->Q || !thread->E ||
          !thread->acc || !thread->S || !thread->H || !thread->Z || !thread->R)
        {
          POTSYNTH_ERROR ("unable to allocate thread buffers", POTSYNTH_ENOMEM);
        }
    }

  return POTSYNTH_SUCCESS;
}

static void
fast_state_free(fast_state_t * state)
{
  if (state->xq)
    free(state->xq);

  if (state->sq)
    free(state->sq);

  if (state->c1)
    free(state->c1);

  if (state->c2)
    free(state->c2);

  if (state->shift)
    free(state->shift);

  if (state->pt)
    destroy_cfft_plan(state->pt);

  if (state->p1)
    destroy_cfft_plan(state->p1);

  if (state->p2)
    destroy_rfft_plan(state->p2);

  if (state->threads)
    {
      int i;

      for (i = 0; i < state->nthreads; ++i)
        {
          fast_thread * thread = &(state->threads[i]);

          if (thread->W)
            potsynth_aligned_free(thread->W);

          if (thread->n0)
            free(thread->n0);

          if (thread->src)
            free(thread->src);

          if (thread->Q)
            potsynth_aligned_free(thread->Q);

          if (thread->E)
            potsynth_aligned_free(thread->E);

          if (thread->acc)
            potsynth_aligned_free(thread->acc);

          if (thread->S)
            potsynth_aligned_free(thread->S);

          if (thread->H)
            potsynth_aligned_free(thread->H);

          if (thread->Z)
            potsynth_aligned_free(thread->Z);

          if (thread->R)
            potsynth_aligned_free(thread->R);
        }

      free(state->threads);
    }
}

/*
fast_order()
  Steps 1-4 for one order m: fill the half-complex slots of all stored rows

Inputs: m      - SH order
        state  - shared set-up data
        thread - per-thread buffers
        w      - workspace

Notes:
1) Pair k = (c, i) of channel c and radial function i has weight rows
A (cos m phi) and B (sin m phi); the angular function is
G = Re[ (S_A - i S_B) e^{i m phi} ]

2) Values on the doubled sphere from the quadrant sums E (l-m even) and
O (l-m odd), with sigma = (-1)^m for P and (-1)^(m-1) for P/sin, dP/dtheta:

theta_j       : E + O
pi - theta_j  : E - O  (P, P/sin),  O - E  (dP/dtheta)
2pi - theta_j : sigma * value at theta_j
pi + theta_j  : sigma * value at pi - theta_j
*/

static int
fast_order(const size_t m, const fast_state_t * state, fast_thread * thread, potsynth_workspace * w)
{
  const size_t L = w->lmax;
  const size_t L1 = L + 1;
  const size_t nn = L - m + 1;
  const size_t rho = w->radial_workspace_p->rho;
  const size_t npair = w->nch;
  const size_t Nt = state->Nt;
  const size_t nqp = state->nqp;
  const size_t nf1 = w->nf1;
  const int scale = w->alf_workspace_p->scale[m];
  const double dm = (double) m;
  size_t k, b, j, l;
  long r;

  /* weight rows of each pair */
  for (k = 0; k < npair; ++k)
    {
      const size_t c = k / rho;
      const size_t i = k % rho;
      const double * beta = w->radial_workspace_p->beta + (c * rho + i) * L1;
      const double * rfac = w->rfac + c * L1;
      double * wa = thread->W + 2 * k * L1;
      double * wb = wa + L1;
      size_t lstart = (w->lmin > m) ? w->lmin : m;

      thread->src[k] = (w->theta_op[c] == POTSYNTH_THETA_DP);
      if (w->theta_op[c] == POTSYNTH_THETA_PSIN && m == 0)
        lstart = L + 1; /* P/sin is not defined for m = 0 */

      thread->n0[k] = lstart - m;

      for (l = m; l <= L; ++l)
        {
          const size_t idx = potsynth_alf_idx(l, m, L);
          const double f = (l >= lstart) ? beta[l] * rfac[l] : 0.0;
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

          wa[l - m] = f * A;
          wb[l - m] = f * B;
        }
    }

  /* recursion + accumulation over node blocks of the first quadrant */
  for (b = 0; b < nqp; b += POTSYNTH_NB)
    {
      potsynth_alf_column(m, state->xq + b, state->sq + b, thread->Q, thread->E, w->alf_workspace_p);
      potsynth_alf_accum(nn, npair, thread->W, L1, thread->src, thread->n0, thread->Q, thread->E,
                         thread->acc);

      for (k = 0; k < 4 * npair; ++k)
        {
          for (j = 0; j < POTSYNTH_NB; ++j)
            thread->S[k * nqp + b + j] = thread->acc[k * POTSYNTH_NB + j];
        }
    }

  /* each pair: unscale, build the doubled-sphere function, FFT, pre-sharpen, IFFT */
  for (k = 0; k < npair; ++k)
    {
      const size_t c = k / rho;
      const int op = w->theta_op[c];
      const int typeE = (op == POTSYNTH_THETA_DP);
      const size_t ps = (m == 0) ? 0 : ((op == POTSYNTH_THETA_P) ? m : m - 1); /* power of sin(theta) */
      const double sgn = (op == POTSYNTH_THETA_P) ? ((m & 1) ? -1.0 : 1.0) : ((m & 1) ? 1.0 : -1.0);
      const double * AE = thread->S + (4 * k + 0) * nqp;
      const double * AO = thread->S + (4 * k + 1) * nqp;
      const double * BE = thread->S + (4 * k + 2) * nqp;
      const double * BO = thread->S + (4 * k + 3) * nqp;
      const double pfac = (m == 0) ? 1.0 : 0.5;
      double * H = thread->H;
      double * Z = thread->Z + 2 * k * nf1;
      long p;

      for (j = 0; j < state->nq; ++j)
        {
          double ae = AE[j], ao = AO[j], be = BE[j], bo = BO[j];
          double vA, vB, uA, uB;

          if (m > 0)
            {
              double mant;
              int expo;

              potsynth_pow_scaled(state->sq[j], ps, &mant, &expo);
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

          /* 2pi - theta_j and pi + theta_j: sin(theta) changes sign */
          H[2 * (Nt - 1 - j)] = sgn * vA;
          H[2 * (Nt - 1 - j) + 1] = -sgn * vB;
          H[2 * (Nt / 2 + j)] = sgn * uA;
          H[2 * (Nt / 2 + j) + 1] = -sgn * uB;
        }

      if (cfft_forward(state->pt, H, 1.0) != 0)
        return POTSYNTH_ENOMEM;

      memset(Z, 0, 2 * nf1 * sizeof(double));
      for (p = -(long) L; p <= (long) L; ++p)
        {
          const size_t ip = (size_t) floor_mod(p, (long) Nt);
          const size_t iz = (size_t) floor_mod(p, (long) nf1);
          const double * sh = state->shift + 2 * (size_t) (p + (long) L);
          const double f = state->c1[labs(p)] * state->c2[m] * pfac;
          const double hr = H[2 * ip], hi = H[2 * ip + 1];

          Z[2 * iz] = f * (sh[0] * hr - sh[1] * hi);
          Z[2 * iz + 1] = f * (sh[0] * hi + sh[1] * hr);
        }

      if (cfft_backward(state->p1, Z, 1.0) != 0)
        return POTSYNTH_ENOMEM;
    }

  /* half-complex slots: m = 0 -> column 0 (real part), m >= 1 -> columns 2m-1 (real), 2m (imag) */
  for (r = w->row_lo; r <= w->row_hi; ++r)
    {
      const size_t lrow = (size_t) floor_mod(r, (long) nf1);
      double * row = w->grid + (size_t) (r - w->row_lo) * w->ncols * w->nchp;

      if (m == 0)
        {
          double * g = row + (size_t) (0 - w->col_lo) * w->nchp;

          for (k = 0; k < npair; ++k)
            g[k] = thread->Z[2 * k * nf1 + 2 * lrow];
        }
      else
        {
          double * gr = row + (size_t) ((long) (2 * m - 1) - w->col_lo) * w->nchp;
          double * gi = gr + w->nchp;

          for (k = 0; k < npair; ++k)
            {
              gr[k] = thread->Z[2 * k * nf1 + 2 * lrow];
              gi[k] = thread->Z[2 * k * nf1 + 2 * lrow + 1];
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

/*
fast_row()
  Inverse real FFT along phi of each channel of one stored row, and
periodic padding columns

Inputs: row   - stored row index
        R     - buffer, length nf2
        state - shared set-up data
        w     - workspace
*/

static int
fast_row(const size_t row, double * R, const fast_state_t * state, potsynth_workspace * w)
{
  const size_t nf2 = w->nf2;
  const size_t nchp = w->nchp;
  double * g = w->grid + row * w->ncols * nchp;
  size_t q, k;

  for (q = 0; q < w->nch; ++q)
    {
      long col;

      for (k = 0; k < nf2; ++k)
        R[k] = g[(size_t) ((long) k - w->col_lo) * nchp + q];

      if (rfft_backward(state->p2, R, 1.0) != 0)
        return POTSYNTH_ENOMEM;

      for (col = w->col_lo; col <= w->col_hi; ++col)
        g[(size_t) (col - w->col_lo) * nchp + q] = R[floor_mod(col, (long) nf2)];
    }

  return POTSYNTH_SUCCESS;
}

/* rms size of the theta operator applied to a degree-l function (radial weights) */
static double
op_norm(const int op, const size_t l)
{
  if (op == POTSYNTH_THETA_P)
    return 1.0;

  return sqrt(0.5 * (double) l * ((double) l + 1.0));
}

/* a mod n in [0, n) */
static long
floor_mod(const long a, const long n)
{
  const long r = a % n;
  return (r < 0) ? r + n : r;
}
