/*
 * potsynth_internal.h -- internal definitions shared by the source files
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#ifndef POTSYNTH_INTERNAL_H
#define POTSYNTH_INTERNAL_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "potsynth.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI   3.14159265358979323846264338327950288
#endif
#ifndef M_LN2
#define M_LN2  0.69314718055994530941723212145817657
#endif

/* GCC/Clang vector extensions for the hot kernels; POTSYNTH_NO_VECEXT forces the
 * portable scalar code (used by other compilers, and tested) */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(POTSYNTH_NO_VECEXT)
#define PS_HAVE_VECEXT 1
#endif

/* block size for vectorised loops over nodes/points; one cache line of doubles */
#define PS_NB            8

/* channel dimension of the grid is padded to a multiple of this (SIMD width) */
#define PS_CHAN_ALIGN    4

/* memory alignment in bytes */
#define PS_ALIGN         64

/* largest exponent e such that the scaled Legendre recursion values stay below 2^e */
#define PS_ALF_EXPMAX    960

/* largest power-of-2 down-scaling allowed for a column (bounds the supported degree) */
#define PS_ALF_SCALEMAX  1000

/* largest window width in the calibration tables */
#define PS_WMAX          16

/* ------------------------------------------------------------------------- */
/* utilities (util.c)                                                        */
/* ------------------------------------------------------------------------- */

void *ps_malloc(size_t nbytes);
void *ps_calloc(size_t nbytes);
void ps_free(void *ptr);
size_t ps_next_smooth(size_t n, size_t mult);
int ps_max_threads(int requested);

/* (mantissa, exponent) representation of s^k, computed by repeated squaring
 * with renormalisation so that it never under- or overflows */
void ps_pow_scaled(double s, size_t k, double *mant, int *expo);

/* accurate cos(m*phi), sin(m*phi) for double phi (corrects the rounding of m*phi) */
void ps_sincos_m(size_t m, double phi, double *c, double *s);

/* ------------------------------------------------------------------------- */
/* Associated Legendre functions (alf.c)                                     */
/* ------------------------------------------------------------------------- */

/*
 * 4-pi fully normalised ALFs, M-major storage: index(l,m) = m(2L+1-m)/2 + l.
 *
 * For m = 0 the column recursion produces P(n,0) and dP(n,0)/dtheta.
 * For m >= 1 it produces the scaled, pole-safe quantities
 *
 *   Q(n,m) = 2^-scale[m] * P(n,m) / sin^m(theta)
 *   E(n,m) = 2^-scale[m] * (dP(n,m)/dtheta) / sin^(m-1)(theta)
 *
 * which are polynomials in cos(theta) and never divide by sin(theta). Then
 *   P        = Q * s^m     * 2^scale[m]
 *   P/sin    = Q * s^(m-1) * 2^scale[m]
 *   dP/dth   = E * s^(m-1) * 2^scale[m]
 * scale[m] >= 0 is the smallest power of two keeping |Q|, |E| < 2^PS_ALF_EXPMAX,
 * derived from the analytic maximum of P/sin^m (attained at the poles).
 */
typedef struct
{
  size_t lmax;
  double *ab;      /* a(l,m), b(l,m) interleaved: ab[2*idx], ab[2*idx+1] */
  double *cl;      /* P(m+1,m) = cl[m] x P(m,m) */
  double *qmm;     /* Q(m,m) (scaled), m >= 1; qmm[0] = 1 */
  int *scale;      /* scale[m] */
} ps_alf;

static inline size_t
ps_alf_idx(size_t l, size_t m, size_t lmax)
{
  return ((m * (2 * lmax + 1 - m)) >> 1) + l;
}

int ps_alf_init(ps_alf *alf, size_t lmax);
void ps_alf_free(ps_alf *alf);

/* column recursion for order m at PS_NB nodes: fills Q[(n-m)*PS_NB + j] and
 * E[(n-m)*PS_NB + j] for n = m..lmax */
void ps_alf_column(const ps_alf *alf, size_t m, const double *x, const double *s,
                   double *Q, double *E);

/*
 * Accumulate weighted sums over n of the column values, split by the parity
 * of n - m, for npair pairs of weight rows (A and B) sharing one source array:
 *
 *   acc[(4*k + 0)*NB + j] = sum_{n-m even} WA_k[n-m] src_k[n-m][j]
 *   acc[(4*k + 1)*NB + j] = sum_{n-m odd } WA_k[n-m] src_k[n-m][j]
 *   acc[(4*k + 2)*NB + j] = ... WB_k even,   acc[(4*k + 3)*NB + j] = WB_k odd
 *
 * W holds 2*npair rows of length ldw (row 2k = A, 2k+1 = B); src[k] selects Q
 * (0) or E (1); the sum runs over nloc = n0[k] .. nn-1.
 */
void ps_alf_accum(size_t nn, size_t npair, const double *W, size_t ldw,
                  const unsigned char *src, const size_t *n0,
                  const double *Q, const double *E, double *acc);

/* ------------------------------------------------------------------------- */
/* radial basis (radial.c)                                                   */
/* ------------------------------------------------------------------------- */

typedef struct
{
  double uc, ud;   /* u = uc + ud t, t in [-1,1] */
  size_t rho;      /* number of radial functions */
  size_t ncheb;    /* Chebyshev terms per function */
  double *basis;   /* basis[i*ncheb + k] */
  double *beta;    /* beta[(c*rho + i)*(lmax+1) + n]: u^(n+rpow_c) ~ sum_i beta b_i(u) */
} ps_radial;

int ps_radial_init(ps_radial *rad, double u0, double u1, size_t lmin, size_t lmax,
                   size_t nchan, const int *rpow, const double *wn, double tol);
void ps_radial_free(ps_radial *rad);

/* one-sided Jacobi SVD of the n x p matrix X (column-major, n >= p):
 * on return X holds U*S (orthogonal columns), V the p x p right singular vectors
 * (column-major), sv the singular values; sorted in decreasing order */
int ps_svd_jacobi(size_t n, size_t p, double *X, double *V, double *sv);

/* ------------------------------------------------------------------------- */
/* window (window.c)                                                         */
/* ------------------------------------------------------------------------- */

int ps_window_choose(double eps, double sigma);
void ps_gauss_legendre(size_t n, double *x, double *w);
/* phi_hat(k) for k = 0..nk-1 of the exponential-of-semicircle window of width
 * w grid steps h, shape beta */
int ps_window_ft(size_t nk, int w, double beta, double h, double *out);

static inline double
ps_window(double z, double beta)
{
  double t = 1.0 - z * z;
  return (t > 0.0) ? exp(beta * (sqrt(t) - 1.0)) : 0.0;
}

/* ------------------------------------------------------------------------- */
/* workspaces                                                                */
/* ------------------------------------------------------------------------- */

struct potsynth_workspace
{
  size_t lmin, lmax, mmax;
  double a, rmin, rmax;
  size_t nchan;
  potsynth_params params;
  potsynth_method method;
  int nthreads;

  /* channels */
  int *rpow;
  int *theta_op;
  int *dphi;
  double *rfac;          /* rfac[c*(lmax+1) + n] */

  /* coefficients, 4-pi normalised, M-major: C[idx(n,m)], S[idx(n,m)] */
  double *C, *S;

  ps_alf alf;

  /* fast method */
  int w;
  double sigma, beta, half;
  size_t nf1, nf2;
  double h1, h2, inv_h1, inv_h2;
  long row_lo, row_hi;   /* stored grid rows [row_lo, row_hi] */
  long col_lo, col_hi;   /* stored grid columns [col_lo, col_hi] (periodic padding) */
  size_t nrows, ncols;
  size_t nch;            /* nchan * rho */
  size_t nchp;           /* nch padded to PS_CHAN_ALIGN */
  double *grid;          /* grid[((row-row_lo)*ncols + (col-col_lo))*nchp + q] */
  ps_radial rad;

  /* direct method: per-order weight rows (A, B per channel) and start offsets */
  double *dW;            /* order m at dW + doff[m], rows of length lmax-m+1 */
  size_t *doff;          /* lmax+1 */
  size_t *dn0;           /* dn0[m*nchan + c] */
  unsigned char *dsrc;   /* dsrc[c]: 0 = Q, 1 = E */
};

struct potsynth_eval_workspace
{
  const potsynth_workspace *w;
  size_t chunk;
  int nthreads;

  /* fast method */
  uint32_t *key;         /* sort key per point */
  uint32_t *order;       /* sorted point order */
  size_t *count;         /* counting sort bins */
  size_t nbins;
  double *tscratch;      /* per-thread scratch */
  size_t tstride;        /* doubles per thread */
};

/* direct method (direct.c) */
int ps_direct_init(potsynth_workspace *w);
void ps_direct_free(potsynth_workspace *w);
size_t ps_direct_scratch_size(const potsynth_workspace *w);
void ps_direct_block(const potsynth_workspace *w, double *scratch, size_t nb,
                     const double *r, const double *theta, const double *phi,
                     double *out, size_t ldo);

/* fast method set-up (setup.c) */
int ps_fast_setup(potsynth_workspace *w);

#endif /* POTSYNTH_INTERNAL_H */
