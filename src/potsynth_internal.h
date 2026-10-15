/*
 * potsynth_internal.h
 *
 * Internal definitions shared by the library source files
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_potsynth_internal_h
#define INCLUDED_potsynth_internal_h

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "potsynth.h"

#ifndef M_PI
#define M_PI   3.14159265358979323846264338327950288
#endif
#ifndef M_LN2
#define M_LN2  0.69314718055994530941723212145817657
#endif

/* GCC/Clang vector extensions for the hot kernels; POTSYNTH_NO_VECEXT forces the
 * portable scalar code (used by other compilers, and tested) */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(POTSYNTH_NO_VECEXT)
#define POTSYNTH_HAVE_VECEXT         1
#endif

#define POTSYNTH_NB                  8      /* nodes/points per vectorized block (one cache line of doubles) */
#define POTSYNTH_CHAN_ALIGN          4      /* grid channel dimension padded to a multiple of this */
#define POTSYNTH_ALIGN               64     /* memory alignment in bytes */
#define POTSYNTH_ALF_EXPMAX          960    /* scaled ALF columns stay below 2^EXPMAX */
#define POTSYNTH_ALF_SCALEMAX        1000   /* largest power-of-2 column scaling (bounds the degree) */
#define POTSYNTH_WMAX                16     /* largest window width in the calibration tables */
#define POTSYNTH_AUTO_LMIN           24     /* METHOD_AUTO uses the fast method for lmax >= this */
#define POTSYNTH_OMAP_MAX            16     /* largest number of channels with an output map */

/* report an error through the error handler and return */
#define POTSYNTH_ERROR(reason, potsynth_errno) \
  do { potsynth_error(reason, __FILE__, __LINE__, potsynth_errno); return potsynth_errno; } while (0)

#define POTSYNTH_ERROR_NULL(reason, potsynth_errno) \
  do { potsynth_error(reason, __FILE__, __LINE__, potsynth_errno); return NULL; } while (0)

/*
 * 4-pi fully normalized ALFs, M-major storage: idx(l,m) = m(2L+1-m)/2 + l
 *
 * For m = 0 the column recursion produces P(l,0) and dP(l,0)/dtheta. For m >= 1
 * it produces the scaled, pole-safe polynomials in cos(theta)
 *
 * Q(l,m) = 2^-scale[m] P(l,m) / sin^m(theta)
 * E(l,m) = 2^-scale[m] (dP(l,m)/dtheta) / sin^(m-1)(theta)
 *
 * so that P = Q s^m 2^scale[m], P/sin = Q s^(m-1) 2^scale[m] and
 * dP/dtheta = E s^(m-1) 2^scale[m]. scale[m] >= 0 is the smallest power of 2
 * keeping |Q|, |E| < 2^POTSYNTH_ALF_EXPMAX, derived from the analytic maximum of
 * P/sin^m (at the poles).
 */
typedef struct
{
  size_t lmax;          /* maximum SH degree */
  double * ab;          /* a(l,m), b(l,m) interleaved, ab[2*idx(l,m)], ab[2*idx(l,m)+1] */
  double * cl;          /* P(m+1,m) = cl[m] x P(m,m), length lmax+1 */
  double * qmm;         /* scaled Q(m,m), length lmax+1; qmm[0] = 1 */
  int * scale;          /* column scaling exponents, length lmax+1 */
} potsynth_alf_workspace;

/* radial basis: u^(l+rpow_c) ~ sum_i beta[c][i][l] b_i(u), u = uc + ud t, t in [-1,1] */
typedef struct
{
  double uc;            /* center of u interval */
  double ud;            /* half width of u interval */
  size_t rho;           /* number of radial functions */
  size_t ncheb;         /* Chebyshev terms per radial function */
  double * basis;       /* Chebyshev coefficients, basis[i*ncheb + k] */
  double * beta;        /* expansion coefficients, beta[(c*rho + i)*(lmax+1) + l] */
} potsynth_radial_workspace;

struct potsynth_workspace
{
  size_t lmin;          /* minimum SH degree */
  size_t lmax;          /* maximum SH degree */
  size_t mmax;          /* maximum SH order */
  double a;             /* reference radius */
  double rmin;          /* minimum radius of evaluation points (fast method) */
  double rmax;          /* maximum radius of evaluation points (fast method) */
  size_t nchan;         /* number of channels */
  potsynth_parameters params;
  int method;           /* method selected (POTSYNTH_METHOD_FAST or _DIRECT) */
  int nthreads;         /* number of threads */

  /* channels */
  int * rpow;           /* radial power offsets, length nchan */
  int * theta_op;       /* theta operators, length nchan */
  int * dphi;           /* phi derivatives, length nchan */
  double * rfac;        /* per-degree factors, rfac[c*(lmax+1) + l] */

  /* outputs: linear combinations of the channels at each point (front-ends) */
  size_t nout;          /* outputs per point (= nchan without a map) */
  double * omap;        /* output k = sum_c omap[k*nchan + c] channel c, or NULL (outputs = channels) */

  /* coefficients, 4-pi normalized, M-major: C[idx(l,m)], S[idx(l,m)] */
  double * C;
  double * S;

  potsynth_alf_workspace * alf_workspace_p;

  /* fast method */
  int w;                /* window width */
  double sigma;         /* oversampling factor */
  double beta;          /* window shape parameter */
  double half;          /* w / 2 */
  size_t nf1;           /* fine grid size in theta (doubled sphere) */
  size_t nf2;           /* fine grid size in phi */
  double h1;            /* grid spacing in theta */
  double h2;            /* grid spacing in phi */
  double inv_h1;        /* 1 / h1 */
  double inv_h2;        /* 1 / h2 */
  long row_lo;          /* first stored grid row */
  long row_hi;          /* last stored grid row */
  long col_lo;          /* first stored grid column (periodic padding) */
  long col_hi;          /* last stored grid column */
  size_t nrows;         /* number of stored rows */
  size_t ncols;         /* number of stored columns */
  size_t nch;           /* number of grid channels, nchan * rho */
  size_t nchp;          /* nch padded to POTSYNTH_CHAN_ALIGN */
  double * grid;        /* grid[((row - row_lo)*ncols + (col - col_lo))*nchp + q] */
  potsynth_radial_workspace * radial_workspace_p;

  /* direct method: each channel is a sum of terms, each the sum over l of one
   * scaled column (Q, E or G) with a weight row, times a power of sin(theta) */
  size_t nterm;         /* number of terms */
  int * tcol;           /* column of each term (POTSYNTH_COL_xxx), length nterm */
  int * tpost;          /* factor applied after the sum (POTSYNTH_POST_xxx), length nterm */
  size_t * tchan;       /* channel of each term, length nterm */
  int needG;            /* 1 if any term uses column G */
  double * dW;          /* weight rows (A, B per term) of order m at dW + doff[m],
                           rows of length lmax-m+1 */
  size_t * doff;        /* offsets into dW, length lmax+1 */
};

/* scaled ALF columns used by the direct method */
#define POTSYNTH_COL_Q               0 /* Q(l,m) = Pbar / sin^m */
#define POTSYNTH_COL_E               1 /* E(l,m) = (dPbar/dtheta) / sin^(m-1); dPbar/dtheta for m = 0 */
#define POTSYNTH_COL_G               2 /* G(l,m) = -dQ/dx, x = cos(theta) */

/* factor applied to a term after the sum over l, s = sin(theta) */
#define POTSYNTH_POST_SM             0 /* s^m */
#define POTSYNTH_POST_SM1            1 /* s^(m-1), 1 for m = 0 */
#define POTSYNTH_POST_XSM            2 /* x s^m */
#define POTSYNTH_POST_SM2            3 /* s^(m-2), m >= 2 only */
#define POTSYNTH_POST_XSM2           4 /* x s^(m-2), m >= 2 only */

struct potsynth_eval_workspace
{
  const potsynth_workspace * w;
  size_t chunk;         /* points per chunk */
  int nthreads;         /* number of threads */

  /* fast method */
  uint32_t * key;       /* sort key of each point, length chunk */
  uint32_t * order;     /* sorted point order, length chunk */
  size_t * count;       /* counting sort bins, length nbins+1 */
  size_t nbins;         /* number of sort bins */

  double * tscratch;    /* per-thread scratch space, nthreads * tstride */
  size_t tstride;       /* doubles of scratch per thread */
};

/* exponential of semicircle window phi(z) = exp(beta (sqrt(1 - z^2) - 1)) */
static inline double
potsynth_window(const double z, const double beta)
{
  const double t = 1.0 - z * z;
  return (t > 0.0) ? exp(beta * (sqrt(t) - 1.0)) : 0.0;
}

/* index of (l,m) in M-major ALF arrays */
static inline size_t
potsynth_alf_idx(const size_t l, const size_t m, const size_t lmax)
{
  return ((m * (2 * lmax + 1 - m)) >> 1) + l;
}

/*
 * Prototypes
 */

/* alf.c */
potsynth_alf_workspace * potsynth_alf_alloc(const size_t lmax);
void potsynth_alf_free(potsynth_alf_workspace * w);
void potsynth_alf_column(const size_t m, const double * x, const double * s,
                         double * Q, double * E, double * G, const potsynth_alf_workspace * w);
void potsynth_alf_accum(const size_t nn, const size_t npair, const double * W, const size_t ldw,
                        const unsigned char * src, const size_t * n0,
                        const double * Q, const double * E, const double * G, double * acc);
size_t potsynth_op_nterm(const int op);
void potsynth_op_term(const int op, const size_t sub, int * col, int * post);
double potsynth_op_factor(const int op, const size_t sub, const size_t l, const size_t m);

/* direct.c */
int potsynth_direct_init(potsynth_workspace * w);
void potsynth_direct_free(potsynth_workspace * w);
size_t potsynth_direct_scratch_size(const potsynth_workspace * w);
void potsynth_direct_block(const size_t nb, const double * r, const double * theta, const double * phi,
                           const size_t ldo, double * out, double * scratch, const potsynth_workspace * w);

/* error.c */
void potsynth_error(const char * reason, const char * file, int line, int potsynth_errno);

/* fast.c */
int potsynth_fast_init(potsynth_workspace * w);

/* radial.c */
potsynth_radial_workspace * potsynth_radial_alloc(const double u0, const double u1,
                                                  const size_t lmin, const size_t lmax,
                                                  const size_t nchan, const int * rpow,
                                                  const double * wn, const double tol);
void potsynth_radial_free(potsynth_radial_workspace * w);
int potsynth_svd_jacobi(const size_t n, const size_t p, double * X, double * V, double * sv);

/* setup.c */
int potsynth_set_outputs(potsynth_workspace * w, const size_t nout, const double * map);
void potsynth_tensor_channels(const size_t lmax, const double f, potsynth_channel * chan, double * rfac);
void potsynth_tensor_map(const size_t nchan, const size_t c0, double * map);

/* util.c */
void * potsynth_aligned_alloc(const size_t nbytes);
void * potsynth_aligned_calloc(const size_t nbytes);
void potsynth_aligned_free(void * ptr);
size_t potsynth_next_smooth(const size_t n, const size_t mult);
int potsynth_max_threads(const int requested);
void potsynth_pow_scaled(const double s, const size_t k, double * mant, int * expo);
void potsynth_sincos_m(const size_t m, const double phi, double * c, double * s);

/* window.c */
int potsynth_window_choose(const double eps, const double sigma);
void potsynth_gauss_legendre(const size_t n, double * x, double * w);
int potsynth_window_ft(const size_t nk, const int w, const double beta, const double h, double * out);

#endif /* INCLUDED_potsynth_internal_h */
