/*
 * potsynth.h -- fast synthesis of internal potential-field models at scattered points
 *
 * Core (field-agnostic) interface. Physical front-ends (potsynth_mag.h, later
 * potsynth_grav.h) translate their quantities into the channels defined here.
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 *
 * Pre-1.0: the interface may still change.
 */

#ifndef POTSYNTH_H
#define POTSYNTH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* symbol export (only the public API is exported from the shared library) */
#if defined(_WIN32) && defined(POTSYNTH_SHARED)
#  ifdef POTSYNTH_BUILDING
#    define POTSYNTH_API __declspec(dllexport)
#  else
#    define POTSYNTH_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define POTSYNTH_API __attribute__((visibility("default")))
#else
#  define POTSYNTH_API
#endif

#define POTSYNTH_VERSION_MAJOR 0
#define POTSYNTH_VERSION_MINOR 1
#define POTSYNTH_VERSION_PATCH 0

/* ------------------------------------------------------------------------- */
/* Status codes                                                              */
/* ------------------------------------------------------------------------- */

enum
{
  POTSYNTH_SUCCESS = 0,
  POTSYNTH_EINVAL  = 1,     /* invalid argument */
  POTSYNTH_ENOMEM  = 2,     /* allocation failed */
  POTSYNTH_EDOM    = 3,     /* evaluation point outside the valid domain */
  POTSYNTH_EUNSUP  = 4      /* unsupported option (e.g. lmax too large) */
};

POTSYNTH_API const char *potsynth_strerror(int status);

/* ------------------------------------------------------------------------- */
/* Model definition                                                          */
/* ------------------------------------------------------------------------- */

/*
 * The core evaluates, for each channel c, the quantity
 *
 *   f_c(r,theta,phi) = sum_{n=0}^{L} sum_{m=0}^{n}
 *                        rfac_c[n] * u^(n + rpow_c)
 *                        * T_c[ Pbar_n^m(cos theta) ]
 *                        * D_c[ C_nm cos(m phi) + S_nm sin(m phi) ],    u = a / r
 *
 * where Pbar_n^m are the 4-pi fully normalized associated Legendre functions
 * (geodesy convention, no Condon-Shortley phase), C_nm, S_nm the corresponding
 * coefficients, T_c an operator in theta and D_c = (d/dphi)^dphi_c.
 * The sums run over lmin <= n <= lmax and 0 <= m <= min(n, mmax).
 *
 * Overall constants (a, GM/a, ...) are folded into rfac by the caller.
 *
 * Coefficient storage (same convention as mainlib green2): a single vector
 *
 *   [ (lmin,-M) ... (lmin,M) (lmin+1,-M') ... (lmin+1,M') ... (lmax,M'') ]
 *
 * with M = min(n,mmax) for each degree n. Entry (n,m) with m >= 0 holds C_nm,
 * entry (n,-m) holds S_nm. The index of (n,m) is POTSYNTH_IDX(n,m,lmin,mmax),
 * and the vector length is potsynth_ncoef(lmin,lmax,mmax). lmin = 0 is allowed.
 *
 * This public layout is only read once, at set-up. Internally all Legendre
 * quantities and coefficients are held in M-major order (fixed m contiguous in n),
 * matching the order of the recursions.
 */

/* largest supported degree (double-precision Holmes-Featherstone scaling) */
#define POTSYNTH_LMAX 2800

#define POTSYNTH_IDX(n, m, lmin, mmax)                                        \
  ( ((lmin) <= (mmax)) ?                                                        \
      ( ((n) <= (mmax)) ?                                                       \
          (n) * ((n) + 1) + (m) - (lmin) * (lmin) :                             \
          (n) * (2 * (mmax) + 1) + (m) - (mmax) * (mmax) - (lmin) * (lmin) )    \
    : ( ((n) - (lmin)) * (2 * (mmax) + 1) + (m) + (mmax) ) )

/* number of coefficients for degrees lmin..lmax, orders up to mmax */
POTSYNTH_API size_t potsynth_ncoef(size_t lmin, size_t lmax, size_t mmax);

typedef enum
{
  POTSYNTH_THETA_P     = 0,   /* Pbar_n^m(cos theta)                    */
  POTSYNTH_THETA_DP    = 1,   /* d Pbar_n^m / d theta                   */
  POTSYNTH_THETA_PSIN  = 2    /* Pbar_n^m / sin theta  (m >= 1 only),
                                 computed without dividing by sin theta */
  /* reserved for the gravity front-end: second derivatives */
} potsynth_theta_op;

typedef struct
{
  int rpow;                   /* radial power offset: u^(n + rpow)            */
  const double *rfac;         /* per-degree factor rfac[n], n = 0..L;
                                 NULL means 1. Copied at alloc time.          */
  potsynth_theta_op theta;    /* operator in theta                            */
  int dphi;                   /* number of phi derivatives: 0 or 1            */
} potsynth_channel;

/* ------------------------------------------------------------------------- */
/* Parameters                                                                */
/* ------------------------------------------------------------------------- */

typedef enum
{
  POTSYNTH_METHOD_AUTO   = 0, /* choose fast or direct from L, eps           */
  POTSYNTH_METHOD_FAST   = 1, /* gridded transform                           */
  POTSYNTH_METHOD_DIRECT = 2  /* point-by-point sum                          */
} potsynth_method;

typedef struct
{
  double eps;                 /* target relative accuracy, default 1e-10     */
  potsynth_method method;     /* default AUTO                                */
  double sigma;               /* oversampling factor; 0 = default (2)        */
  int w;                      /* window width; 0 = chosen from eps           */
  double rank_tol;            /* radial SVD tolerance; 0 = 0.1 * eps         */
  int nthreads;               /* OpenMP threads; 0 = all available           */
} potsynth_params;

/* fill params with defaults; always call before setting individual fields */
POTSYNTH_API void potsynth_params_init(potsynth_params *params);

/* ------------------------------------------------------------------------- */
/* Workspaces                                                                */
/* ------------------------------------------------------------------------- */

/*
 * potsynth_workspace: everything that depends on the model (grids, radial basis).
 * Built once by potsynth_alloc(); read-only afterwards, so a single workspace may
 * be shared by any number of threads.
 *
 * potsynth_eval_workspace: per-caller scratch for evaluation (point sorting,
 * per-thread buffers). Evaluation performs no memory allocation; each thread
 * that calls potsynth_eval() concurrently needs its own eval workspace.
 */
typedef struct potsynth_workspace potsynth_workspace;
typedef struct potsynth_eval_workspace potsynth_eval_workspace;

/*
 * Build the transform.
 *
 * lmin, lmax  degree range (0 <= lmin <= lmax)
 * mmax        maximum order (<= lmax)
 * coef        4-pi normalized coefficients, length potsynth_ncoef(lmin,lmax,mmax),
 *             indexed by POTSYNTH_IDX; copied (not retained)
 * a           reference radius (same unit as r)
 * rmin, rmax  radial range of the evaluation points; the fast method rejects
 *             points outside it (POTSYNTH_EDOM). Ignored by the direct method.
 * nchan, chan channel definitions (rfac indexed by degree n, length lmax + 1)
 * params      NULL for defaults
 * status      optional; set to a POTSYNTH_* code
 *
 * Returns NULL on failure.
 */
POTSYNTH_API potsynth_workspace *potsynth_alloc(size_t lmin, size_t lmax, size_t mmax,
                                                const double *coef, double a,
                                                double rmin, double rmax,
                                                size_t nchan, const potsynth_channel *chan,
                                                const potsynth_params *params, int *status);

POTSYNTH_API void potsynth_free(potsynth_workspace *w);

/*
 * Scratch for evaluating with workspace w. Points are processed in chunks of at
 * most 'chunk' points (0 = default); any number of points may be passed to
 * potsynth_eval().
 */
POTSYNTH_API potsynth_eval_workspace *potsynth_eval_alloc(const potsynth_workspace *w, size_t chunk);

POTSYNTH_API void potsynth_eval_free(potsynth_eval_workspace *ew);

/*
 * Evaluate all channels at npts points.
 *
 * r, theta, phi  geocentric radius, colatitude [rad] in [0, pi] (both poles
 *                included), longitude [rad] (any finite value)
 * out            output, channel-major: out[c * npts + i]
 *
 * All points are validated before anything is written. Returns POTSYNTH_EDOM if
 * any point has theta outside [0, pi], r <= 0, or (fast method) r outside
 * [rmin, rmax]. Points with a NaN or infinite coordinate are not an error: their
 * outputs are set to NaN.
 */
POTSYNTH_API int potsynth_eval(const potsynth_workspace *w, potsynth_eval_workspace *ew,
                  size_t npts, const double *r, const double *theta, const double *phi,
                  double *out);

/* ------------------------------------------------------------------------- */
/* Introspection                                                             */
/* ------------------------------------------------------------------------- */

typedef struct
{
  size_t lmin, lmax, mmax;
  size_t nchan;
  potsynth_method method;     /* method actually selected                    */
  int w;                      /* window width                                */
  double sigma;               /* oversampling factor                         */
  size_t rho;                 /* radial rank                                 */
  size_t ncheb;               /* Chebyshev terms per radial function         */
  size_t nf1, nf2;            /* fine grid size (theta on doubled sphere, phi) */
  size_t nrows;               /* stored theta rows                           */
  size_t grid_bytes;          /* memory held by the grids                    */
} potsynth_info;

POTSYNTH_API int potsynth_get_info(const potsynth_workspace *w, potsynth_info *info);

#ifdef __cplusplus
}
#endif

#endif /* POTSYNTH_H */
