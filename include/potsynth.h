/*
 * potsynth.h
 *
 * Fast synthesis of internal potential-field models at scattered points:
 * generic (field-agnostic) interface. Physical front-ends (potsynth_mag.h,
 * later potsynth_grav.h) translate their quantities into the channels
 * defined here.
 *
 * Each channel c is the quantity
 *
 * f_c(r,theta,phi) = \sum_{l=lmin}^{lmax} \sum_{m=0}^{min(l,mmax)}
 *                      rfac_c(l) (a/r)^(l + rpow_c) T_c[ Pbar_l^m(cos theta) ]
 *                      D_c[ C_lm cos(m phi) + S_lm sin(m phi) ]
 *
 * where Pbar_l^m are 4-pi fully normalized associated Legendre functions
 * (no Condon-Shortley phase), T_c an operator in theta and D_c = (d/dphi)^dphi_c.
 * Constant factors (a, GM/a, ...) are folded into rfac.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_potsynth_h
#define INCLUDED_potsynth_h

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* symbol export: only the public API is exported from the shared library */
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

#define POTSYNTH_VERSION_MAJOR       0
#define POTSYNTH_VERSION_MINOR       1
#define POTSYNTH_VERSION_PATCH       0

/* largest supported SH degree (double precision Holmes-Featherstone scaling) */
#define POTSYNTH_LMAX                2800

/* status codes */
#define POTSYNTH_SUCCESS             0
#define POTSYNTH_EINVAL              1 /* invalid argument */
#define POTSYNTH_ENOMEM              2 /* memory allocation failed */
#define POTSYNTH_EDOM                3 /* evaluation point outside valid domain */
#define POTSYNTH_EUNSUP              4 /* unsupported option (e.g. lmax too large) */

/* operators in theta */
#define POTSYNTH_THETA_P             0 /* Pbar_l^m(cos theta) */
#define POTSYNTH_THETA_DP            1 /* d/dtheta Pbar_l^m */
#define POTSYNTH_THETA_PSIN          2 /* Pbar_l^m / sin(theta), m >= 1, computed without dividing by sin(theta) */

/* evaluation methods */
#define POTSYNTH_METHOD_AUTO         0 /* fast or direct, chosen from lmax (and npts_hint) */
#define POTSYNTH_METHOD_FAST         1 /* gridded transform */
#define POTSYNTH_METHOD_DIRECT       2 /* point-by-point sum */

/*
potsynth_idx()
  Index of SH coefficient (l,m) in a coefficient vector spanning degrees
[lmin,lmax] and orders up to mmax (same convention as mainlib green2_idx):

[ (lmin,-M) ... (lmin,M) (lmin+1,-M') ... (lmin+1,M') ... (lmax,M'') ]

with M = min(l,mmax). Entry (l,m), m >= 0 holds the cosine coefficient
(C_lm or g_l^m), entry (l,-m) the sine coefficient (S_lm or h_l^m).

Input conditions: 0 <= lmin <= l, |m| <= min(l,mmax); use signed arguments
*/
#define potsynth_idx(l, m, lmin, mmax) ( ((lmin) <= (mmax)) ? \
                                           ( ((l) <= (mmax)) ? \
                                               (l)*((l)+1) + (m) - (lmin)*(lmin) : \
                                               (l)*(2*(mmax)+1) + (m) - (mmax)*(mmax) - (lmin)*(lmin) ) \
                                         : ( ((l)-(lmin))*(2*(mmax)+1) + (m) + (mmax) ) )

typedef struct
{
  int rpow;             /* radial power offset: term carries (a/r)^(l + rpow) */
  const double * rfac;  /* per-degree factors rfac[l], l = 0..lmax, or NULL for 1; copied at allocation */
  int theta;            /* operator in theta (POTSYNTH_THETA_xxx) */
  int dphi;             /* number of phi derivatives: 0 or 1 */
} potsynth_channel;

typedef struct
{
  double eps;           /* requested accuracy relative to max output, default 1e-10 */
  int method;           /* POTSYNTH_METHOD_xxx, default POTSYNTH_METHOD_AUTO */
  double sigma;         /* grid oversampling factor; 0 = default (2) */
  int w;                /* window width; 0 = chosen from eps */
  double rank_tol;      /* radial SVD tolerance; 0 = 0.1 * eps */
  int nthreads;         /* OpenMP threads; 0 = all available */
  size_t npts_hint;     /* expected number of evaluation points, used by METHOD_AUTO to
                           choose the faster method including set-up; 0 = unknown */
} potsynth_parameters;

typedef struct
{
  size_t lmin;          /* minimum SH degree */
  size_t lmax;          /* maximum SH degree */
  size_t mmax;          /* maximum SH order */
  size_t nchan;         /* number of channels */
  int method;           /* method selected (POTSYNTH_METHOD_FAST or POTSYNTH_METHOD_DIRECT) */
  int w;                /* window width */
  double sigma;         /* oversampling factor */
  size_t rho;           /* number of radial functions */
  size_t ncheb;         /* Chebyshev terms per radial function */
  size_t nf1;           /* fine grid size in theta (doubled sphere) */
  size_t nf2;           /* fine grid size in phi */
  size_t nrows;         /* stored grid rows */
  size_t grid_bytes;    /* memory used by the grid */
} potsynth_info;

/*
 * potsynth_workspace: everything that depends on the model (coefficients,
 * radial basis, grid). Read-only after allocation, so it may be shared by any
 * number of threads.
 *
 * potsynth_eval_workspace: scratch space for evaluation; each thread calling
 * potsynth_eval() concurrently needs its own.
 */
typedef struct potsynth_workspace potsynth_workspace;
typedef struct potsynth_eval_workspace potsynth_eval_workspace;

/* error handler, called by all functions on error before returning */
typedef void potsynth_error_handler_t(const char * reason, const char * file, int line,
                                      int potsynth_errno);

/*
 * Prototypes
 */

/* error.c */
POTSYNTH_API potsynth_error_handler_t * potsynth_set_error_handler(potsynth_error_handler_t * new_handler);
POTSYNTH_API potsynth_error_handler_t * potsynth_set_error_handler_off(void);
POTSYNTH_API const char * potsynth_strerror(const int potsynth_errno);

/* setup.c */
POTSYNTH_API potsynth_parameters potsynth_default_parameters(void);
POTSYNTH_API size_t potsynth_nlm(const size_t lmin, const size_t lmax, const size_t mmax);
POTSYNTH_API potsynth_workspace * potsynth_alloc(const size_t lmin, const size_t lmax, const size_t mmax,
                                                 const double * coef, const double a,
                                                 const double rmin, const double rmax,
                                                 const size_t nchan, const potsynth_channel * chan,
                                                 const potsynth_parameters * params);
POTSYNTH_API void potsynth_free(potsynth_workspace * w);
POTSYNTH_API int potsynth_get_info(potsynth_info * info, const potsynth_workspace * w);

/* eval.c */
POTSYNTH_API potsynth_eval_workspace * potsynth_eval_alloc(const size_t chunk, const potsynth_workspace * w);
POTSYNTH_API void potsynth_eval_free(potsynth_eval_workspace * ew);
POTSYNTH_API int potsynth_eval(const size_t n, const double * r, const double * theta, const double * phi,
                               double * out, potsynth_eval_workspace * ew);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_potsynth_h */
