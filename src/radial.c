/*
 * radial.c
 *
 * Optimal radial basis (step 1 of the fast method)
 *
 * The radial factors u^(l+rpow), u = a/r, l = lmin..lmax, are sampled at K+1
 * Chebyshev-Lobatto points of [u0, u1], weighted by the size of their
 * contribution to the field, and compressed with an SVD into rho functions
 * b_i(u), stored as Chebyshev series:
 *
 * u^(l+rpow_c) ~ \sum_{i=0}^{rho-1} beta_{c,i,l} b_i(u)
 *
 * The coefficients beta follow from the discrete (Chebyshev-weighted)
 * projection. The SVD is a one-sided Jacobi method, which computes small
 * singular values to high relative accuracy (needed for machine precision
 * operation).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "potsynth_internal.h"

/* temporary arrays of the radial set-up */
typedef struct
{
  size_t nfun;          /* number of weighted functions */
  int * fpow;           /* power of u of each function, length nfun */
  double * fw;          /* weight of each function, length nfun */
  size_t K;             /* Chebyshev degree */
  double * A;           /* samples A[f*(K+1) + j] */
  double * ca;          /* Chebyshev coefficients, same layout as A */
  double * ct;          /* cos(pi i / K), length 2K */
  double * uj;          /* nodes u_j, length K+1 */
  double * sq;          /* sqrt of Lobatto weights, length K+1 */
  double * X;           /* SVD matrix, column-major nfun-by-(K+1) */
  double * V;           /* right singular vectors, (K+1)-by-(K+1) */
  double * sv;          /* singular values, length K+1 */
} radial_state_t;

static void radial_state_free(radial_state_t * state);
static int radial_state_resize(const size_t K, radial_state_t * state);
static void cheb_coeffs(const size_t K, const double * f, const double * ct, double * a);
static double radial_sample(const double uc, const double ud, radial_state_t * state);
static int radial_build(const size_t lmin, const size_t lmax, const size_t nchan, const int * rpow,
                        const double * wn, const double tol, const int maxpow,
                        potsynth_radial_workspace * w, radial_state_t * state);

/*
potsynth_radial_alloc()
  Compute the optimal radial basis for a set of channels

Inputs: u0    - a / rmax
        u1    - a / rmin
        lmin  - minimum SH degree
        lmax  - maximum SH degree
        nchan - number of channels
        rpow  - radial power offset of each channel, length nchan
        wn    - weights wn[c*(lmax+1) + l]: size of the contribution of degree l
                to channel c (0 = absent)
        tol   - relative tolerance of the rank truncation

Return: pointer to workspace

Notes:
1) If rmin = rmax (to rounding), or all functions are constant, a single
constant basis function b_0 = 1 is used
*/

potsynth_radial_workspace *
potsynth_radial_alloc(const double u0, const double u1, const size_t lmin, const size_t lmax,
                      const size_t nchan, const int * rpow, const double * wn, const double tol)
{
  const size_t L1 = lmax + 1;
  int maxpow = rpow[0], minpow = rpow[0];
  potsynth_radial_workspace * w;
  radial_state_t state;
  size_t c, l;
  int status;

  for (c = 1; c < nchan; ++c)
    {
      maxpow = (rpow[c] > maxpow) ? rpow[c] : maxpow;
      minpow = (rpow[c] < minpow) ? rpow[c] : minpow;
    }

  if ((long) lmin + minpow < 0)
    {
      POTSYNTH_ERROR_NULL ("negative powers of a/r are not supported", POTSYNTH_EINVAL);
    }
  else if (!isfinite(pow(u1, (double) lmax + maxpow)))
    {
      POTSYNTH_ERROR_NULL ("(a/r)^lmax overflows: rmin is too small", POTSYNTH_EINVAL);
    }

  w = calloc(1, sizeof(potsynth_radial_workspace));
  if (w == NULL)
    {
      POTSYNTH_ERROR_NULL ("unable to allocate radial workspace", POTSYNTH_ENOMEM);
    }

  w->uc = 0.5 * (u0 + u1);
  w->ud = 0.5 * (u1 - u0);

  /* thin shell, or constant functions: a single radial function b_0 = 1 */
  if (!(w->ud > 1.0e-15 * w->uc) || (long) lmax + maxpow == 0)
    {
      w->rho = 1;
      w->ncheb = 1;
      w->ud = 0.0;
      w->basis = potsynth_aligned_alloc(sizeof(double));
      w->beta = potsynth_aligned_calloc(nchan * L1 * sizeof(double));
      if (!w->basis || !w->beta)
        {
          potsynth_radial_free(w);
          POTSYNTH_ERROR_NULL ("unable to allocate radial basis", POTSYNTH_ENOMEM);
        }

      w->basis[0] = 1.0;
      for (c = 0; c < nchan; ++c)
        {
          for (l = lmin; l <= lmax; ++l)
            w->beta[c * L1 + l] = pow(w->uc, (double) l + rpow[c]);
        }

      return w;
    }

  memset(&state, 0, sizeof(radial_state_t));

  status = radial_build(lmin, lmax, nchan, rpow, wn, tol, maxpow, w, &state);
  radial_state_free(&state);

  if (status)
    {
      potsynth_radial_free(w);
      return NULL;
    }

  return w;
}

void
potsynth_radial_free(potsynth_radial_workspace * w)
{
  if (w->basis)
    potsynth_aligned_free(w->basis);

  if (w->beta)
    potsynth_aligned_free(w->beta);

  free(w);
}

/*
potsynth_svd_jacobi()
  One-sided Jacobi SVD of an n-by-p matrix

Inputs: n  - number of rows
        p  - number of columns
        X  - on input, matrix (column-major); on output, U*S (orthogonal columns)
        V  - (output) right singular vectors, p-by-p column-major
        sv - (output) singular values, length p, in decreasing order

Return: success or error

Notes:
1) Columns below (eps ||X||_F)^2 are rounding noise and are not rotated
against each other (they would never satisfy the relative convergence test)
*/

int
potsynth_svd_jacobi(const size_t n, const size_t p, double * X, double * V, double * sv)
{
  const double tol = DBL_EPSILON * sqrt((double) n);
  double fro2 = 0.0, tiny;
  size_t i, j, k, sweep;

  for (i = 0; i < n * p; ++i)
    fro2 += X[i] * X[i];
  tiny = fro2 * DBL_EPSILON * DBL_EPSILON;

  for (i = 0; i < p * p; ++i)
    V[i] = 0.0;
  for (i = 0; i < p; ++i)
    V[i * p + i] = 1.0;

  for (sweep = 0; sweep < 100; ++sweep)
    {
      int rotated = 0;

      for (i = 0; i + 1 < p; ++i)
        {
          for (j = i + 1; j < p; ++j)
            {
              double * xi = X + i * n;
              double * xj = X + j * n;
              double * vi = V + i * p;
              double * vj = V + j * p;
              double alpha = 0.0, beta = 0.0, gamma = 0.0;
              double zeta, t, c, s;

              for (k = 0; k < n; ++k)
                {
                  alpha += xi[k] * xi[k];
                  beta += xj[k] * xj[k];
                  gamma += xi[k] * xj[k];
                }

              if (alpha <= tiny || beta <= tiny || fabs(gamma) <= tol * sqrt(alpha * beta))
                continue;

              rotated = 1;
              zeta = (beta - alpha) / (2.0 * gamma);
              t = ((zeta >= 0.0) ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
              c = 1.0 / sqrt(1.0 + t * t);
              s = c * t;

              for (k = 0; k < n; ++k)
                {
                  const double xik = xi[k], xjk = xj[k];

                  xi[k] = c * xik - s * xjk;
                  xj[k] = s * xik + c * xjk;
                }

              for (k = 0; k < p; ++k)
                {
                  const double vik = vi[k], vjk = vj[k];

                  vi[k] = c * vik - s * vjk;
                  vj[k] = s * vik + c * vjk;
                }
            }
        }

      if (!rotated)
        break;
    }

  for (i = 0; i < p; ++i)
    {
      double sum = 0.0;

      for (k = 0; k < n; ++k)
        sum += X[i * n + k] * X[i * n + k];

      sv[i] = sqrt(sum);
    }

  /* selection sort into decreasing order (p is small) */
  for (i = 0; i < p; ++i)
    {
      size_t imax = i;

      for (j = i + 1; j < p; ++j)
        {
          if (sv[j] > sv[imax])
            imax = j;
        }

      if (imax != i)
        {
          double tmp = sv[i];

          sv[i] = sv[imax];
          sv[imax] = tmp;

          for (k = 0; k < n; ++k)
            {
              tmp = X[i * n + k];
              X[i * n + k] = X[imax * n + k];
              X[imax * n + k] = tmp;
            }

          for (k = 0; k < p; ++k)
            {
              tmp = V[i * p + k];
              V[i * p + k] = V[imax * p + k];
              V[imax * p + k] = tmp;
            }
        }
    }

  if (sweep == 100)
    {
      POTSYNTH_ERROR ("Jacobi SVD did not converge", POTSYNTH_EINVAL);
    }

  return POTSYNTH_SUCCESS;
}

static void
radial_state_free(radial_state_t * state)
{
  if (state->fpow)
    free(state->fpow);

  if (state->fw)
    free(state->fw);

  if (state->A)
    free(state->A);

  if (state->ca)
    free(state->ca);

  if (state->ct)
    free(state->ct);

  if (state->uj)
    free(state->uj);

  if (state->sq)
    free(state->sq);

  if (state->X)
    free(state->X);

  if (state->V)
    free(state->V);

  if (state->sv)
    free(state->sv);
}

/* (re)allocate the arrays which depend on the Chebyshev degree K */
static int
radial_state_resize(const size_t K, radial_state_t * state)
{
  const size_t nfun = state->nfun;

  free(state->A);
  free(state->ca);
  free(state->ct);
  free(state->uj);
  free(state->sq);

  state->K = K;
  state->A = malloc(nfun * (K + 1) * sizeof(double));
  state->ca = malloc(nfun * (K + 1) * sizeof(double));
  state->ct = malloc(2 * K * sizeof(double));
  state->uj = malloc((K + 1) * sizeof(double));
  state->sq = malloc((K + 1) * sizeof(double));
  if (!state->A || !state->ca || !state->ct || !state->uj || !state->sq)
    {
      POTSYNTH_ERROR ("unable to allocate radial arrays", POTSYNTH_ENOMEM);
    }

  return POTSYNTH_SUCCESS;
}

/*
cheb_coeffs()
  Chebyshev coefficients a[0..K] of the interpolant through f[j] at the
Lobatto points t_j = cos(pi j / K), j = 0..K (K >= 1); ct[i] = cos(pi i / K)
*/

static void
cheb_coeffs(const size_t K, const double * f, const double * ct, double * a)
{
  size_t j, k;

  for (k = 0; k <= K; ++k)
    {
      double sum = 0.5 * (f[0] + ((k & 1) ? -f[K] : f[K]));

      for (j = 1; j < K; ++j)
        sum += f[j] * ct[(j * k) % (2 * K)];

      a[k] = (2.0 / K) * sum;
    }

  a[0] *= 0.5;
  a[K] *= 0.5;
}

/*
radial_sample()
  Sample all weighted functions at the K+1 Lobatto points and compute their
Chebyshev coefficients

Inputs: uc    - center of u interval
        ud    - half width of u interval
        state - state

Return: tail measure sum_f (|a_{K-1}| + |a_K|) relative to sum_f max_j |f(u_j)|
*/

static double
radial_sample(const double uc, const double ud, radial_state_t * state)
{
  const size_t K = state->K;
  double tail = 0.0, scale = 0.0;
  size_t f, j;

  for (j = 0; j < 2 * K; ++j)
    state->ct[j] = cos(M_PI * (double) j / (double) K);

  for (j = 0; j <= K; ++j)
    {
      state->uj[j] = uc + ud * state->ct[j];
      state->sq[j] = (j == 0 || j == K) ? sqrt(0.5) : 1.0;
    }

  for (f = 0; f < state->nfun; ++f)
    {
      double * A = state->A + f * (K + 1);
      double * ca = state->ca + f * (K + 1);
      double fabsmax = 0.0;

      for (j = 0; j <= K; ++j)
        {
          A[j] = state->fw[f] * pow(state->uj[j], (double) state->fpow[f]);
          fabsmax = fmax(fabsmax, fabs(A[j]));
        }

      cheb_coeffs(K, A, state->ct, ca);
      tail += fabs(ca[K]) + fabs(ca[K - 1]);
      scale += fabsmax;
    }

  return (scale > 0.0) ? tail / scale : 0.0;
}

/*
radial_build()
  Build the radial basis

Inputs: lmin   - minimum SH degree
        lmax   - maximum SH degree
        nchan  - number of channels
        rpow   - radial power offsets, length nchan
        wn     - weights, wn[c*(lmax+1) + l]
        tol    - truncation tolerance
        maxpow - largest rpow
        w      - (output) radial workspace
        state  - temporary arrays
*/

static int
radial_build(const size_t lmin, const size_t lmax, const size_t nchan, const int * rpow,
             const double * wn, const double tol, const int maxpow,
             potsynth_radial_workspace * w, radial_state_t * state)
{
  const size_t L1 = lmax + 1;
  const size_t Kmax = (size_t) ((long) lmax + maxpow); /* exact polynomial degree */
  size_t K, Kopt, f, c, i, j, k, l, ncheb, rho, nfun = 0;
  double bmax = 0.0;
  int status;

  /* weighted functions w * u^(l+rpow): one per (distinct rpow, l), with the
   * largest weight over the channels sharing that rpow */
  state->fpow = malloc(nchan * L1 * sizeof(int));
  state->fw = malloc(nchan * L1 * sizeof(double));
  if (!state->fpow || !state->fw)
    {
      POTSYNTH_ERROR ("unable to allocate radial functions", POTSYNTH_ENOMEM);
    }

  for (c = 0; c < nchan; ++c)
    {
      size_t c2;
      int seen = 0;

      for (c2 = 0; c2 < c; ++c2)
        {
          if (rpow[c2] == rpow[c])
            seen = 1;
        }

      if (seen)
        continue;

      for (l = lmin; l <= lmax; ++l)
        {
          double wmax = 0.0;

          for (c2 = c; c2 < nchan; ++c2)
            {
              if (rpow[c2] == rpow[c])
                wmax = fmax(wmax, wn[c2 * L1 + l]);
            }

          if (wmax > 0.0)
            {
              state->fpow[nfun] = (int) l + rpow[c];
              state->fw[nfun] = wmax;
              ++nfun;
            }
        }
    }

  if (nfun == 0)
    {
      /* model is identically zero: keep a single (unused) constant function */
      state->fpow[0] = 0;
      state->fw[0] = 1.0;
      nfun = 1;
    }

  state->nfun = nfun;

  /* Chebyshev degree: double K until the series are resolved to tol/2 */
  K = (Kmax < 8) ? Kmax : 8;
  if (K < 1)
    K = 1;

  while (1)
    {
      double tail;

      status = radial_state_resize(K, state);
      if (status)
        return status;

      tail = radial_sample(w->uc, w->ud, state);
      if (tail <= 0.5 * tol || K >= Kmax)
        break;

      K = (2 * K < Kmax) ? 2 * K : Kmax;
    }

  /* smallest K whose truncation tail is below tol/2 */
  {
    double scale = 0.0, tail = 0.0;

    for (f = 0; f < nfun; ++f)
      {
        double fabsmax = 0.0;

        for (j = 0; j <= K; ++j)
          fabsmax = fmax(fabsmax, fabs(state->A[f * (K + 1) + j]));

        scale += fabsmax;
      }

    Kopt = K;
    for (k = K; k >= 2; --k)
      {
        for (f = 0; f < nfun; ++f)
          tail += fabs(state->ca[f * (K + 1) + k]);

        if (tail > 0.5 * tol * scale)
          break;

        Kopt = k - 1;
      }
  }

  if (Kopt < K)
    {
      K = Kopt;
      status = radial_state_resize(K, state);
      if (status)
        return status;

      radial_sample(w->uc, w->ud, state);
    }

  /* weighted SVD: column j of X = sqrt(c_j) * samples at node j */
  state->X = malloc(nfun * (K + 1) * sizeof(double));
  state->V = malloc((K + 1) * (K + 1) * sizeof(double));
  state->sv = malloc((K + 1) * sizeof(double));
  if (!state->X || !state->V || !state->sv)
    {
      POTSYNTH_ERROR ("unable to allocate SVD arrays", POTSYNTH_ENOMEM);
    }

  for (j = 0; j <= K; ++j)
    {
      for (f = 0; f < nfun; ++f)
        state->X[j * nfun + f] = state->sq[j] * state->A[f * (K + 1) + j];
    }

  status = potsynth_svd_jacobi(nfun, K + 1, state->X, state->V, state->sv);
  if (status)
    return status;

  rho = 0;
  for (i = 0; i <= K; ++i)
    {
      if (state->sv[i] > tol * state->sv[0])
        ++rho;
    }

  if (rho == 0)
    rho = 1;

  /* basis functions b_i(u_j) = U_ji / sqrt(c_j), U_ji = V[i*(K+1) + j] */
  w->basis = potsynth_aligned_calloc(rho * (K + 1) * sizeof(double));
  w->beta = potsynth_aligned_calloc(nchan * rho * L1 * sizeof(double));
  if (!w->basis || !w->beta)
    {
      POTSYNTH_ERROR ("unable to allocate radial basis", POTSYNTH_ENOMEM);
    }

  for (i = 0; i < rho; ++i)
    {
      double * bval = state->A; /* reuse as scratch, length K+1 */

      for (j = 0; j <= K; ++j)
        bval[j] = state->V[i * (K + 1) + j] / state->sq[j];

      cheb_coeffs(K, bval, state->ct, w->basis + i * (K + 1));

      for (k = 0; k <= K; ++k)
        bmax = fmax(bmax, fabs(w->basis[i * (K + 1) + k]));
    }

  /* drop trailing Chebyshev terms below the tolerance (rounding noise) */
  ncheb = 1;
  for (k = 0; k <= K; ++k)
    {
      for (i = 0; i < rho; ++i)
        {
          if (fabs(w->basis[i * (K + 1) + k]) > 0.1 * tol * bmax)
            ncheb = k + 1;
        }
    }

  if (ncheb < K + 1)
    {
      for (i = 0; i < rho; ++i)
        memmove(w->basis + i * ncheb, w->basis + i * (K + 1), ncheb * sizeof(double));
    }

  w->rho = rho;
  w->ncheb = ncheb;

  /* beta_{c,i,l} = sum_j U_ji sqrt(c_j) u_j^(l+rpow_c) (Chebyshev-weighted projection) */
  for (c = 0; c < nchan; ++c)
    {
      for (i = 0; i < rho; ++i)
        {
          for (l = lmin; l <= lmax; ++l)
            {
              double sum = 0.0;

              for (j = 0; j <= K; ++j)
                sum += state->V[i * (K + 1) + j] * state->sq[j] * pow(state->uj[j], (double) l + rpow[c]);

              w->beta[(c * rho + i) * L1 + l] = sum;
            }
        }
    }

  return POTSYNTH_SUCCESS;
}
