/*
 * radial.c -- optimal radial basis (Step 1 of the method)
 *
 * The radial factors u^(n+rpow), u = a/r, n = lmin..lmax, are sampled at K+1
 * Chebyshev-Lobatto points of [u0, u1], weighted by the size of their contribution
 * to the field, and compressed with an SVD into rho functions b_i(u), stored as
 * Chebyshev series. The coefficients beta of each u^(n+rpow) in that basis follow
 * from the discrete (Chebyshev-weighted) projection.
 *
 * The SVD is a one-sided Jacobi method, which computes small singular values to
 * high relative accuracy (needed for machine-precision operation).
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"

int
ps_svd_jacobi(size_t n, size_t p, double *X, double *V, double *sv)
{
  const double tol = DBL_EPSILON * sqrt((double) n);
  double fro2 = 0.0, tiny;
  size_t i, j, k, sweep;

  /* columns below (eps * ||X||_F)^2 are rounding noise: rotating them against
   * each other would never satisfy the relative convergence test */
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
              double *xi = X + i * n, *xj = X + j * n;
              double *vi = V + i * p, *vj = V + j * p;
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
                  const double a = xi[k], b = xj[k];
                  xi[k] = c * a - s * b;
                  xj[k] = s * a + c * b;
                }

              for (k = 0; k < p; ++k)
                {
                  const double a = vi[k], b = vj[k];
                  vi[k] = c * a - s * b;
                  vj[k] = s * a + c * b;
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
        if (sv[j] > sv[imax])
          imax = j;

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

  return (sweep < 100) ? POTSYNTH_SUCCESS : POTSYNTH_EINVAL;
}

/* temporaries of the radial set-up, freed in one place */
typedef struct
{
  int *fpow;       /* power of u of each weighted function */
  double *fw;      /* weight of each function */
  double *A;       /* samples A[f*(K+1) + j] */
  double *ca;      /* Chebyshev coefficients, same layout */
  double *ct;      /* cosine table, 2K entries */
  double *X;       /* SVD matrix, column-major nfun x (K+1) */
  double *V;       /* right singular vectors */
  double *sv;      /* singular values */
  double *uj;      /* nodes u_j */
  double *sq;      /* sqrt of the Lobatto weights */
} radial_tmp;

static void
radial_tmp_free(radial_tmp *t)
{
  free(t->fpow);
  free(t->fw);
  free(t->A);
  free(t->ca);
  free(t->ct);
  free(t->X);
  free(t->V);
  free(t->sv);
  free(t->uj);
  free(t->sq);
  memset(t, 0, sizeof(radial_tmp));
}

/* (re)allocate the K-dependent temporaries */
static int
radial_tmp_resize(radial_tmp *t, size_t nfun, size_t K)
{
  free(t->A);
  free(t->ca);
  free(t->ct);
  free(t->uj);
  free(t->sq);
  t->A = malloc(nfun * (K + 1) * sizeof(double));
  t->ca = malloc(nfun * (K + 1) * sizeof(double));
  t->ct = malloc(2 * K * sizeof(double));
  t->uj = malloc((K + 1) * sizeof(double));
  t->sq = malloc((K + 1) * sizeof(double));

  return (t->A && t->ca && t->ct && t->uj && t->sq) ? POTSYNTH_SUCCESS : POTSYNTH_ENOMEM;
}

/* Chebyshev coefficients a[0..K] of the interpolant through f[j*stride] at the
 * Lobatto points t_j = cos(pi j / K), j = 0..K (K >= 1); ct[i] = cos(pi i / K) */
static void
cheb_coeffs(size_t K, const double *f, size_t stride, const double *ct, double *a)
{
  size_t j, k;

  for (k = 0; k <= K; ++k)
    {
      double sum = 0.5 * (f[0] + ((k & 1) ? -f[K * stride] : f[K * stride]));

      for (j = 1; j < K; ++j)
        sum += f[j * stride] * ct[(j * k) % (2 * K)];

      a[k] = (2.0 / K) * sum;
    }

  a[0] *= 0.5;
  a[K] *= 0.5;
}

/* sample all functions at the K+1 Lobatto points and compute their Chebyshev
 * coefficients; returns the tail measure sum_f (|a_{K-1}| + |a_K|) relative to
 * sum_f max_j |f(u_j)| */
static double
radial_sample(radial_tmp *t, size_t nfun, size_t K, double uc, double ud)
{
  double tail = 0.0, scale = 0.0;
  size_t f, j;

  for (j = 0; j < 2 * K; ++j)
    t->ct[j] = cos(M_PI * (double) j / (double) K);

  for (j = 0; j <= K; ++j)
    {
      t->uj[j] = uc + ud * t->ct[j];
      t->sq[j] = (j == 0 || j == K) ? sqrt(0.5) : 1.0;
    }

  for (f = 0; f < nfun; ++f)
    {
      double *A = t->A + f * (K + 1);
      double *ca = t->ca + f * (K + 1);
      double fmaxv = 0.0;

      for (j = 0; j <= K; ++j)
        {
          A[j] = t->fw[f] * pow(t->uj[j], (double) t->fpow[f]);
          fmaxv = fmax(fmaxv, fabs(A[j]));
        }

      cheb_coeffs(K, A, 1, t->ct, ca);
      tail += fabs(ca[K]) + ((K >= 1) ? fabs(ca[K - 1]) : 0.0);
      scale += fmaxv;
    }

  return (scale > 0.0) ? tail / scale : 0.0;
}

static int
radial_build(ps_radial *rad, radial_tmp *t, size_t lmin, size_t lmax, size_t nchan,
             const int *rpow, const double *wn, double tol, int maxpow)
{
  const size_t L1 = lmax + 1;
  const size_t Kmax = (size_t) ((long) lmax + maxpow);  /* exact polynomial degree */
  size_t nfun = 0, K, Kopt, f, c, i, j, k, n, ncheb, rho;
  double bmax = 0.0;
  int status;

  /* list of weighted functions w * u^(n+rpow): one per (distinct rpow, n), with the
   * largest weight over the channels sharing that rpow */
  t->fpow = malloc(nchan * L1 * sizeof(int));
  t->fw = malloc(nchan * L1 * sizeof(double));
  if (!t->fpow || !t->fw)
    return POTSYNTH_ENOMEM;

  for (c = 0; c < nchan; ++c)
    {
      size_t c2;
      int seen = 0;

      for (c2 = 0; c2 < c; ++c2)
        if (rpow[c2] == rpow[c])
          seen = 1;
      if (seen)
        continue;

      for (n = lmin; n <= lmax; ++n)
        {
          double wmax = 0.0;

          for (c2 = c; c2 < nchan; ++c2)
            if (rpow[c2] == rpow[c])
              wmax = fmax(wmax, wn[c2 * L1 + n]);

          if (wmax > 0.0)
            {
              t->fpow[nfun] = (int) n + rpow[c];
              t->fw[nfun] = wmax;
              ++nfun;
            }
        }
    }

  if (nfun == 0)
    {
      /* model is identically zero: keep a single (unused) constant function */
      t->fpow[0] = 0;
      t->fw[0] = 1.0;
      nfun = 1;
    }

  /* Chebyshev degree: double K until the series are resolved to tol/2, then
   * pick the smallest K whose truncation tail is below tol/2 */
  K = (Kmax < 8) ? Kmax : 8;
  if (K < 1)
    K = 1;

  for (;;)
    {
      double tail;

      status = radial_tmp_resize(t, nfun, K);
      if (status)
        return status;

      tail = radial_sample(t, nfun, K, rad->uc, rad->ud);
      if (tail <= 0.5 * tol || K >= Kmax)
        break;

      K = (2 * K < Kmax) ? 2 * K : Kmax;
    }

  {
    /* same normalisation as radial_sample(): sum over functions of max |f| */
    double scale = 0.0, tail = 0.0;

    for (f = 0; f < nfun; ++f)
      {
        double fmaxv = 0.0;
        for (j = 0; j <= K; ++j)
          fmaxv = fmax(fmaxv, fabs(t->A[f * (K + 1) + j]));
        scale += fmaxv;
      }

    Kopt = K;
    for (k = K; k >= 2; --k)
      {
        for (f = 0; f < nfun; ++f)
          tail += fabs(t->ca[f * (K + 1) + k]);
        if (tail > 0.5 * tol * scale)
          break;
        Kopt = k - 1;
      }
  }

  if (Kopt < K)
    {
      K = Kopt;
      status = radial_tmp_resize(t, nfun, K);
      if (status)
        return status;
      radial_sample(t, nfun, K, rad->uc, rad->ud);
    }

  /* weighted SVD: X (nfun x K+1) column j = sqrt(c_j) * samples at node j */
  t->X = malloc(nfun * (K + 1) * sizeof(double));
  t->V = malloc((K + 1) * (K + 1) * sizeof(double));
  t->sv = malloc((K + 1) * sizeof(double));
  if (!t->X || !t->V || !t->sv)
    return POTSYNTH_ENOMEM;

  for (j = 0; j <= K; ++j)
    for (f = 0; f < nfun; ++f)
      t->X[j * nfun + f] = t->sq[j] * t->A[f * (K + 1) + j];

  status = ps_svd_jacobi(nfun, K + 1, t->X, t->V, t->sv);
  if (status)
    return status;

  rho = 0;
  for (i = 0; i <= K; ++i)
    if (t->sv[i] > tol * t->sv[0])
      ++rho;
  if (rho == 0)
    rho = 1;

  /* basis functions: b_i(u_j) = U_ji / sqrt(c_j), U_ji = V[i*(K+1) + j] */
  rad->basis = ps_calloc(rho * (K + 1) * sizeof(double));
  rad->beta = ps_calloc(nchan * rho * L1 * sizeof(double));
  if (!rad->basis || !rad->beta)
    return POTSYNTH_ENOMEM;

  for (i = 0; i < rho; ++i)
    {
      double *bval = t->A;      /* reuse as scratch, length K+1 */

      for (j = 0; j <= K; ++j)
        bval[j] = t->V[i * (K + 1) + j] / t->sq[j];

      cheb_coeffs(K, bval, 1, t->ct, rad->basis + i * (K + 1));

      for (k = 0; k <= K; ++k)
        bmax = fmax(bmax, fabs(rad->basis[i * (K + 1) + k]));
    }

  /* drop trailing Chebyshev terms below the tolerance (rounding noise) */
  ncheb = 1;
  for (k = 0; k <= K; ++k)
    for (i = 0; i < rho; ++i)
      if (fabs(rad->basis[i * (K + 1) + k]) > 0.1 * tol * bmax)
        ncheb = k + 1;

  if (ncheb < K + 1)
    {
      for (i = 0; i < rho; ++i)
        memmove(rad->basis + i * ncheb, rad->basis + i * (K + 1), ncheb * sizeof(double));
    }

  rad->rho = rho;
  rad->ncheb = ncheb;

  /* beta[c][i][n] = sum_j U_ji sqrt(c_j) u_j^(n+rpow_c) (Chebyshev-weighted projection) */
  for (c = 0; c < nchan; ++c)
    for (i = 0; i < rho; ++i)
      for (n = lmin; n <= lmax; ++n)
        {
          double sum = 0.0;

          for (j = 0; j <= K; ++j)
            sum += t->V[i * (K + 1) + j] * t->sq[j] * pow(t->uj[j], (double) n + rpow[c]);

          rad->beta[(c * rho + i) * L1 + n] = sum;
        }

  return POTSYNTH_SUCCESS;
}

int
ps_radial_init(ps_radial *rad, double u0, double u1, size_t lmin, size_t lmax,
               size_t nchan, const int *rpow, const double *wn, double tol)
{
  const size_t L1 = lmax + 1;
  int maxpow = rpow[0], minpow = rpow[0];
  radial_tmp t;
  size_t c, n;
  int status;

  memset(rad, 0, sizeof(ps_radial));
  memset(&t, 0, sizeof(radial_tmp));
  rad->uc = 0.5 * (u0 + u1);
  rad->ud = 0.5 * (u1 - u0);

  for (c = 1; c < nchan; ++c)
    {
      maxpow = (rpow[c] > maxpow) ? rpow[c] : maxpow;
      minpow = (rpow[c] < minpow) ? rpow[c] : minpow;
    }

  /* negative powers of u are not supported */
  if ((long) lmin + minpow < 0)
    return POTSYNTH_EINVAL;

  /* u^(n+rpow) must be representable over the radial range */
  if (!isfinite(pow(u1, (double) lmax + maxpow)))
    return POTSYNTH_EINVAL;

  /* thin shell, or constant functions: a single radial function b_0 = 1 */
  if (!(rad->ud > 1e-15 * rad->uc) || (long) lmax + maxpow == 0)
    {
      rad->rho = 1;
      rad->ncheb = 1;
      rad->ud = 0.0;
      rad->basis = ps_malloc(sizeof(double));
      rad->beta = ps_calloc(nchan * L1 * sizeof(double));
      if (!rad->basis || !rad->beta)
        {
          ps_radial_free(rad);
          return POTSYNTH_ENOMEM;
        }

      rad->basis[0] = 1.0;
      for (c = 0; c < nchan; ++c)
        for (n = lmin; n <= lmax; ++n)
          rad->beta[c * L1 + n] = pow(rad->uc, (double) n + rpow[c]);

      return POTSYNTH_SUCCESS;
    }

  status = radial_build(rad, &t, lmin, lmax, nchan, rpow, wn, tol, maxpow);
  radial_tmp_free(&t);

  if (status)
    ps_radial_free(rad);

  return status;
}

void
ps_radial_free(ps_radial *rad)
{
  ps_free(rad->basis);
  ps_free(rad->beta);
  memset(rad, 0, sizeof(ps_radial));
}
