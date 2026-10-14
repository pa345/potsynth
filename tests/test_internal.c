/*
 * test_internal.c -- unit tests of the internal components, each against an
 * independent reference (long double, brute force, or an analytic identity)
 *
 * White-box test: compiled from the library sources.
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"
#include "pocketfft.h"

static int nfail = 0;

#define CHECK(cond, ...)                                                        \
  do {                                                                          \
    if (!(cond))                                                                \
      {                                                                         \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);                             \
        printf(__VA_ARGS__);                                                    \
        printf("\n");                                                           \
        ++nfail;                                                                \
      }                                                                         \
  } while (0)

static unsigned long long rng_state = 0x2545F4914F6CDD1DULL;

static double
urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double) (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* ------------------------------------------------------------------------- */

static void
test_next_smooth(void)
{
  size_t n, mult, k;
  int ok = 1;

  for (mult = 1; mult <= 4; mult *= 2)
    for (n = 1; n <= 3000; ++n)
      {
        const size_t r = ps_next_smooth(n, mult);
        size_t t = r;

        while (t % 2 == 0) t /= 2;
        while (t % 3 == 0) t /= 3;
        while (t % 5 == 0) t /= 5;

        if (r < n || r % mult != 0 || t != 1)
          ok = 0;

        /* minimal: no smaller valid value in [n, r) */
        for (k = n; k < r && ok; ++k)
          {
            size_t q = k;
            if (k % mult)
              continue;
            while (q % 2 == 0) q /= 2;
            while (q % 3 == 0) q /= 3;
            while (q % 5 == 0) q /= 5;
            if (q == 1)
              ok = 0;
          }
      }

  CHECK(ok, "ps_next_smooth");
  printf("  next_smooth: %s\n", ok ? "ok" : "FAIL");
}

static void
test_pow_scaled(void)
{
  const double svals[] = { 0.0, 1.0, -1.0, 0.5, -0.7, 0.999999, 1e-3, -1e-3, 1e-100, 1e-300,
                           3e-310, 0.389, 1.0 / 3.0 };
  double maxerr = 0.0;
  size_t i, k;

  for (i = 0; i < sizeof(svals) / sizeof(svals[0]); ++i)
    for (k = 0; k <= 3000; k += (k < 20) ? 1 : 37)
      {
        const double s = svals[i];
        double mant;
        int expo;

        ps_pow_scaled(s, k, &mant, &expo);

        if (s == 0.0)
          {
            CHECK(mant == ((k == 0) ? 1.0 : 0.0), "pow_scaled(0,%zu)", k);
            continue;
          }

        /* compare in log2 (the value itself may be far outside double range);
         * relative error = |delta log2| * ln 2. Repeated squaring doubles earlier
         * rounding errors, so the error is bounded by about k * eps (as for a
         * running product); measured relative to (k + 1) * eps. */
        {
          const long double ref = (long double) k * log2l(fabsl((long double) s));
          const long double got = log2l(fabsl((long double) mant)) + expo;
          const int refsign = (s < 0.0 && (k & 1)) ? -1 : 1;
          const double err = (double) (fabsl(got - ref) * M_LN2) / ((k + 1.0) * DBL_EPSILON);

          CHECK((mant < 0) == (refsign < 0), "pow_scaled sign s=%g k=%zu", s, k);
          CHECK((k == 0 && mant == 1.0 && expo == 0) || (fabs(mant) >= 0.5 && fabs(mant) < 1.0),
                "pow_scaled mantissa s=%g k=%zu", s, k);
          if (err > maxerr)
            maxerr = err;
        }
      }

  CHECK(maxerr < 1.0, "pow_scaled error %.2f (k+1) eps", maxerr);
  printf("  pow_scaled: max err %.2f (k+1) eps %s\n", maxerr, (maxerr < 1.0) ? "ok" : "FAIL");
}

static void
test_sincos_m(void)
{
  const double phis[] = { 0.0, 1e-10, 0.1, 1.0, 2.0, M_PI, 4.0, 2.0 * M_PI - 1e-12, 6.2 };
  double maxerr = 0.0;
  size_t i, m;

  for (i = 0; i < sizeof(phis) / sizeof(phis[0]); ++i)
    for (m = 0; m <= 2800; m += (m < 50) ? 1 : 13)
      {
        double c, s;
        /* m * phi needs up to 65 bits: keep the exact long-double rounding error */
        const long double a = (long double) m * (long double) phis[i];
        const long double da = fmal((long double) m, (long double) phis[i], -a);
        const long double cref = cosl(a) - da * sinl(a), sref = sinl(a) + da * cosl(a);

        ps_sincos_m(m, phis[i], &c, &s);
        maxerr = fmax(maxerr, (double) fabsl((long double) c - cref));
        maxerr = fmax(maxerr, (double) fabsl((long double) s - sref));
      }

  /* sin(m phi) for the double phi, accurate to a few ulp regardless of m */
  CHECK(maxerr < 4.0 * DBL_EPSILON, "sincos_m error %.2e", maxerr);
  printf("  sincos_m: max abs err %.1e %s\n", maxerr, (maxerr < 4.0 * DBL_EPSILON) ? "ok" : "FAIL");
}

/* ------------------------------------------------------------------------- */

static void
test_gauss_legendre(void)
{
  const size_t ns[] = { 1, 2, 3, 10, 57, 100, 228 };
  double maxerr = 0.0;
  size_t i, j, d;

  for (i = 0; i < sizeof(ns) / sizeof(ns[0]); ++i)
    {
      const size_t n = ns[i];
      double *x = malloc(n * sizeof(double)), *w = malloc(n * sizeof(double));

      ps_gauss_legendre(n, x, w);

      /* exact for x^d, d <= 2n-1 (d <= 60 to stay well conditioned) */
      for (d = 0; d <= 2 * n - 1 && d <= 60; ++d)
        {
          long double sum = 0.0L;
          const double ref = (d & 1) ? 0.0 : 2.0 / (d + 1.0);

          for (j = 0; j < n; ++j)
            sum += (long double) w[j] * powl(x[j], (long double) d);
          maxerr = fmax(maxerr, fabs((double) sum - ref));
        }

      for (j = 0; j < n; ++j)
        {
          CHECK(fabs(x[j] + x[n - 1 - j]) < 1e-15, "GL symmetry n=%zu", n);
          CHECK(j == 0 || x[j] > x[j - 1], "GL ordering n=%zu", n);
        }

      free(x);
      free(w);
    }

  CHECK(maxerr < 1e-14, "Gauss-Legendre moments %.2e", maxerr);
  printf("  gauss_legendre: max moment err %.1e %s\n", maxerr, (maxerr < 1e-14) ? "ok" : "FAIL");
}

/* phi_hat(k) by tanh-sinh (double exponential) quadrature in z, in long double:
 * converges exponentially despite the endpoint singularity of phi, and is
 * independent of the library's substitution + Gauss-Legendre rule */
static long double
window_ft_ref(double k, int w, double beta, double h)
{
  const long double alpha = 0.5L * w * h, step = 1.0L / 64.0L;
  long double sum = 0.0L;
  int i;

  for (i = -64 * 6; i <= 64 * 6; ++i)
    {
      const long double tau = i * step;
      const long double sh = 0.5L * M_PI * sinhl(tau);
      const long double z = tanhl(sh);
      const long double wgt = 0.5L * M_PI * coshl(tau) / (coshl(sh) * coshl(sh));
      const long double omz2 = 1.0L / (coshl(sh) * coshl(sh));   /* 1 - z^2, exact form */

      sum += wgt * expl(beta * (sqrtl(omz2) - 1.0L)) * cosl(k * alpha * z);
    }

  return alpha * sum * step;
}

static void
test_window_ft(void)
{
  const int ws[] = { 4, 7, 11, 16 };
  double maxerr = 0.0;
  size_t i, k;

  for (i = 0; i < sizeof(ws) / sizeof(ws[0]); ++i)
    {
      const int w = ws[i];
      const double sigma = 2.0, beta = M_PI * (1.0 - 1.0 / (2.0 * sigma)) * w;
      const size_t L = 185, nf = ps_next_smooth((size_t) (sigma * (2 * L + 1)), 2);
      const double h = 2.0 * M_PI / nf;
      double *ft = malloc((L + 1) * sizeof(double));

      ps_window_ft(L + 1, w, beta, h, ft);
      for (k = 0; k <= L; k += 5)
        {
          const long double ref = window_ft_ref((double) k, w, beta, h);
          maxerr = fmax(maxerr, (double) fabsl((ft[k] - ref) / ref));
        }

      /* phi_hat must be positive and decreasing over the band */
      for (k = 1; k <= L; ++k)
        CHECK(ft[k] > 0.0 && ft[k] < ft[k - 1], "window_ft not decreasing w=%d k=%zu", w, k);

      free(ft);
    }

  CHECK(maxerr < 1e-13, "window_ft rel err %.2e", maxerr);
  printf("  window_ft: max rel err %.1e %s\n", maxerr, (maxerr < 1e-13) ? "ok" : "FAIL");

  /* window width choice: monotone in eps */
  CHECK(ps_window_choose(1e-6, 2.0) == 7 && ps_window_choose(1e-10, 2.0) == 11, "window_choose");
  CHECK(ps_window_choose(1e-30, 2.0) == PS_WMAX && ps_window_choose(1.0, 2.0) == 4,
        "window_choose limits");
}

/* ------------------------------------------------------------------------- */

/* random orthogonal n x n (column-major) from Householder reflections */
static void
random_orthogonal(size_t n, double *Q)
{
  double *v = malloc(n * sizeof(double));
  size_t i, j, k, r;

  for (i = 0; i < n * n; ++i)
    Q[i] = 0.0;
  for (i = 0; i < n; ++i)
    Q[i * n + i] = 1.0;

  for (r = 0; r < n; ++r)
    {
      double nrm = 0.0;

      for (i = 0; i < n; ++i)
        {
          v[i] = 2.0 * urand() - 1.0;
          nrm += v[i] * v[i];
        }
      nrm = sqrt(nrm);
      for (i = 0; i < n; ++i)
        v[i] /= nrm;

      /* Q = (I - 2 v v^T) Q */
      for (j = 0; j < n; ++j)
        {
          double d = 0.0;
          for (k = 0; k < n; ++k)
            d += v[k] * Q[j * n + k];
          for (k = 0; k < n; ++k)
            Q[j * n + k] -= 2.0 * d * v[k];
        }
    }

  free(v);
}

static void
test_svd(void)
{
  const size_t cases[][2] = { { 40, 10 }, { 300, 25 }, { 7, 7 }, { 5, 9 } };
  size_t ic;

  for (ic = 0; ic < sizeof(cases) / sizeof(cases[0]); ++ic)
    {
      const size_t n = cases[ic][0], p = cases[ic][1];
      const size_t r = (n < p) ? n : p;
      double *U = malloc(n * n * sizeof(double)), *W = malloc(p * p * sizeof(double));
      double *X = malloc(n * p * sizeof(double)), *X0 = malloc(n * p * sizeof(double));
      double *V = malloc(p * p * sizeof(double)), *sv = malloc(p * sizeof(double));
      double *sref = malloc(p * sizeof(double));
      double sverr = 0.0, orth = 0.0, rec = 0.0, xnorm = 0.0;
      size_t i, j, k;
      int status;

      /* graded singular values 1 .. 1e-15 (and exact zeros beyond the rank) */
      for (k = 0; k < p; ++k)
        sref[k] = (k < r) ? pow(10.0, -15.0 * k / (double) (r > 1 ? r - 1 : 1)) : 0.0;
      if (ic == 2)
        sref[5] = sref[6] = 0.0;   /* rank deficient: must still converge */

      random_orthogonal(n, U);
      random_orthogonal(p, W);
      for (j = 0; j < p; ++j)
        for (i = 0; i < n; ++i)
          {
            long double sum = 0.0L;
            for (k = 0; k < r; ++k)
              sum += (long double) U[k * n + i] * sref[k] * W[k * p + j];
            X[j * n + i] = X0[j * n + i] = (double) sum;
            xnorm = fmax(xnorm, fabs((double) sum));
          }

      status = ps_svd_jacobi(n, p, X, V, sv);
      CHECK(status == POTSYNTH_SUCCESS, "svd did not converge (%zu x %zu)", n, p);

      for (k = 0; k < p; ++k)
        {
          CHECK(k == 0 || sv[k] <= sv[k - 1], "svd not sorted");
          /* one-sided Jacobi: absolute accuracy ~ eps * s_1 */
          sverr = fmax(sverr, fabs(sv[k] - sref[k]));
        }

      /* V orthogonal */
      for (i = 0; i < p; ++i)
        for (j = 0; j < p; ++j)
          {
            double d = 0.0;
            for (k = 0; k < p; ++k)
              d += V[i * p + k] * V[j * p + k];
            orth = fmax(orth, fabs(d - (i == j ? 1.0 : 0.0)));
          }

      /* X0 = X V^T (X now holds U*S) */
      for (j = 0; j < p; ++j)
        for (i = 0; i < n; ++i)
          {
            double d = 0.0;
            for (k = 0; k < p; ++k)
              d += X[k * n + i] * V[k * p + j];
            rec = fmax(rec, fabs(d - X0[j * n + i]));
          }

      CHECK(sverr < 1e-14 && orth < 1e-13 && rec < 1e-14 * xnorm * p,
            "svd %zux%zu: sv err %.1e, orth %.1e, rec %.1e", n, p, sverr, orth, rec);
      printf("  svd %3zu x %2zu: sv err %.1e  orth %.1e  rec %.1e %s\n", n, p, sverr, orth,
             rec, (sverr < 1e-14 && orth < 1e-13 && rec < 1e-14 * xnorm * p) ? "ok" : "FAIL");

      free(U); free(W); free(X); free(X0); free(V); free(sv); free(sref);
    }
}

/* ------------------------------------------------------------------------- */

static double
radial_eval(const ps_radial *rad, size_t i, double u)
{
  const double t = (rad->ud > 0.0) ? (u - rad->uc) / rad->ud : 0.0;
  double T0 = 1.0, T1 = t, sum = rad->basis[i * rad->ncheb];
  size_t k;

  if (rad->ncheb > 1)
    sum += rad->basis[i * rad->ncheb + 1] * T1;
  for (k = 2; k < rad->ncheb; ++k)
    {
      const double T2 = 2.0 * t * T1 - T0;
      sum += rad->basis[i * rad->ncheb + k] * T2;
      T0 = T1;
      T1 = T2;
    }

  return sum;
}

/* weighted approximation error of u^(n+rpow) over a fine grid, relative to the
 * largest weighted function value */
static double
radial_error(const ps_radial *rad, double u0, double u1, size_t lmin, size_t lmax,
             size_t nchan, const int *rpow, const double *wn)
{
  const size_t L1 = lmax + 1, ng = 500;
  double emax = 0.0, fmaxv = 0.0;
  size_t c, n, g, i;

  for (c = 0; c < nchan; ++c)
    for (n = lmin; n <= lmax; ++n)
      for (g = 0; g <= ng; ++g)
        {
          const double u = u0 + (u1 - u0) * g / (double) ng;
          const double f = pow(u, (double) n + rpow[c]);
          double approx = 0.0;

          for (i = 0; i < rad->rho; ++i)
            approx += rad->beta[(c * rad->rho + i) * L1 + n] * radial_eval(rad, i, u);

          emax = fmax(emax, wn[c * L1 + n] * fabs(approx - f));
          fmaxv = fmax(fmaxv, wn[c * L1 + n] * fabs(f));
        }

  return emax / fmaxv;
}

static void
test_radial(void)
{
  const double a = 6371.2;
  const struct { double rmin, rmax; size_t lmin, lmax; } cases[] =
    { { a + 250, a + 800, 1, 185 }, { a, 2 * a, 1, 185 }, { a, a + 1000, 0, 13 },
      { a + 450, a + 450, 1, 60 } };
  const double tols[] = { 1e-6, 1e-10, 1e-14 };
  size_t ic, it, n;

  for (ic = 0; ic < sizeof(cases) / sizeof(cases[0]); ++ic)
    {
      const size_t L = cases[ic].lmax, L1 = L + 1;
      const double u0 = a / cases[ic].rmax, u1 = a / cases[ic].rmin;
      const int rpow[2] = { 2, 1 };
      double *wn = malloc(2 * L1 * sizeof(double));
      size_t lastrho = 0;

      for (n = 0; n <= L; ++n)
        {
          wn[n] = (n + 1.0) * pow(0.8, (double) n);       /* channel 0 */
          wn[L1 + n] = pow(0.8, (double) n);              /* channel 1 */
        }

      for (it = 0; it < sizeof(tols) / sizeof(tols[0]); ++it)
        {
          ps_radial rad;
          int status = ps_radial_init(&rad, u0, u1, cases[ic].lmin, L, 2, rpow, wn, tols[it]);
          double err;

          CHECK(status == POTSYNTH_SUCCESS, "radial_init case %zu tol %.0e", ic, tols[it]);
          if (status)
            continue;

          err = radial_error(&rad, u0, u1, cases[ic].lmin, L, 2, rpow, wn);
          CHECK(err < 20.0 * tols[it] + 1e-15, "radial case %zu tol %.0e: err %.2e", ic,
                tols[it], err);
          CHECK(rad.rho >= lastrho, "radial rank not monotone in tol");
          printf("  radial case %zu tol %.0e: rho %2zu ncheb %2zu err %.1e %s\n", ic, tols[it],
                 rad.rho, rad.ncheb, err, (err < 20.0 * tols[it] + 1e-15) ? "ok" : "FAIL");
          lastrho = rad.rho;
          ps_radial_free(&rad);
        }

      free(wn);
    }
}

/* ------------------------------------------------------------------------- */

static void
test_accum(void)
{
  size_t trial;
  double maxerr = 0.0;

  for (trial = 0; trial < 200; ++trial)
    {
      const size_t nn = 1 + (size_t) (urand() * 60), npair = 1 + (size_t) (urand() * 5);
      const size_t ldw = nn + 3;
      double *W = malloc(2 * npair * ldw * sizeof(double));
      double *Q = malloc(nn * PS_NB * sizeof(double)), *E = malloc(nn * PS_NB * sizeof(double));
      double *acc = malloc(4 * npair * PS_NB * sizeof(double));
      unsigned char *src = malloc(npair);
      size_t *n0 = malloc(npair * sizeof(size_t));
      size_t i, k, j, n;

      for (i = 0; i < 2 * npair * ldw; ++i)
        W[i] = 2.0 * urand() - 1.0;
      for (i = 0; i < nn * PS_NB; ++i)
        {
          Q[i] = 2.0 * urand() - 1.0;
          E[i] = 2.0 * urand() - 1.0;
        }
      for (k = 0; k < npair; ++k)
        {
          src[k] = urand() < 0.5;
          n0[k] = (size_t) (urand() * (nn + 1));     /* includes n0 = nn (empty) */
        }

      ps_alf_accum(nn, npair, W, ldw, src, n0, Q, E, acc);

      for (k = 0; k < npair; ++k)
        for (j = 0; j < PS_NB; ++j)
          {
            long double ref[4] = { 0, 0, 0, 0 };
            const double *V = src[k] ? E : Q;

            for (n = n0[k]; n < nn; ++n)
              {
                ref[0 + (n & 1)] += (long double) W[2 * k * ldw + n] * V[n * PS_NB + j];
                ref[2 + (n & 1)] += (long double) W[(2 * k + 1) * ldw + n] * V[n * PS_NB + j];
              }

            for (i = 0; i < 4; ++i)
              maxerr = fmax(maxerr, (double) fabsl(acc[(4 * k + i) * PS_NB + j] - ref[i]));
          }

      free(W); free(Q); free(E); free(acc); free(src); free(n0);
    }

  CHECK(maxerr < 1e-13, "alf_accum err %.2e", maxerr);
  printf("  alf_accum: max err %.1e %s\n", maxerr, (maxerr < 1e-13) ? "ok" : "FAIL");
}

/* ------------------------------------------------------------------------- */

/* long-double 4-pi P(n,m), dP/dtheta and P/sin by the standard unscaled
 * recursions (fine for L <= 185: no underflow in long double) */
static void
alf_ref(size_t L, long double th, long double *P, long double *D, long double *PS)
{
  const long double x = cosl(th), s = sinl(th);
  long double pmm = 1.0L, dmm = 0.0L, qmm = 0.0L;   /* P(m,m), dP(m,m), P(m,m)/s */
  size_t m, n;

  for (m = 0; m <= L; ++m)
    {
      long double p1, p2 = 0.0L, d1, d2 = 0.0L, q1, q2 = 0.0L;

      if (m == 1)
        {
          pmm = sqrtl(3.0L) * s;
          dmm = sqrtl(3.0L) * x;
          qmm = sqrtl(3.0L);
        }
      else if (m >= 2)
        {
          const long double f = sqrtl(1.0L + 0.5L / m);
          dmm = f * (s * dmm + x * pmm);
          pmm *= f * s;
          qmm *= f * s;
        }

      p1 = pmm;
      d1 = dmm;
      q1 = qmm;
      for (n = m; n <= L; ++n)
        {
          const size_t idx = ps_alf_idx(n, m, L);

          if (n > m)
            {
              long double p, d, q;

              if (n == m + 1)
                {
                  const long double c = sqrtl(2.0L * m + 3.0L);
                  p = c * x * p1;
                  d = c * (x * d1 - s * p1);
                  q = c * x * q1;
                }
              else
                {
                  const long double a = sqrtl((2.0L * n + 1) * (2.0L * n - 1) / ((long double) (n + m) * (n - m)));
                  const long double b = -sqrtl((long double) (n + m - 1) * (n - m - 1) * (2.0L * n + 1) /
                                               ((long double) (n + m) * (n - m) * (2.0L * n - 3)));
                  p = a * x * p1 + b * p2;
                  d = a * (x * d1 - s * p1) + b * d2;
                  q = a * x * q1 + b * q2;
                }

              p2 = p1; d2 = d1; q2 = q1;
              p1 = p; d1 = d; q1 = q;
            }

          P[idx] = p1;
          D[idx] = d1;
          PS[idx] = q1;
        }
    }
}

static void
test_alf_columns(void)
{
  const size_t L = 185, nlm = (L + 1) * (L + 2) / 2;
  const double ths[] = { 0.0, 1e-300, 1e-8, 0.05, 0.3, 1.2, M_PI / 2, 2.5, M_PI - 1e-6, M_PI };
  long double *P = malloc(nlm * sizeof(long double)), *D = malloc(nlm * sizeof(long double));
  long double *PS = malloc(nlm * sizeof(long double));
  double *Q = malloc((L + 1) * PS_NB * sizeof(double)), *E = malloc((L + 1) * PS_NB * sizeof(double));
  double maxerr = 0.0;
  ps_alf alf;
  size_t it, j, m, n;

  CHECK(ps_alf_init(&alf, L) == POTSYNTH_SUCCESS, "alf_init");

  for (it = 0; it < sizeof(ths) / sizeof(ths[0]); ++it)
    {
      /* errors are measured relative to the largest value of each quantity
       * (P/sin reaches ~L^1.5 at the poles, far above max |P| ~ sqrt(2L+1)) */
      double x[PS_NB], s[PS_NB], scP = 1.0, scD = 1.0, scS = 1.0;

      alf_ref(L, (long double) ths[it], P, D, PS);
      for (j = 0; j < PS_NB; ++j)
        {
          x[j] = cos(ths[it]);
          s[j] = sin(ths[it]);
        }

      for (m = 0; m <= L; ++m)
        for (n = m; n <= L; ++n)
          {
            const size_t idx = ps_alf_idx(n, m, L);
            scP = fmax(scP, (double) fabsl(P[idx]));
            scD = fmax(scD, (double) fabsl(D[idx]));
            if (m > 0)
              scS = fmax(scS, (double) fabsl(PS[idx]));
          }

      for (m = 0; m <= L; ++m)
        {
          ps_alf_column(&alf, m, x, s, Q, E);

          for (n = m; n <= L; ++n)
            {
              const size_t idx = ps_alf_idx(n, m, L);
              double p, d, ps = 0.0, mant;
              int expo;

              if (m == 0)
                {
                  p = Q[(n - m) * PS_NB];
                  d = E[(n - m) * PS_NB];
                }
              else
                {
                  ps_pow_scaled(s[0], m, &mant, &expo);
                  p = ldexp(Q[(n - m) * PS_NB] * mant, expo + alf.scale[m]);
                  ps_pow_scaled(s[0], m - 1, &mant, &expo);
                  ps = ldexp(Q[(n - m) * PS_NB] * mant, expo + alf.scale[m]);
                  d = ldexp(E[(n - m) * PS_NB] * mant, expo + alf.scale[m]);
                  maxerr = fmax(maxerr, (double) fabsl(ps - PS[idx]) / scS);
                }

              CHECK(isfinite(p) && isfinite(d) && isfinite(ps), "non-finite ALF n=%zu m=%zu", n, m);
              maxerr = fmax(maxerr, (double) fabsl(p - P[idx]) / scP);
              maxerr = fmax(maxerr, (double) fabsl(d - D[idx]) / scD);
            }
        }
    }

  /* analytic identities at the north pole: P(n,0) = sqrt(2n+1), P(n,1)/sin = sqrt((2n+1) n (n+1) / 2) */
  {
    double x[PS_NB], s[PS_NB], perr = 0.0;

    for (j = 0; j < PS_NB; ++j)
      {
        x[j] = 1.0;
        s[j] = 0.0;
      }
    ps_alf_column(&alf, 0, x, s, Q, E);
    for (n = 0; n <= L; ++n)
      perr = fmax(perr, fabs(Q[n * PS_NB] - sqrt(2.0 * n + 1.0)) / sqrt(2.0 * n + 1.0));
    ps_alf_column(&alf, 1, x, s, Q, E);
    for (n = 1; n <= L; ++n)
      {
        const double ref = sqrt((2.0 * n + 1.0) * n * (n + 1.0) / 2.0);
        perr = fmax(perr, fabs(ldexp(Q[(n - 1) * PS_NB], alf.scale[1]) - ref) / ref);
      }
    CHECK(perr < 5e-13, "pole identities %.2e", perr);
    printf("  alf pole identities: max rel err %.1e %s\n", perr, (perr < 5e-13) ? "ok" : "FAIL");
  }

  /* near the poles the recursion error grows like eps * L^2 */
  CHECK(maxerr < 2e-12, "alf columns vs long double %.2e", maxerr);
  printf("  alf columns (L=%zu) vs long double: max err %.1e %s\n", L, maxerr,
         (maxerr < 2e-12) ? "ok" : "FAIL");

  /* scaling: none needed below L ~ 1370; supported up to POTSYNTH_LMAX */
  {
    ps_alf a2;
    int okscale = 1, anyscale = 0;

    CHECK(ps_alf_init(&a2, 1300) == POTSYNTH_SUCCESS, "alf_init 1300");
    for (m = 0; m <= 1300; ++m)
      okscale = okscale && a2.scale[m] == 0;
    ps_alf_free(&a2);
    CHECK(okscale, "no scaling expected at L=1300");
    CHECK(ps_alf_init(&a2, 2190) == POTSYNTH_SUCCESS, "alf_init 2190");
    for (m = 0; m <= 2190; ++m)
      anyscale = anyscale || a2.scale[m] > 0;
    ps_alf_free(&a2);
    CHECK(anyscale, "scaling expected at L=2190");
    CHECK(ps_alf_init(&a2, POTSYNTH_LMAX) == POTSYNTH_SUCCESS, "alf_init LMAX");
    ps_alf_free(&a2);
    CHECK(ps_alf_init(&a2, POTSYNTH_LMAX + 100) == POTSYNTH_EUNSUP, "alf_init beyond LMAX");
  }

  ps_alf_free(&alf);
  free(P); free(D); free(PS); free(Q); free(E);
}

/* ------------------------------------------------------------------------- */

/* the FFT conventions the set-up relies on (setup.c) */
static void
test_fft_conventions(void)
{
  const size_t ns[] = { 12, 30, 64, 375, 750 };
  double maxerr = 0.0;
  size_t in;

  for (in = 0; in < sizeof(ns) / sizeof(ns[0]); ++in)
    {
      const size_t N = ns[in];
      double *c = malloc(2 * N * sizeof(double)), *c0 = malloc(2 * N * sizeof(double));
      double *r = malloc(N * sizeof(double)), *r0 = malloc(N * sizeof(double));
      cfft_plan cp = make_cfft_plan(N);
      rfft_plan rp = make_rfft_plan(N);
      size_t j, k;

      for (j = 0; j < 2 * N; ++j)
        c[j] = c0[j] = 2.0 * urand() - 1.0;

      /* forward: F_k = sum_j c_j e^{-2 pi i jk/N} */
      cfft_forward(cp, c, 1.0);
      for (k = 0; k < N; ++k)
        {
          long double re = 0.0L, im = 0.0L;
          for (j = 0; j < N; ++j)
            {
              const long double a = -2.0L * M_PI * (long double) ((j * k) % N) / N;
              re += c0[2 * j] * cosl(a) - c0[2 * j + 1] * sinl(a);
              im += c0[2 * j] * sinl(a) + c0[2 * j + 1] * cosl(a);
            }
          maxerr = fmax(maxerr, (double) fmaxl(fabsl(c[2 * k] - re), fabsl(c[2 * k + 1] - im)) / N);
        }

      /* backward: c_j = sum_k F_k e^{+2 pi i jk/N}, so backward(forward(c)) = N c */
      cfft_backward(cp, c, 1.0 / N);
      for (j = 0; j < 2 * N; ++j)
        maxerr = fmax(maxerr, fabs(c[j] - c0[j]));

      /* real backward, half-complex input r0, r1, i1, r2, i2, ...:
       * x_k = r0 + 2 sum_m (r_m cos - i_m sin)(2 pi mk/N) + r_{N/2} (-1)^k */
      for (j = 0; j < N; ++j)
        r[j] = r0[j] = 2.0 * urand() - 1.0;
      rfft_backward(rp, r, 1.0);
      for (k = 0; k < N; ++k)
        {
          long double sum = r0[0];
          size_t m;

          for (m = 1; 2 * m < N; ++m)
            {
              const long double a = 2.0L * M_PI * (long double) ((m * k) % N) / N;
              sum += 2.0L * (r0[2 * m - 1] * cosl(a) - r0[2 * m] * sinl(a));
            }
          if (N % 2 == 0)
            sum += r0[N - 1] * ((k & 1) ? -1.0L : 1.0L);
          maxerr = fmax(maxerr, (double) fabsl(r[k] - sum) / N);
        }

      destroy_cfft_plan(cp);
      destroy_rfft_plan(rp);
      free(c); free(c0); free(r); free(r0);
    }

  CHECK(maxerr < 1e-14, "FFT conventions %.2e", maxerr);
  printf("  fft conventions: max err %.1e %s\n", maxerr, (maxerr < 1e-14) ? "ok" : "FAIL");
}

int
main(void)
{
  printf("internal unit tests:\n");
  test_next_smooth();
  test_pow_scaled();
  test_sincos_m();
  test_gauss_legendre();
  test_window_ft();
  test_svd();
  test_radial();
  test_accum();
  test_alf_columns();
  test_fft_conventions();

  printf("%s\n", nfail ? "FAILED" : "all passed");
  return nfail != 0;
}
