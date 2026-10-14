/*
 * test_alf.c -- compare the potsynth Legendre column recursion with
 * gsl_sf_alf_vsh_array() (optional GSL oracle)
 *
 * Errors are measured relative to the largest value over all (l,m) (columns lying
 * entirely below ~1e-300 are irrelevant), with a tolerance proportional to
 * eps*L^2: near the poles the three-term
 * recursion accumulates rounding errors at that rate, in both GSL and potsynth
 * (checked separately against 60-digit references).
 *
 * At high degree GSL underflows to exactly zero where potsynth's scaled columns
 * still hold the correct value (Holmes-Featherstone region); those entries are
 * counted but not treated as failures.
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <gsl/gsl_sf_alf.h>

#include "potsynth_internal.h"

static int nfail = 0;

static double
unscale(double v, double s, size_t p, int scale)
{
  double mant;
  int expo;

  ps_pow_scaled(s, p, &mant, &expo);
  return ldexp(v * mant, expo + scale);
}

static void
check(size_t lmax, double theta)
{
  const double x = cos(theta);
  const double tol = 1e-14 + 5e-17 * (lmax + 1.0) * (lmax + 1.0);
  const size_t asize = gsl_sf_alf_array_size(lmax, lmax);
  double *gP = malloc(asize * sizeof(double));
  double *gdP = malloc(asize * sizeof(double));
  double *Q = malloc((lmax + 1) * PS_NB * sizeof(double));
  double *E = malloc((lmax + 1) * PS_NB * sizeof(double));
  double *P = malloc((lmax + 1) * sizeof(double));
  double *D = malloc((lmax + 1) * sizeof(double));
  double xv[PS_NB], sv[PS_NB];
  double maxerr = 0.0;
  size_t nunder = 0, nbad = 0;
  ps_alf alf;
  size_t j, l, m;
  int status;

  gsl_sf_alf_precompute(GSL_SF_ALF_FOURPI, lmax, lmax, 0, gP);
  gsl_sf_alf_vsh_array(lmax, lmax, x, gP, gdP);

  status = ps_alf_init(&alf, lmax);
  if (status != POTSYNTH_SUCCESS)
    {
      printf("FAIL ps_alf_init lmax=%zu: %s\n", lmax, potsynth_strerror(status));
      ++nfail;
      free(gP); free(gdP); free(Q); free(E); free(P); free(D);
      return;
    }

  /* GSL uses u = sqrt(1-x^2) >= 0; for theta in [0,pi] this equals sin(theta) */
  for (j = 0; j < PS_NB; ++j)
    {
      xv[j] = x;
      sv[j] = sqrt((1.0 - x) * (1.0 + x));
    }

  /* global magnitudes (GSL values are reliable where they are large) */
  {
    double pmax = 1.0, dmax = 1.0;
    double perr = 0.0, derr = 0.0;

    for (m = 0; m <= lmax; ++m)
      for (l = m; l <= lmax; ++l)
        {
          const size_t gi = gsl_sf_alf_array_index(l, m, lmax);
          pmax = fmax(pmax, fabs(gP[gi]));
          dmax = fmax(dmax, fabs(gdP[gi]));
        }

    for (m = 0; m <= lmax; ++m)
      {
        /* GSL's diagonal seed P(m,m) is subnormal (precision lost) or zero:
         * the whole GSL column is unreliable there (verified against mpmath) */
        const int gsl_lost = fabs(gP[gsl_sf_alf_array_index(m, m, lmax)]) < 1e-290;

        ps_alf_column(&alf, m, xv, sv, Q, E);

        for (l = m; l <= lmax; ++l)
          {
            const size_t gi = gsl_sf_alf_array_index(l, m, lmax);

            P[l] = Q[(l - m) * PS_NB];
            D[l] = E[(l - m) * PS_NB];
            if (m > 0)
              {
                P[l] = unscale(P[l], sv[0], m - 1, alf.scale[m]);
                D[l] = unscale(D[l], sv[0], m - 1, alf.scale[m]);
              }

            if (!isfinite(P[l]) || !isfinite(D[l]))
              ++nbad;

            if (gsl_lost)
              {
                ++nunder;         /* GSL underflow */
                continue;
              }

            perr = fmax(perr, fabs(P[l] - gP[gi]));
            derr = fmax(derr, fabs(D[l] - gdP[gi]));
          }
      }

    maxerr = fmax(perr / pmax, derr / dmax);
  }

  printf("lmax=%5zu theta=%-10.4g max err %.2e (tol %.1e)  gsl-underflow %7zu  %s\n",
         lmax, theta, maxerr, tol, nunder, (maxerr <= tol && nbad == 0) ? "ok" : "FAIL");
  if (!(maxerr <= tol) || nbad)
    ++nfail;

  ps_alf_free(&alf);
  free(gP); free(gdP); free(Q); free(E); free(P); free(D);
}

int
main(void)
{
  const size_t lmaxs[] = { 0, 1, 2, 13, 185, 720, 2190, 2700 };
  const double thetas[] = { 0.0, 1e-300, 1e-8, 1e-3, 0.01, 0.4, M_PI / 2, 2.0, M_PI - 1e-6, M_PI };
  size_t i, j;

  for (i = 0; i < sizeof(lmaxs) / sizeof(lmaxs[0]); ++i)
    for (j = 0; j < sizeof(thetas) / sizeof(thetas[0]); ++j)
      check(lmaxs[i], thetas[j]);

  printf("%s\n", nfail ? "FAILED" : "all passed");
  return nfail != 0;
}
