/*
 * test_alf_gsl.c
 *
 * Legendre columns against gsl_sf_alf_vsh_array() (an independent
 * implementation), up to L = 2700. Errors are relative to the largest value
 * over all (l,m), with a tolerance proportional to eps*L^2: near the poles the
 * three-term recursion accumulates rounding errors at that rate, in both GSL
 * and potsynth (checked against 60-digit references).
 *
 * GSL does not rescale the diagonal seed P(m,m): where it becomes subnormal or
 * zero (high degree, theta not near the equator), whole GSL columns are wrong
 * (verified against mpmath); those columns are skipped.
 */

#ifdef HAVE_GSL

static void
test_alf_gsl_case(const size_t lmax, const double theta)
{
  const double x = cos(theta);
  const double tol = 1.0e-14 + 5.0e-17 * (lmax + 1.0) * (lmax + 1.0);
  const size_t asize = gsl_sf_alf_array_size(lmax, lmax);
  double * gP = malloc(asize * sizeof(double));
  double * gdP = malloc(asize * sizeof(double));
  double * Q = malloc((lmax + 1) * POTSYNTH_NB * sizeof(double));
  double * E = malloc((lmax + 1) * POTSYNTH_NB * sizeof(double));
  potsynth_alf_workspace * w = potsynth_alf_alloc(lmax);
  double xv[POTSYNTH_NB], sv[POTSYNTH_NB];
  double pmax = 1.0, dmax = 1.0, perr = 0.0, derr = 0.0;
  size_t j, l, m;
  int nbad = 0;

  gsl_sf_alf_precompute(GSL_SF_ALF_FOURPI, lmax, lmax, 0, gP);
  gsl_sf_alf_vsh_array(lmax, lmax, x, gP, gdP);

  /* GSL uses u = sqrt(1-x^2) >= 0; for theta in [0,pi] this equals sin(theta) */
  for (j = 0; j < POTSYNTH_NB; ++j)
    {
      xv[j] = x;
      sv[j] = sqrt((1.0 - x) * (1.0 + x));
    }

  for (m = 0; m <= lmax; ++m)
    {
      for (l = m; l <= lmax; ++l)
        {
          const size_t gi = gsl_sf_alf_array_index(l, m, lmax);

          pmax = fmax(pmax, fabs(gP[gi]));
          dmax = fmax(dmax, fabs(gdP[gi]));
        }
    }

  for (m = 0; m <= lmax; ++m)
    {
      /* GSL's diagonal seed is subnormal or zero: column unreliable */
      const int gsl_lost = fabs(gP[gsl_sf_alf_array_index(m, m, lmax)]) < 1.0e-290;

      potsynth_alf_column(m, xv, sv, Q, E, NULL, w);

      for (l = m; l <= lmax; ++l)
        {
          const size_t gi = gsl_sf_alf_array_index(l, m, lmax);
          double p = Q[(l - m) * POTSYNTH_NB];
          double d = E[(l - m) * POTSYNTH_NB];

          /* GSL returns P/sin for m > 0 */
          if (m > 0)
            {
              p = test_alf_unscale(p, sv[0], m - 1, w->scale[m]);
              d = test_alf_unscale(d, sv[0], m - 1, w->scale[m]);
            }

          if (!isfinite(p) || !isfinite(d))
            ++nbad;

          if (gsl_lost)
            continue;

          perr = fmax(perr, fabs(p - gP[gi]));
          derr = fmax(derr, fabs(d - gdP[gi]));
        }
    }

  ps_test(nbad, "alf_gsl: L=%zu theta=%g finite", lmax, theta);
  ps_test(fmax(perr / pmax, derr / dmax) > tol, "alf_gsl: L=%zu theta=%g (error %.2e, tol %.1e)",
          lmax, theta, fmax(perr / pmax, derr / dmax), tol);

  potsynth_alf_free(w);
  free(gP);
  free(gdP);
  free(Q);
  free(E);
}

static void
test_alf_gsl(void)
{
  const size_t lmaxs[] = { 0, 1, 2, 13, 185, 720, 2190, 2700 };
  const double thetas[] = { 0.0, 1.0e-300, 1.0e-8, 1.0e-3, 0.01, 0.4, M_PI / 2.0, 2.0,
                            M_PI - 1.0e-6, M_PI };
  size_t i, j;

  for (i = 0; i < sizeof(lmaxs) / sizeof(lmaxs[0]); ++i)
    {
      for (j = 0; j < sizeof(thetas) / sizeof(thetas[0]); ++j)
        test_alf_gsl_case(lmaxs[i], thetas[j]);
    }
}

#else

static void
test_alf_gsl(void)
{
}

#endif /* HAVE_GSL */
