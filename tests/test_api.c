/*
 * test_api.c
 *
 * API contract: reproducibility, NaN inputs, domain errors, invalid arguments
 */

static void
test_api(void)
{
  const double a = 6371.2;
  const double rmin = a + 250.0;
  const double rmax = a + 800.0;
  const size_t L = 40;
  const size_t n = 5000;
  double * gh = test_random_coef(1, L, L);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * B1 = malloc(3 * n * sizeof(double));
  double * B2 = malloc(3 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w, * w1;
  potsynth_eval_workspace * ew, * ew1, * ewsmall;
  potsynth_info info;
  double x, y, z, out[3];
  size_t i;

  test_random_points(n, rmin, rmax, r, theta, phi);

  params.method = POTSYNTH_METHOD_FAST;
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params);
  ew = potsynth_eval_alloc(0, w);
  ewsmall = potsynth_eval_alloc(777, w);

  /* chunk size does not change results (bitwise) */
  ps_test_int(potsynth_eval(n, r, theta, phi, B1, ew), POTSYNTH_SUCCESS, "api: evaluation");
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ewsmall), POTSYNTH_SUCCESS, "api: chunked evaluation");
  ps_test(memcmp(B1, B2, 3 * n * sizeof(double)) != 0, "api: chunked results bitwise equal");

  /* thread count does not change results (bitwise) */
  params.nthreads = 1;
  w1 = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params);
  ew1 = potsynth_eval_alloc(0, w1);
  potsynth_eval(n, r, theta, phi, B2, ew1);
  ps_test(memcmp(B1, B2, 3 * n * sizeof(double)) != 0, "api: 1-thread results bitwise equal");
  potsynth_eval_free(ew1);
  potsynth_free(w1);

  /* longitude periodicity */
  for (i = 0; i < n; ++i)
    phi[i] += 2.0 * M_PI * 1000.0;

  potsynth_eval(n, r, theta, phi, B2, ew);
  {
    double emax = 0.0, bmax = 0.0;

    for (i = 0; i < 3 * n; ++i)
      {
        emax = fmax(emax, fabs(B2[i] - B1[i]));
        bmax = fmax(bmax, fabs(B1[i]));
      }

    ps_test(emax / bmax > 1.0e-9, "api: phi + 2000 pi (difference %.2e)", emax / bmax);
  }

  for (i = 0; i < n; ++i)
    phi[i] -= 2.0 * M_PI * 1000.0;

  /* NaN inputs give NaN outputs for that point only */
  r[5] = NAN;
  theta[7] = INFINITY;
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ew), POTSYNTH_SUCCESS, "api: NaN input is not an error");
  for (i = 0; i < 3; ++i)
    {
      ps_test(!isnan(B2[i * n + 5]) || !isnan(B2[i * n + 7]), "api: NaN propagated, component %zu", i);
      ps_test_rel(B2[i * n + 6], B1[i * n + 6], 0.0, "api: neighbor of NaN point, component %zu", i);
    }
  r[5] = rmin;
  theta[7] = 1.0;

  /* domain errors leave the output untouched */
  for (i = 0; i < 3 * n; ++i)
    B2[i] = -12345.0;

  theta[100] = -1.0e-9;
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ew), POTSYNTH_EDOM, "api: theta < 0");
  theta[100] = M_PI + 1.0e-12;
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ew), POTSYNTH_EDOM, "api: theta > pi");
  theta[100] = 1.0;
  r[100] = rmax * (1.0 + 1.0e-6);
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ew), POTSYNTH_EDOM, "api: r > rmax");
  r[100] = 0.0;
  ps_test_int(potsynth_eval(n, r, theta, phi, B2, ew), POTSYNTH_EDOM, "api: r = 0");
  r[100] = rmin;
  ps_test(B2[0] != -12345.0 || B2[3 * n - 1] != -12345.0, "api: output untouched on error");
  ps_test_int(test_last_error(), POTSYNTH_EDOM, "api: domain error reported to handler");

  /* empty input, NULL arrays */
  ps_test_int(potsynth_eval(0, NULL, NULL, NULL, NULL, ew), POTSYNTH_SUCCESS, "api: n = 0");
  ps_test_int(potsynth_eval(1, r, NULL, phi, B2, ew), POTSYNTH_EINVAL, "api: NULL theta");

  /* METHOD_AUTO uses the direct method at low degree; direct accepts any r > 0 */
  params = potsynth_default_parameters();
  w1 = potsynth_mag_alloc(1, 13, 13, gh, a, 0.0, 0.0, POTSYNTH_MAG_B, &params);
  ps_test(w1 == NULL, "api: METHOD_AUTO allocation at L=13");
  if (w1)
    {
      potsynth_get_info(&info, w1);
      ps_test_int(info.method, POTSYNTH_METHOD_DIRECT, "api: METHOD_AUTO chooses direct at L=13");

      ew1 = potsynth_eval_alloc(0, w1);
      x = 10.0 * a;
      y = 0.0;
      z = 0.0;
      ps_test_int(potsynth_eval(1, &x, &y, &z, out, ew1), POTSYNTH_SUCCESS, "api: direct at pole, r = 10a");
      ps_test(!isfinite(out[0]) || !isfinite(out[2]), "api: direct at pole finite");

      potsynth_eval_free(ew1);
      potsynth_free(w1);
    }

  /* METHOD_AUTO with npts_hint: total cost of set-up + evaluation decides */
  {
    const size_t L60 = 60;
    double * gh60 = test_random_coef(1, L60, L60);
    const size_t hints[] = { 1000, 10000000 };
    const int expect60[] = { POTSYNTH_METHOD_DIRECT, POTSYNTH_METHOD_FAST };
    size_t k;

    for (k = 0; k < 2; ++k)
      {
        params = potsynth_default_parameters();
        params.npts_hint = hints[k];
        w1 = potsynth_mag_alloc(1, L60, L60, gh60, a, rmin, rmax, POTSYNTH_MAG_B, &params);
        potsynth_get_info(&info, w1);
        ps_test_int(info.method, expect60[k], "api: METHOD_AUTO L=60 npts_hint=%zu", hints[k]);
        potsynth_free(w1);
      }

    /* at L = 13 the direct sum is cheaper per point: never the fast method */
    params = potsynth_default_parameters();
    params.npts_hint = 100000000;
    w1 = potsynth_mag_alloc(1, 13, 13, gh60, a, rmin, rmax, POTSYNTH_MAG_B, &params);
    potsynth_get_info(&info, w1);
    ps_test_int(info.method, POTSYNTH_METHOD_DIRECT, "api: METHOD_AUTO L=13 npts_hint=1e8");
    potsynth_free(w1);

    free(gh60);
  }

  /* METHOD_AUTO without a radial range uses the direct method at any degree */
  {
    double * gh60 = test_random_coef(1, 60, 60);

    params = potsynth_default_parameters();
    w1 = potsynth_mag_alloc(1, 60, 60, gh60, a, 0.0, 0.0, POTSYNTH_MAG_B, &params);
    ps_test(w1 == NULL, "api: METHOD_AUTO without radial range at L=60");
    if (w1)
      {
        potsynth_get_info(&info, w1);
        ps_test_int(info.method, POTSYNTH_METHOD_DIRECT, "api: METHOD_AUTO without radial range chooses direct");
        potsynth_free(w1);
      }

    free(gh60);
  }

  /* invalid arguments */
  params = potsynth_default_parameters();
  ps_test(potsynth_mag_alloc(0, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params) != NULL, "api: lmin = 0 rejected by mag");
  ps_test_int(test_last_error(), POTSYNTH_EINVAL, "api: lmin = 0 error code");
  ps_test(potsynth_mag_alloc(1, L, L + 1, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params) != NULL, "api: mmax > lmax rejected");
  params.method = POTSYNTH_METHOD_FAST;
  ps_test(potsynth_mag_alloc(1, L, L, gh, a, rmax, rmin, POTSYNTH_MAG_B, &params) != NULL, "api: rmin > rmax rejected");
  ps_test(potsynth_mag_alloc(1, 3000, 3000, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params) != NULL, "api: lmax = 3000 rejected");
  ps_test_int(test_last_error(), POTSYNTH_EUNSUP, "api: lmax = 3000 error code");
  params.eps = 0.0;
  ps_test(potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params) != NULL, "api: eps = 0 rejected");
  test_last_error();

  /* NEC conversion */
  out[0] = 1.0;
  out[1] = 2.0;
  out[2] = 3.0;
  potsynth_mag_rtp2nec(1, out);
  ps_test(out[0] != -2.0 || out[1] != 3.0 || out[2] != -1.0, "api: rtp2nec");

  /* index macro is contiguous and matches potsynth_nlm() */
  {
    long lmin, lmax = 12, mmax, l, m;
    int nfail = 0;

    for (lmin = 0; lmin <= lmax; ++lmin)
      {
        for (mmax = 0; mmax <= lmax; ++mmax)
          {
            size_t expect = 0;

            for (l = lmin; l <= lmax; ++l)
              {
                const long M = (l < mmax) ? l : mmax;

                for (m = -M; m <= M; ++m, ++expect)
                  {
                    if ((size_t) potsynth_idx(l, m, lmin, mmax) != expect)
                      ++nfail;
                  }
              }

            if (expect != potsynth_nlm((size_t) lmin, (size_t) lmax, (size_t) mmax))
              ++nfail;
          }
      }

    ps_test(nfail, "api: potsynth_idx / potsynth_nlm (%d failures)", nfail);
  }

  potsynth_eval_free(ew);
  potsynth_eval_free(ewsmall);
  potsynth_free(w);
  free(gh);
  free(r);
  free(theta);
  free(phi);
  free(B1);
  free(B2);
}
