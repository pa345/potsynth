/*
 * test_eval.c
 *
 * Fast method against direct method for many models and channel sets
 */

/* random coefficients with a decaying spectrum, indexed by potsynth_idx() */
static double *
test_random_coef(const size_t lmin, const size_t lmax, const size_t mmax)
{
  double * c = malloc(potsynth_nlm(lmin, lmax, mmax) * sizeof(double));
  size_t l, m;

  for (l = lmin; l <= lmax; ++l)
    {
      const size_t M = (l < mmax) ? l : mmax;
      const double sd = 1000.0 * pow(0.7, (double) l);

      for (m = 0; m <= M; ++m)
        {
          c[potsynth_idx((long) l, (long) m, (long) lmin, (long) mmax)] = sd * (2.0 * test_urand() - 1.0);
          if (m > 0)
            c[potsynth_idx((long) l, -(long) m, (long) lmin, (long) mmax)] = sd * (2.0 * test_urand() - 1.0);
        }
    }

  return c;
}

/* random points in [rmin,rmax], including poles, seams and large longitudes */
static void
test_random_points(const size_t n, const double rmin, const double rmax,
                   double * r, double * theta, double * phi)
{
  const double special[][2] = { { 0.0, 0.0 }, { M_PI, 0.0 }, { 0.0, 2.0 }, { M_PI, -1.0 },
                                { 1.0e-12, 3.0 }, { M_PI - 1.0e-12, 5.0 }, { M_PI / 2.0, 0.0 },
                                { 1.0, 2.0 * M_PI - 1.0e-14 }, { 2.0, -7.0 }, { 0.5, 1.0e6 } };
  const size_t nspecial = sizeof(special) / sizeof(special[0]);
  size_t i;

  for (i = 0; i < n; ++i)
    {
      if (i < nspecial)
        {
          r[i] = (i & 1) ? rmax : rmin;
          theta[i] = special[i][0];
          phi[i] = special[i][1];
        }
      else
        {
          r[i] = rmin + (rmax - rmin) * test_urand();
          theta[i] = acos(2.0 * test_urand() - 1.0);
          phi[i] = 2.0 * M_PI * test_urand();
        }
    }
}

/*
test_eval_model()
  Evaluate the same model with the fast and direct methods and compare

Inputs: desc  - description
        lmin  - minimum degree
        lmax  - maximum degree
        mmax  - maximum order
        rmin  - minimum radius
        rmax  - maximum radius
        nchan - number of channels
        chan  - channels, or NULL for the magnetic front-end
        eps   - requested accuracy
        tol   - tolerance of the comparison
*/

static void
test_eval_model(const char * desc, const size_t lmin, const size_t lmax, const size_t mmax,
                const double rmin, const double rmax, const size_t nchan,
                const potsynth_channel * chan, const double eps, const double tol)
{
  const double a = 6371.2;
  const size_t n = 3000;
  double * coef = test_random_coef(lmin, lmax, mmax);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * Bf = malloc(nchan * n * sizeof(double));
  double * Bd = malloc(nchan * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * wf, * wd;
  double emax = 0.0, bmax = 0.0;
  size_t i;

  test_random_points(n, rmin, rmax, r, theta, phi);

  params.eps = eps;

  params.method = POTSYNTH_METHOD_FAST;
  if (chan)
    wf = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, &params);
  else
    wf = potsynth_mag_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, POTSYNTH_MAG_B, &params);

  params.method = POTSYNTH_METHOD_DIRECT;
  if (chan)
    wd = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, &params);
  else
    wd = potsynth_mag_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, POTSYNTH_MAG_B, &params);

  ps_test(wf == NULL || wd == NULL, "eval: %s allocation", desc);

  if (wf && wd)
    {
      potsynth_eval_workspace * ef = potsynth_eval_alloc(0, wf);
      potsynth_eval_workspace * ed = potsynth_eval_alloc(0, wd);

      ps_test_int(potsynth_eval(n, r, theta, phi, Bf, ef), POTSYNTH_SUCCESS, "eval: %s fast", desc);
      ps_test_int(potsynth_eval(n, r, theta, phi, Bd, ed), POTSYNTH_SUCCESS, "eval: %s direct", desc);

      for (i = 0; i < nchan * n; ++i)
        {
          emax = fmax(emax, fabs(Bf[i] - Bd[i]));
          bmax = fmax(bmax, fabs(Bd[i]));
        }

      ps_test(emax / bmax > tol, "eval: %s fast vs direct (difference %.2e)", desc, emax / bmax);

      potsynth_eval_free(ef);
      potsynth_eval_free(ed);
    }

  if (wf)
    potsynth_free(wf);

  if (wd)
    potsynth_free(wd);

  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(Bf);
  free(Bd);
}

static void
test_eval(void)
{
  const double a = 6371.2;
  const int ops[7] = { POTSYNTH_THETA_P, POTSYNTH_THETA_P, POTSYNTH_THETA_DP, POTSYNTH_THETA_DP,
                       POTSYNTH_THETA_PSIN, POTSYNTH_THETA_PSIN, POTSYNTH_THETA_P };
  const int dphi[7] = { 0, 1, 0, 1, 0, 1, 0 };
  const int rpow[7] = { 2, 2, 2, 2, 2, 2, 1 };
  double rfac_v[121], rfac_dv[121];
  potsynth_channel grav[2], all[7];
  size_t l, c;

  /* magnetic front-end */
  test_eval_model("mag L=60", 1, 60, 60, a + 250.0, a + 800.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag lmin=16 L=80", 16, 80, 80, a + 250.0, a + 800.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag mmax=10 L=50", 1, 50, 10, a + 250.0, a + 800.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag lmin=5 mmax=3 L=40", 5, 40, 3, a + 250.0, a + 800.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag rmin=rmax", 1, 60, 60, a + 450.0, a + 450.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag L=1", 1, 1, 1, a, a + 1000.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag L=2", 1, 2, 2, a, a + 1000.0, 3, NULL, 1.0e-12, 1.0e-11);
  test_eval_model("mag L=30 eps=1e-6", 1, 30, 30, a, a + 1000.0, 3, NULL, 1.0e-6, 2.0e-6);

  /* gravity-like: potential (u^(l+1)) and radial derivative (u^(l+2)), lmin = 0 */
  for (l = 0; l <= 120; ++l)
    {
      rfac_v[l] = 1.0;
      rfac_dv[l] = l + 1.0;
    }

  grav[0].rpow = 1;
  grav[0].rfac = rfac_v;
  grav[0].theta = POTSYNTH_THETA_P;
  grav[0].dphi = 0;

  grav[1].rpow = 2;
  grav[1].rfac = rfac_dv;
  grav[1].theta = POTSYNTH_THETA_P;
  grav[1].dphi = 0;

  test_eval_model("V and dV/dr lmin=0 L=120", 0, 120, 120, a, a + 500.0, 2, grav, 1.0e-12, 1.0e-11);
  test_eval_model("V only L=50", 0, 50, 50, a, a + 500.0, 1, grav, 1.0e-12, 1.0e-11);

  /* every theta operator with and without phi derivative: 7 channels, more than
   * the specialized direct kernels, so the generic direct path runs */
  for (c = 0; c < 7; ++c)
    {
      all[c].rpow = rpow[c];
      all[c].rfac = (c == 0) ? rfac_dv : rfac_v;
      all[c].theta = ops[c];
      all[c].dphi = dphi[c];
    }

  test_eval_model("7 channels L=70", 1, 70, 70, a, a + 500.0, 7, all, 1.0e-12, 1.0e-11);
}
