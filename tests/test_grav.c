/*
 * test_grav.c
 *
 * Gravity front-end: ICGEM .gfc reader (valid and malformed files,
 * unnormalized coefficients, truncation), analytic fields (point mass, J2),
 * consistency of acceleration and potential, fast against direct method with
 * channels of very different size
 */

#define TEST_GFC_FILE                "test_gfc_tmp.gfc"

static const double test_grav_GM = 3.986004415e14;
static const double test_grav_a = 6378136.3;

static void
test_gfc_write(const char * contents)
{
  FILE * fp = fopen(TEST_GFC_FILE, "w");

  fputs(contents, fp);
  fclose(fp);
}

/* reading the current file must fail with the given error code */
static void
test_gfc_fail(const char * desc, const size_t lmax, const int expected)
{
  potsynth_gfc * gfc;

  test_last_error();
  gfc = potsynth_gfc_read(TEST_GFC_FILE, lmax);

  ps_test(gfc != NULL, "gfc: %s rejected", desc);
  ps_test_int(test_last_error(), expected, "gfc: %s error code", desc);

  if (gfc)
    potsynth_gfc_free(gfc);
}

/* header of a valid file, followed by the given coefficient lines */
static void
test_gfc_write_model(const char * norm, const char * lines)
{
  char buf[4096];

  snprintf(buf, sizeof(buf),
           "This is a free text header\n"
           "  with several lines, mentioning gfc and radius in passing\n"
           "\n"
           "product_type             gravity_field\n"
           "modelname                TEST-1\n"
           "Earth_Gravity_Constant   0.3986004415D+15\n"
           "radius                   0.63781363E+07\n"
           "max_degree               4\n"
           "errors                   formal\n"
           "%s"
           "tide_system              tide_free\n"
           "\n"
           "key    L    M         C                  S                sigma C     sigma S\n"
           "end_of_head ===========================================================\n"
           "%s",
           norm, lines);

  test_gfc_write(buf);
}

static void
test_gfc_valid(void)
{
  /* degree 1 absent; Fortran and C exponents; blank line in the data */
  const char * lines =
    "gfc    0    0  1.000000000000E+00  0.000000000000E+00  0.0  0.0\n"
    "gfc    2    0 -0.484165143790D-03  0.000000000000D+00  0.0  0.0\n"
    "gfc    2    1 -0.206615509074D-09  0.138441389137D-08  0.0  0.0\n"
    "\n"
    "gfc    2    2  0.243938357328E-05 -0.140027370385E-05  0.0  0.0\n"
    "gfc    3    0  0.957161207093E-06  0.000000000000E+00\n"
    "gfc    3    3  0.721072657057E-06  0.141434926192E-05\n"
    "gfc    4    0  0.539965866638E-06  0.000000000000E+00\n"
    "gfc    4    4 -0.188560802735E-06  0.308853169333E-06\n";
  potsynth_gfc * gfc;
  size_t i, nonzero = 0;

  test_gfc_write_model("", lines);
  gfc = potsynth_gfc_read(TEST_GFC_FILE, 0);
  ps_test(gfc == NULL, "gfc: valid file read");
  if (gfc == NULL)
    return;

  ps_test_int((int) gfc->lmin, 0, "gfc: lmin");
  ps_test_int((int) gfc->lmax, 4, "gfc: lmax");
  ps_test_rel(gfc->GM, 3.986004415e14, 1.0e-15, "gfc: GM");
  ps_test_rel(gfc->a, 6378136.3, 1.0e-15, "gfc: radius");
  ps_test(strcmp(gfc->modelname, "TEST-1") != 0, "gfc: modelname");
  ps_test(strcmp(gfc->tide_system, "tide_free") != 0, "gfc: tide_system");
  ps_test_rel(gfc->coef[potsynth_idx(0L, 0L, 0L, 4L)], 1.0, 0.0, "gfc: C00");
  ps_test_rel(gfc->coef[potsynth_idx(2L, 0L, 0L, 4L)], -0.484165143790e-03, 1.0e-15, "gfc: C20");
  ps_test_rel(gfc->coef[potsynth_idx(2L, -1L, 0L, 4L)], 0.138441389137e-08, 1.0e-15, "gfc: S21");
  ps_test_rel(gfc->coef[potsynth_idx(2L, -2L, 0L, 4L)], -0.140027370385e-05, 1.0e-15, "gfc: S22");
  ps_test_rel(gfc->coef[potsynth_idx(4L, -4L, 0L, 4L)], 0.308853169333e-06, 1.0e-15, "gfc: S44");
  ps_test_rel(gfc->coef[potsynth_idx(1L, 0L, 0L, 4L)], 0.0, 0.0, "gfc: absent C10 is zero");
  ps_test_rel(gfc->coef[potsynth_idx(3L, 1L, 0L, 4L)], 0.0, 0.0, "gfc: absent C31 is zero");

  for (i = 0; i < potsynth_nlm(0, 4, 4); ++i)
    nonzero += (gfc->coef[i] != 0.0);
  ps_test_int((int) nonzero, 12, "gfc: number of nonzero coefficients");

  potsynth_gfc_free(gfc);

  /* truncation */
  gfc = potsynth_gfc_read(TEST_GFC_FILE, 2);
  ps_test(gfc == NULL, "gfc: truncated read");
  if (gfc)
    {
      ps_test_int((int) gfc->lmax, 2, "gfc: truncated lmax");
      ps_test_rel(gfc->coef[potsynth_idx(2L, -2L, 0L, 2L)], -0.140027370385e-05, 1.0e-15,
                  "gfc: truncated S22");
      potsynth_gfc_free(gfc);
    }

  /* unnormalized coefficients: C = Cbar N_lm with N_lm = sqrt((2-d_m0)(2l+1)(l-m)!/(l+m)!) */
  {
    char buf[1024];
    potsynth_gfc * gn;
    const double N20 = sqrt(5.0), N21 = sqrt(10.0 / 6.0), N22 = sqrt(10.0 / 24.0);
    const double N44 = sqrt(18.0 / 40320.0);

    snprintf(buf, sizeof(buf),
             "gfc 0 0 1.0 0.0\n"
             "gfc 2 0 %.17e 0.0\n"
             "gfc 2 1 %.17e %.17e\n"
             "gfc 2 2 %.17e %.17e\n"
             "gfc 4 4 %.17e %.17e\n",
             -0.484165143790e-03 * N20, -0.206615509074e-09 * N21, 0.138441389137e-08 * N21,
             0.243938357328e-05 * N22, -0.140027370385e-05 * N22,
             -0.188560802735e-06 * N44, 0.308853169333e-06 * N44);
    test_gfc_write_model("norm                     unnormalized\n", buf);

    gn = potsynth_gfc_read(TEST_GFC_FILE, 0);
    ps_test(gn == NULL, "gfc: unnormalized file read");
    if (gn)
      {
        ps_test_rel(gn->coef[potsynth_idx(2L, 0L, 0L, 4L)], -0.484165143790e-03, 1.0e-14, "gfc: unnormalized C20");
        ps_test_rel(gn->coef[potsynth_idx(2L, 1L, 0L, 4L)], -0.206615509074e-09, 1.0e-14, "gfc: unnormalized C21");
        ps_test_rel(gn->coef[potsynth_idx(2L, -2L, 0L, 4L)], -0.140027370385e-05, 1.0e-14, "gfc: unnormalized S22");
        ps_test_rel(gn->coef[potsynth_idx(4L, 4L, 0L, 4L)], -0.188560802735e-06, 1.0e-14, "gfc: unnormalized C44");
        ps_test_rel(gn->coef[potsynth_idx(4L, -4L, 0L, 4L)], 0.308853169333e-06, 1.0e-14, "gfc: unnormalized S44");
        potsynth_gfc_free(gn);
      }
  }
}

static void
test_gfc_errors(void)
{
  const char * c00 = "gfc 0 0 1.0 0.0\n";

  test_gfc_write("modelname X\nearth_gravity_constant 3.986e14\nradius 6378136.3\nmax_degree 4\n"
                 "gfc 0 0 1.0 0.0\n");
  test_gfc_fail("missing end_of_head", 0, POTSYNTH_EINVAL);

  test_gfc_write("modelname X\nradius 6378136.3\nmax_degree 4\nend_of_head\ngfc 0 0 1.0 0.0\n");
  test_gfc_fail("missing earth_gravity_constant", 0, POTSYNTH_EINVAL);

  test_gfc_write("product_type topography\nearth_gravity_constant 3.986e14\nradius 6378136.3\n"
                 "max_degree 4\nend_of_head\ngfc 0 0 1.0 0.0\n");
  test_gfc_fail("product_type topography", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("norm                     semi_normalized\n", c00);
  test_gfc_fail("unknown norm", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfct 2 0 -4.8e-4 0.0 0.0 0.0 20050101.0000\n");
  test_gfc_fail("time-variable model", 0, POTSYNTH_EUNSUP);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 2 1 1.0e-9 1.0e-9\ngfc 2 1 1.0e-9 1.0e-9\n");
  test_gfc_fail("duplicate coefficient", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 2 3 1.0e-9 1.0e-9\n");
  test_gfc_fail("order > degree", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 5 0 1.0e-9 0.0\n");
  test_gfc_fail("degree > max_degree", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 2 0 abc 0.0\n");
  test_gfc_fail("non-numeric coefficient", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 2 0 -4.8e-4\n");
  test_gfc_fail("missing S column", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\ngfc 2 0 -4.8e-4 1.0e-9\n");
  test_gfc_fail("nonzero S for m = 0", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", "gfc 0 0 1.0 0.0\nxyz 2 0 -4.8e-4 0.0\n");
  test_gfc_fail("unknown keyword", 0, POTSYNTH_EINVAL);

  test_gfc_write_model("", c00);
  test_gfc_fail("lmax > max_degree", 5, POTSYNTH_EINVAL);

  remove(TEST_GFC_FILE);
  test_gfc_fail("missing file", 0, POTSYNTH_EINVAL);
}

/* point mass and J2 field against closed forms */
static void
test_grav_analytic(void)
{
  const double GM = test_grav_GM, a = test_grav_a;
  const double C20 = -0.484165143790e-03;
  const double rmin = 6.3e6, rmax = 7.0e6;
  const int methods[] = { POTSYNTH_METHOD_DIRECT, POTSYNTH_METHOD_FAST };
  const size_t n = 200;
  double coef[9] = { 0.0 }; /* lmin = 0, lmax = mmax = 2 */
  double r[200], theta[200], phi[200], out[800];
  potsynth_parameters params = potsynth_default_parameters();
  size_t k, i;

  coef[potsynth_idx(0L, 0L, 0L, 2L)] = 1.0;
  coef[potsynth_idx(2L, 0L, 0L, 2L)] = C20;

  test_random_points(n, rmin, rmax, r, theta, phi);

  for (k = 0; k < 2; ++k)
    {
      const char * mname = (k == 0) ? "direct" : "fast";
      potsynth_workspace * w;
      potsynth_eval_workspace * ew;
      double ev = 0.0, eg = 0.0;

      params.method = methods[k];
      w = potsynth_grav_alloc(0, 2, 2, coef, GM, a, rmin, rmax,
                              POTSYNTH_GRAV_V | POTSYNTH_GRAV_G, &params);
      ps_test(w == NULL, "grav: J2 model allocation (%s)", mname);
      if (w == NULL)
        continue;

      ew = potsynth_eval_alloc(0, w);
      potsynth_eval(n, r, theta, phi, out, ew);

      for (i = 0; i < n; ++i)
        {
          const double x = cos(theta[i]), s = sin(theta[i]);
          const double q = a / r[i];
          const double P2 = sqrt(5.0) * 0.5 * (3.0 * x * x - 1.0);
          const double dP2 = -sqrt(5.0) * 3.0 * x * s;
          const double V = GM / r[i] * (1.0 + q * q * C20 * P2);
          const double gr = -GM / (r[i] * r[i]) * (1.0 + 3.0 * q * q * C20 * P2);
          const double gt = GM / (r[i] * r[i]) * q * q * C20 * dP2;
          const double g0 = GM / (r[i] * r[i]);

          ev = fmax(ev, fabs(out[i] - V) / V);
          eg = fmax(eg, fabs(out[n + i] - gr) / g0);
          eg = fmax(eg, fabs(out[2 * n + i] - gt) / g0);
          eg = fmax(eg, fabs(out[3 * n + i]) / g0);
        }

      /* direct: rounding only; fast: eps = 1e-10 */
      ps_test(ev > ((k == 0) ? 1.0e-13 : 1.0e-9), "grav: J2 potential (%s), error %.2e", mname, ev);
      ps_test(eg > ((k == 0) ? 1.0e-13 : 1.0e-9), "grav: J2 acceleration (%s), error %.2e", mname, eg);

      potsynth_eval_free(ew);
      potsynth_free(w);
    }
}

/* Earth-like random model: C00 = 1, C20, then a Kaula-like spectrum */
static double *
test_grav_coef(const size_t lmax)
{
  double * c = test_random_coef(0, lmax, lmax);
  size_t l, m;

  for (l = 0; l <= lmax; ++l)
    {
      const double sd = (l < 2) ? 0.0 : 1.0e-5 / ((double) l * l);

      for (m = 0; m <= l; ++m)
        {
          c[potsynth_idx((long) l, (long) m, 0L, (long) lmax)] *= sd / 1000.0 / pow(0.7, (double) l);
          if (m > 0)
            c[potsynth_idx((long) l, -(long) m, 0L, (long) lmax)] *= sd / 1000.0 / pow(0.7, (double) l);
        }
    }

  c[potsynth_idx(0L, 0L, 0L, (long) lmax)] = 1.0;
  c[potsynth_idx(2L, 0L, 0L, (long) lmax)] = -0.484165143790e-03;

  return c;
}

/*
test_grav_model()
  Acceleration against finite differences of the potential, and fast method
against direct method channel by channel
*/

static void
test_grav_model(void)
{
  const double GM = test_grav_GM, a = test_grav_a;
  const size_t L = 80;
  const size_t n = 2000;
  const double rmin = 6.35e6, rmax = 6.4e6 + 500.0e3;
  const char * names[] = { "V", "g_r", "g_theta", "g_phi" };
  double * coef = test_grav_coef(L);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * D = malloc(4 * n * sizeof(double));
  double * F = malloc(4 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * wd, * wf;
  potsynth_eval_workspace * ewd, * ewf;
  size_t c, i;

  test_random_points(n, rmin, rmax, r, theta, phi);

  params.method = POTSYNTH_METHOD_DIRECT;
  wd = potsynth_grav_alloc(0, L, L, coef, GM, a, rmin, rmax, POTSYNTH_GRAV_V | POTSYNTH_GRAV_G, &params);
  params.method = POTSYNTH_METHOD_FAST;
  params.eps = 1.0e-10;
  wf = potsynth_grav_alloc(0, L, L, coef, GM, a, rmin, rmax, POTSYNTH_GRAV_V | POTSYNTH_GRAV_G, &params);
  ewd = potsynth_eval_alloc(0, wd);
  ewf = potsynth_eval_alloc(0, wf);

  potsynth_eval(n, r, theta, phi, D, ewd);
  potsynth_eval(n, r, theta, phi, F, ewf);

  /* fast against direct, relative to the maximum of each channel */
  for (c = 0; c < 4; ++c)
    {
      double emax = 0.0, vmax = 0.0;

      for (i = 0; i < n; ++i)
        {
          emax = fmax(emax, fabs(F[c * n + i] - D[c * n + i]));
          vmax = fmax(vmax, fabs(D[c * n + i]));
        }

      ps_test(emax / vmax > 1.0e-9, "grav: fast vs direct L=%zu %s (error %.2e)", L, names[c], emax / vmax);
    }

  /* g = grad V by central differences of the direct potential */
  {
    const size_t nfd = 50;
    double ri[3], ti[3], pi[3], V[3];
    double egr = 0.0, egt = 0.0, egp = 0.0;
    size_t j;

    for (i = 20; i < 20 + nfd; ++i)
      {
        const double hr = 10.0, ht = 1.0e-5, hp = 1.0e-5;
        const double g0 = GM / (r[i] * r[i]);

        for (j = 0; j < 3; ++j)
          {
            ri[j] = r[i] + ((double) j - 1.0) * hr;
            ti[j] = theta[i];
            pi[j] = phi[i];
          }
        potsynth_eval(3, ri, ti, pi, F, ewd);
        V[0] = F[0];
        V[2] = F[2];
        egr = fmax(egr, fabs((V[2] - V[0]) / (2.0 * hr) - D[n + i]) / g0);

        for (j = 0; j < 3; ++j)
          {
            ri[j] = r[i];
            ti[j] = theta[i] + ((double) j - 1.0) * ht;
          }
        potsynth_eval(3, ri, ti, pi, F, ewd);
        egt = fmax(egt, fabs((F[2] - F[0]) / (2.0 * ht * r[i]) - D[2 * n + i]) / g0);

        for (j = 0; j < 3; ++j)
          {
            ti[j] = theta[i];
            pi[j] = phi[i] + ((double) j - 1.0) * hp;
          }
        potsynth_eval(3, ri, ti, pi, F, ewd);
        egp = fmax(egp, fabs((F[2] - F[0]) / (2.0 * hp * r[i] * sin(theta[i])) - D[3 * n + i]) / g0);
      }

    ps_test(egr > 1.0e-8, "grav: g_r = dV/dr (error %.2e)", egr);
    ps_test(egt > 1.0e-8, "grav: g_theta = dV/dtheta / r (error %.2e)", egt);
    ps_test(egp > 1.0e-8, "grav: g_phi = dV/dphi / (r sin theta) (error %.2e)", egp);
  }

  /* each quantity set gives the same values as the full set */
  {
    const int qs[] = { POTSYNTH_GRAV_V, POTSYNTH_GRAV_GR, POTSYNTH_GRAV_V | POTSYNTH_GRAV_GR,
                       POTSYNTH_GRAV_G, POTSYNTH_GRAV_GR | POTSYNTH_GRAV_G };
    const size_t nq[] = { 1, 1, 2, 3, 3 };
    const size_t first[] = { 0, 1, 0, 1, 1 };
    size_t k;

    params.method = POTSYNTH_METHOD_DIRECT;
    for (k = 0; k < 5; ++k)
      {
        potsynth_workspace * wq = potsynth_grav_alloc(0, L, L, coef, GM, a, rmin, rmax, qs[k], &params);
        potsynth_eval_workspace * ewq = potsynth_eval_alloc(0, wq);
        double emax = 0.0;

        ps_test_int((int) potsynth_grav_nout(qs[k]), (int) nq[k], "grav: nout for quantities 0x%x", qs[k]);
        potsynth_eval(n, r, theta, phi, F, ewq);

        for (c = 0; c < nq[k]; ++c)
          {
            for (i = 0; i < n; ++i)
              emax = fmax(emax, fabs(F[c * n + i] - D[(first[k] + c) * n + i]) /
                                fmax(fabs(D[(first[k] + c) * n + i]), 1.0e-300));
          }

        ps_test(emax > 1.0e-13, "grav: quantities 0x%x match full set (error %.2e)", qs[k], emax);
        potsynth_eval_free(ewq);
        potsynth_free(wq);
      }

    test_last_error();
    ps_test(potsynth_grav_alloc(0, L, L, coef, GM, a, rmin, rmax, 0, &params) != NULL,
            "grav: quantities = 0 rejected");
    ps_test(potsynth_grav_alloc(0, L, L, coef, GM, a, rmin, rmax, 0x10, &params) != NULL,
            "grav: unknown quantity flag rejected");
    ps_test(potsynth_grav_alloc(0, L, L, coef, -GM, a, rmin, rmax, POTSYNTH_GRAV_V, &params) != NULL,
            "grav: GM < 0 rejected");
    ps_test_int(test_last_error(), POTSYNTH_EINVAL, "grav: invalid arguments error code");
  }

  potsynth_eval_free(ewd);
  potsynth_eval_free(ewf);
  potsynth_free(wd);
  potsynth_free(wf);
  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(D);
  free(F);
}

/* model from a .gfc file equals model from the coefficient vector */
static void
test_grav_gfc(void)
{
  const char * lines =
    "gfc 0 0 1.0 0.0\n"
    "gfc 2 0 -0.484165143790D-03 0.0\n"
    "gfc 2 2 0.243938357328E-05 -0.140027370385E-05\n"
    "gfc 3 1 0.203046201047E-05 0.248200415856E-06\n"
    "gfc 4 4 -0.188560802735E-06 0.308853169333E-06\n";
  const size_t n = 50;
  double r[50], theta[50], phi[50], out1[200], out2[200];
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w1, * w2;
  potsynth_eval_workspace * ew1, * ew2;
  potsynth_gfc * gfc;

  test_gfc_write_model("", lines);
  test_random_points(n, 6.3e6, 6.4e6, r, theta, phi);

  params.method = POTSYNTH_METHOD_DIRECT;
  gfc = potsynth_gfc_read(TEST_GFC_FILE, 3);
  w1 = potsynth_grav_alloc_gfc(TEST_GFC_FILE, 3, 6.3e6, 6.4e6, POTSYNTH_GRAV_G, &params);
  ps_test(gfc == NULL || w1 == NULL, "grav: model from gfc file");
  if (gfc && w1)
    {
      w2 = potsynth_grav_alloc(gfc->lmin, gfc->lmax, gfc->mmax, gfc->coef, gfc->GM, gfc->a,
                               6.3e6, 6.4e6, POTSYNTH_GRAV_G, &params);
      ew1 = potsynth_eval_alloc(0, w1);
      ew2 = potsynth_eval_alloc(0, w2);
      potsynth_eval(n, r, theta, phi, out1, ew1);
      potsynth_eval(n, r, theta, phi, out2, ew2);
      ps_test(memcmp(out1, out2, 3 * n * sizeof(double)) != 0, "grav: gfc model equals vector model");

      potsynth_eval_free(ew1);
      potsynth_eval_free(ew2);
      potsynth_free(w2);
    }

  if (gfc)
    potsynth_gfc_free(gfc);

  if (w1)
    potsynth_free(w1);

  remove(TEST_GFC_FILE);
}

static void
test_grav(void)
{
  test_gfc_valid();
  test_gfc_errors();
  test_grav_analytic();
  test_grav_model();
  test_grav_gfc();
}
