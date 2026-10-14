/*
 * test_hess.c
 *
 * Gradient tensor (Hessian) of the potential with the second-derivative
 * operators THETA_S2A and THETA_S2B:
 *
 * - against 40-digit reference values (tests/gen_hessian_ref.py), computed
 *   from the Cartesian Hessian independently of the spherical formulas, at
 *   points including both poles at several longitudes
 * - against finite differences of the gradient (covariant-derivative formulas)
 * - continuity at the poles, isotropy of a zonal field on the axis
 * - terms processed in groups (more channels than one vector kernel holds)
 * - fast method against direct method, channel by channel, at several
 *   accuracies, alone and mixed with the gradient channels
 *
 * The tensor of V = sum (a/r)^(l+1) Pbar_l^m [C cos + S sin] is built from
 * five channels, with factors u^(l+3)/a^2:
 *
 * T_rr = (l+1)(l+2) P, T_rt = -(l+2) dP, T_rp = -(l+2) Psin d/dphi,
 * D = T_tt - T_pp = S2A, T_tp = S2B d/dphi,
 *
 * and T_tt = (-T_rr + D)/2, T_pp = (-T_rr - D)/2.
 */

#define TEST_HESS_NCHAN              5

/*
test_hess_alloc()
  Allocate a model with the five tensor channels, optionally followed by the
three gradient channels g = grad V (g_r, g_theta, g_phi)
*/

static potsynth_workspace *
test_hess_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * coef,
                const double a, const double rmin, const double rmax, const int with_grad,
                const potsynth_parameters * params)
{
  const size_t L1 = lmax + 1;
  const int ops[] = { POTSYNTH_THETA_P, POTSYNTH_THETA_DP, POTSYNTH_THETA_PSIN,
                      POTSYNTH_THETA_S2A, POTSYNTH_THETA_S2B,
                      POTSYNTH_THETA_P, POTSYNTH_THETA_DP, POTSYNTH_THETA_PSIN };
  const int dphi[] = { 0, 0, 1, 0, 1, 0, 0, 1 };
  const int rpow[] = { 3, 3, 3, 3, 3, 2, 2, 2 };
  const size_t nchan = with_grad ? 8 : 5;
  double * rfac = malloc(8 * L1 * sizeof(double));
  potsynth_channel chan[8];
  potsynth_workspace * w;
  size_t c, l;

  for (l = 0; l <= lmax; ++l)
    {
      const double dl = (double) l;

      rfac[0 * L1 + l] = (dl + 1.0) * (dl + 2.0) / (a * a);
      rfac[1 * L1 + l] = -(dl + 2.0) / (a * a);
      rfac[2 * L1 + l] = -(dl + 2.0) / (a * a);
      rfac[3 * L1 + l] = 1.0 / (a * a);
      rfac[4 * L1 + l] = 1.0 / (a * a);
      rfac[5 * L1 + l] = -(dl + 1.0) / a;
      rfac[6 * L1 + l] = 1.0 / a;
      rfac[7 * L1 + l] = 1.0 / a;
    }

  for (c = 0; c < nchan; ++c)
    {
      chan[c].rpow = rpow[c];
      chan[c].rfac = rfac + c * L1;
      chan[c].theta = ops[c];
      chan[c].dphi = dphi[c];
    }

  w = potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, params);
  free(rfac);

  return w;
}

/*
test_hess_tensor()
  Convert the five channels F[c*n + i] to the six components
T[k*n + i], k = rr, rt, rp, tt, tp, pp
*/

static void
test_hess_tensor(const size_t n, const double * F, double * T)
{
  size_t i;

  for (i = 0; i < n; ++i)
    {
      const double Trr = F[i], D = F[3 * n + i];

      T[i] = Trr;
      T[n + i] = F[n + i];
      T[2 * n + i] = F[2 * n + i];
      T[3 * n + i] = 0.5 * (-Trr + D);
      T[4 * n + i] = F[4 * n + i];
      T[5 * n + i] = 0.5 * (-Trr - D);
    }
}

/* against the 40-digit reference values */
static void
test_hess_ref(const char * file)
{
  char filename[1024];
  FILE * fp;
  int64_t hdr[2];
  double a, * coef, * r, * theta, * phi, * Tref, * F, * T;
  size_t L, n, nlm, i, k;
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  double emax = 0.0, emax_pole = 0.0, tmax = 0.0;
  int ok;

  snprintf(filename, sizeof(filename), "%s/%s", TEST_DATA_DIR, file);
  fp = fopen(filename, "rb");
  ps_test(fp == NULL, "hess: open %s", file);
  if (fp == NULL)
    return;

  ok = fread(hdr, sizeof(int64_t), 2, fp) == 2 && fread(&a, sizeof(double), 1, fp) == 1;
  L = (size_t) hdr[0];
  n = (size_t) hdr[1];
  nlm = potsynth_nlm(0, L, L);
  coef = malloc(nlm * sizeof(double));
  r = malloc(n * sizeof(double));
  theta = malloc(n * sizeof(double));
  phi = malloc(n * sizeof(double));
  Tref = malloc(6 * n * sizeof(double));
  F = malloc(TEST_HESS_NCHAN * n * sizeof(double));
  T = malloc(6 * n * sizeof(double));
  ok = ok && fread(coef, sizeof(double), nlm, fp) == nlm;
  ok = ok && fread(r, sizeof(double), n, fp) == n;
  ok = ok && fread(theta, sizeof(double), n, fp) == n;
  ok = ok && fread(phi, sizeof(double), n, fp) == n;
  ok = ok && fread(Tref, sizeof(double), 6 * n, fp) == 6 * n;
  fclose(fp);
  ps_test(!ok, "hess: read %s", file);

  params.method = POTSYNTH_METHOD_DIRECT;
  w = test_hess_alloc(0, L, L, coef, a, 0.0, 0.0, 0, &params);
  ew = potsynth_eval_alloc(0, w);
  potsynth_eval(n, r, theta, phi, F, ew);
  test_hess_tensor(n, F, T);

  for (i = 0; i < 6 * n; ++i)
    tmax = fmax(tmax, fabs(Tref[i]));

  for (k = 0; k < 6; ++k)
    {
      for (i = 0; i < n; ++i)
        {
          const double e = fabs(T[k * n + i] - Tref[k * n + i]) / tmax;

          emax = fmax(emax, e);
          if (theta[i] == 0.0 || theta[i] == M_PI)
            emax_pole = fmax(emax_pole, e);
        }
    }

  ps_test(emax > 1.0e-13, "hess: %s direct, all points (error %.2e)", file, emax);
  ps_test(emax_pole > 1.0e-13, "hess: %s direct, at the poles (error %.2e)", file, emax_pole);

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(Tref);
  free(F);
  free(T);
}

/*
test_hess_fd()
  Tensor against finite differences of the gradient B = grad V (spherical
basis, rows = component, columns = derivative direction):

T_rr = d_r B_r, T_rt = d_r B_t, T_rp = d_r B_p,
T_tt = (d_t B_t + B_r)/r, T_tp = d_p B_t/(r sin) - B_p cot/r,
T_pp = d_p B_p/(r sin) + B_r/r + B_t cot/r
*/

static void
test_hess_fd(void)
{
  const double a = 6371.2, rmin = a, rmax = a + 800.0;
  const size_t L = 60, n = 40;
  const double hr = 1.0e-3, ht = 2.0e-6;
  double * coef = test_random_coef(0, L, L);
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  double F[8 * 7], T[6], Tfd[6];
  double emax = 0.0, tmax = 0.0;
  size_t i, k;

  params.method = POTSYNTH_METHOD_DIRECT;
  w = test_hess_alloc(0, L, L, coef, a, 0.0, 0.0, 1, &params);
  ew = potsynth_eval_alloc(0, w);

  for (i = 0; i < n; ++i)
    {
      const double r0 = rmin + (rmax - rmin) * test_urand();
      const double t0 = 0.2 + (M_PI - 0.4) * test_urand();
      const double p0 = 2.0 * M_PI * test_urand();
      const double st = sin(t0), ct = cos(t0);
      /* point, r +- hr, theta +- ht, phi +- ht */
      double rr[7] = { r0, r0 + hr, r0 - hr, r0, r0, r0, r0 };
      double tt[7] = { t0, t0, t0, t0 + ht, t0 - ht, t0, t0 };
      double pp[7] = { p0, p0, p0, p0, p0, p0 + ht, p0 - ht };
      double Br, Bt, Bp;

      potsynth_eval(7, rr, tt, pp, F, ew);

      /* gradient channels 5, 6, 7 */
#define TEST_HESS_B(c, j) F[(c) * 7 + (j)]
      Br = TEST_HESS_B(5, 0);
      Bt = TEST_HESS_B(6, 0);
      Bp = TEST_HESS_B(7, 0);

      Tfd[0] = (TEST_HESS_B(5, 1) - TEST_HESS_B(5, 2)) / (2.0 * hr);
      Tfd[1] = (TEST_HESS_B(6, 1) - TEST_HESS_B(6, 2)) / (2.0 * hr);
      Tfd[2] = (TEST_HESS_B(7, 1) - TEST_HESS_B(7, 2)) / (2.0 * hr);
      Tfd[3] = ((TEST_HESS_B(6, 3) - TEST_HESS_B(6, 4)) / (2.0 * ht) + Br) / r0;
      Tfd[4] = (TEST_HESS_B(6, 5) - TEST_HESS_B(6, 6)) / (2.0 * ht * r0 * st) - Bp * ct / (st * r0);
      Tfd[5] = (TEST_HESS_B(7, 5) - TEST_HESS_B(7, 6)) / (2.0 * ht * r0 * st) + Br / r0 + Bt * ct / (st * r0);
#undef TEST_HESS_B

      T[0] = F[0];
      T[1] = F[7];
      T[2] = F[14];
      T[3] = 0.5 * (-F[0] + F[21]);
      T[4] = F[28];
      T[5] = 0.5 * (-F[0] - F[21]);

      for (k = 0; k < 6; ++k)
        {
          emax = fmax(emax, fabs(T[k] - Tfd[k]));
          tmax = fmax(tmax, fabs(T[k]));
        }
    }

  ps_test(emax / tmax > 1.0e-7, "hess: finite differences of the gradient, L=%zu (error %.2e)",
          L, emax / tmax);

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(coef);
}

/* continuity at the poles; isotropy of a zonal field on the axis */
static void
test_hess_poles(void)
{
  const double a = 1.0;
  const size_t L = 60;
  const size_t nlm = potsynth_nlm(0, L, L);
  const double phis[] = { 0.0, 0.7, 2.0, 4.5 };
  double * coef = malloc(nlm * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  double r[16], theta[16], phi[16], F[5 * 16], T[6 * 16];
  double econt = 0.0, tmax = 0.0, eiso = 0.0;
  size_t i, k, l, m;

  /* flat spectrum, so that high degrees matter at the poles */
  for (l = 0; l <= L; ++l)
    {
      for (m = 0; m <= l; ++m)
        {
          coef[potsynth_idx((long) l, (long) m, 0L, (long) L)] = (2.0 * test_urand() - 1.0) / (l + 1.0);
          if (m > 0)
            coef[potsynth_idx((long) l, -(long) m, 0L, (long) L)] = (2.0 * test_urand() - 1.0) / (l + 1.0);
        }
    }

  params.method = POTSYNTH_METHOD_DIRECT;

  /* pole and 1e-13 from it, same longitude: north (0..7), south (8..15) */
  for (i = 0; i < 4; ++i)
    {
      r[2 * i] = r[2 * i + 1] = 1.2;
      theta[2 * i] = 0.0;
      theta[2 * i + 1] = 1.0e-13;
      r[8 + 2 * i] = r[8 + 2 * i + 1] = 1.2;
      theta[8 + 2 * i] = M_PI;
      theta[8 + 2 * i + 1] = M_PI - 1.0e-13;
      phi[2 * i] = phi[2 * i + 1] = phi[8 + 2 * i] = phi[8 + 2 * i + 1] = phis[i];
    }

  w = test_hess_alloc(0, L, L, coef, a, 0.0, 0.0, 0, &params);
  ew = potsynth_eval_alloc(0, w);
  potsynth_eval(16, r, theta, phi, F, ew);
  test_hess_tensor(16, F, T);

  for (k = 0; k < 6; ++k)
    {
      for (i = 0; i < 16; i += 2)
        {
          econt = fmax(econt, fabs(T[k * 16 + i] - T[k * 16 + i + 1]));
          tmax = fmax(tmax, fabs(T[k * 16 + i]));
        }
    }

  ps_test(econt / tmax > 1.0e-10, "hess: continuity at the poles, L=%zu (difference %.2e)", L, econt / tmax);
  ps_test(!(tmax > 0.0) || !isfinite(tmax), "hess: finite at the poles");

  potsynth_eval_free(ew);
  potsynth_free(w);

  /* zonal model: on the axis T_rt = T_rp = T_tp = 0 and T_tt = T_pp = -T_rr/2 */
  for (l = 0; l <= L; ++l)
    {
      for (m = 1; m <= l; ++m)
        {
          coef[potsynth_idx((long) l, (long) m, 0L, (long) L)] = 0.0;
          coef[potsynth_idx((long) l, -(long) m, 0L, (long) L)] = 0.0;
        }
    }

  w = test_hess_alloc(0, L, L, coef, a, 0.0, 0.0, 0, &params);
  ew = potsynth_eval_alloc(0, w);
  potsynth_eval(16, r, theta, phi, F, ew);
  test_hess_tensor(16, F, T);

  for (i = 0; i < 16; i += 2)
    {
      const double s = fabs(T[i]);

      eiso = fmax(eiso, fabs(T[16 + i]) / s);
      eiso = fmax(eiso, fabs(T[2 * 16 + i]) / s);
      eiso = fmax(eiso, fabs(T[4 * 16 + i]) / s);
      eiso = fmax(eiso, fabs(T[3 * 16 + i] - T[5 * 16 + i]) / s);
    }

  ps_test(eiso > 1.0e-12, "hess: zonal field isotropic on the axis (error %.2e)", eiso);

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(coef);
}

/* terms in groups: 8 channels (11 terms) give the same tensor as 5 (bitwise) */
static void
test_hess_groups(void)
{
  const double a = 6371.2;
  const size_t L = 30, n = 300;
  double * coef = test_random_coef(0, L, 20);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * F5 = malloc(5 * n * sizeof(double));
  double * F8 = malloc(8 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w5, * w8;
  potsynth_eval_workspace * ew5, * ew8;

  test_random_points(n, a, a + 500.0, r, theta, phi);

  params.method = POTSYNTH_METHOD_DIRECT;
  w5 = test_hess_alloc(0, L, 20, coef, a, a, a + 500.0, 0, &params);
  w8 = test_hess_alloc(0, L, 20, coef, a, a, a + 500.0, 1, &params);
  ew5 = potsynth_eval_alloc(0, w5);
  ew8 = potsynth_eval_alloc(0, w8);
  potsynth_eval(n, r, theta, phi, F5, ew5);
  potsynth_eval(n, r, theta, phi, F8, ew8);
  ps_test(memcmp(F5, F8, 5 * n * sizeof(double)) != 0, "hess: 5 and 8 channels agree (mmax < lmax)");

  potsynth_eval_free(ew5);
  potsynth_eval_free(ew8);
  potsynth_free(w5);
  potsynth_free(w8);
  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(F5);
  free(F8);
}

/*
test_hess_fast()
  Fast method against direct method for every channel, relative to the
maximum of each channel, with the tensor alone and mixed with the gradient
(8 channels of different operators and sizes); the points include both
poles at several longitudes
*/

static void
test_hess_fast(void)
{
  const char * names[] = { "T_rr", "T_rt", "T_rp", "D", "T_tp", "g_r", "g_t", "g_p" };
  const double a = 6371.2, rmin = a + 250.0, rmax = a + 800.0;
  const double eps[] = { 1.0e-6, 1.0e-10, 1.0e-13 };
  const double tol[] = { 1.0e-5, 1.0e-9, 1.0e-12 };
  const size_t n = 3000;
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * D = malloc(8 * n * sizeof(double));
  double * F = malloc(8 * n * sizeof(double));
  size_t iL;

  test_random_points(n, rmin, rmax, r, theta, phi);

  for (iL = 0; iL < 2; ++iL)
    {
      const size_t L = (iL == 0) ? 40 : 120;
      double * coef = test_random_coef(1, L, L);
      potsynth_parameters params = potsynth_default_parameters();
      potsynth_workspace * wd;
      potsynth_eval_workspace * ewd;
      size_t ie, ig;

      /* lithospheric-like spectrum for L = 120: flat instead of decaying */
      if (iL == 1)
        {
          size_t i;

          for (i = 0; i < potsynth_nlm(1, L, L); ++i)
            coef[i] = 2.0 * test_urand() - 1.0;
        }

      params.method = POTSYNTH_METHOD_DIRECT;
      wd = test_hess_alloc(1, L, L, coef, a, rmin, rmax, 1, &params);
      ewd = potsynth_eval_alloc(0, wd);
      potsynth_eval(n, r, theta, phi, D, ewd);

      for (ig = 0; ig < 2; ++ig)
        {
          const size_t nchan = (ig == 0) ? 5 : 8;

          for (ie = 0; ie < 3; ++ie)
            {
              potsynth_workspace * wf;
              potsynth_eval_workspace * ewf;
              size_t c, i;

              params.method = POTSYNTH_METHOD_FAST;
              params.eps = eps[ie];
              wf = test_hess_alloc(1, L, L, coef, a, rmin, rmax, (int) ig, &params);
              ps_test(wf == NULL, "hess: fast allocation L=%zu nchan=%zu eps=%.0e", L, nchan, eps[ie]);
              if (wf == NULL)
                continue;

              ewf = potsynth_eval_alloc(0, wf);
              potsynth_eval(n, r, theta, phi, F, ewf);

              for (c = 0; c < nchan; ++c)
                {
                  double emax = 0.0, vmax = 0.0, epole = 0.0;

                  for (i = 0; i < n; ++i)
                    {
                      const double e = fabs(F[c * n + i] - D[c * n + i]);

                      emax = fmax(emax, e);
                      vmax = fmax(vmax, fabs(D[c * n + i]));
                      if (theta[i] == 0.0 || theta[i] == M_PI)
                        epole = fmax(epole, e);
                    }

                  ps_test(emax / vmax > tol[ie] || epole / vmax > tol[ie],
                          "hess: fast vs direct L=%zu nchan=%zu eps=%.0e %s (error %.2e, poles %.2e)",
                          L, nchan, eps[ie], names[c], emax / vmax, epole / vmax);
                }

              potsynth_eval_free(ewf);
              potsynth_free(wf);
            }
        }

      potsynth_eval_free(ewd);
      potsynth_free(wd);
      free(coef);
    }

  free(r);
  free(theta);
  free(phi);
  free(D);
  free(F);
}

static void
test_hess(void)
{
  test_hess_ref("hess_L8_s11.bin");
  test_hess_ref("hess_L24_s12.bin");
  test_hess_fd();
  test_hess_poles();
  test_hess_groups();
  test_hess_fast();
}
