/*
 * test_tensor.c
 *
 * Gradient tensors of the front-ends (POTSYNTH_MAG_GRADB, POTSYNTH_GRAV_T)
 * and the tensor frame conversions:
 *
 * - gravity tensor against the 40-digit reference values (GM = a = 1, so the
 *   potential is that of tests/gen_hessian_ref.py), point mass closed form
 * - magnetic grad B against finite differences of B, and against the
 *   gravity front-end for the same potential (grad B = -T)
 * - B and grad B evaluated together equal separate models (direct, bitwise)
 * - fast method against direct method for both front-ends, poles included
 * - output counts, NEC conversion, and Cartesian components that do not
 *   depend on the longitude given for a polar point
 */

/* read a Hessian reference file (format of tests/gen_hessian_ref.py) */
static int
test_tensor_read(const char * file, size_t * L, size_t * n, double * a, double ** coef,
                 double ** r, double ** theta, double ** phi, double ** T)
{
  char filename[1024];
  int64_t hdr[2];
  size_t nlm;
  FILE * fp;
  int ok;

  snprintf(filename, sizeof(filename), "%s/%s", TEST_DATA_DIR, file);
  fp = fopen(filename, "rb");
  if (fp == NULL)
    return -1;

  ok = fread(hdr, sizeof(int64_t), 2, fp) == 2 && fread(a, sizeof(double), 1, fp) == 1;
  *L = (size_t) hdr[0];
  *n = (size_t) hdr[1];
  nlm = potsynth_nlm(0, *L, *L);
  *coef = malloc(nlm * sizeof(double));
  *r = malloc(*n * sizeof(double));
  *theta = malloc(*n * sizeof(double));
  *phi = malloc(*n * sizeof(double));
  *T = malloc(6 * *n * sizeof(double));
  ok = ok && fread(*coef, sizeof(double), nlm, fp) == nlm;
  ok = ok && fread(*r, sizeof(double), *n, fp) == *n;
  ok = ok && fread(*theta, sizeof(double), *n, fp) == *n;
  ok = ok && fread(*phi, sizeof(double), *n, fp) == *n;
  ok = ok && fread(*T, sizeof(double), 6 * *n, fp) == 6 * *n;
  fclose(fp);

  return ok ? 0 : -1;
}

/* gravity tensor: reference values, point mass */
static void
test_tensor_grav(void)
{
  size_t L, n, i, k;
  double a, * coef, * r, * theta, * phi, * Tref, * T;
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  potsynth_info info;
  double emax = 0.0, tmax = 0.0;

  ps_test(test_tensor_read("hess_L24_s12.bin", &L, &n, &a, &coef, &r, &theta, &phi, &Tref) != 0,
          "tensor: read reference");

  /* V = GM/r sum (a/r)^l ... with GM = a = 1 is the reference potential */
  params.method = POTSYNTH_METHOD_DIRECT;
  w = potsynth_grav_alloc(0, L, L, coef, 1.0, 1.0, 0.0, 0.0, POTSYNTH_GRAV_T, &params);
  potsynth_get_info(&info, w);
  ps_test_int((int) info.nchan, 5, "tensor: GRAV_T uses 5 channels");
  ps_test_int((int) info.nout, 6, "tensor: GRAV_T gives 6 outputs");

  T = malloc(6 * n * sizeof(double));
  ew = potsynth_eval_alloc(0, w);
  potsynth_eval(n, r, theta, phi, T, ew);

  for (k = 0; k < 6 * n; ++k)
    {
      emax = fmax(emax, fabs(T[k] - Tref[k]));
      tmax = fmax(tmax, fabs(Tref[k]));
    }

  ps_test(emax / tmax > 1.0e-13, "tensor: GRAV_T vs reference (error %.2e)", emax / tmax);

  potsynth_eval_free(ew);
  potsynth_free(w);

  /* point mass: T_rr = 2 GM/r^3, T_tt = T_pp = -GM/r^3, off-diagonal 0 */
  {
    const double GM = 3.986004415e14, ae = 6378136.3;
    double one = 1.0, rr[4] = { 6.4e6, 7.0e6, 6.4e6, 6.5e6 }, tt[4] = { 0.0, 1.0, M_PI, 2.0 };
    double pp[4] = { 0.0, 2.0, 1.0, 5.0 }, out[24];
    double e = 0.0;

    w = potsynth_grav_alloc(0, 0, 0, &one, GM, ae, 0.0, 0.0, POTSYNTH_GRAV_T, &params);
    ew = potsynth_eval_alloc(0, w);
    potsynth_eval(4, rr, tt, pp, out, ew);

    for (i = 0; i < 4; ++i)
      {
        const double g3 = GM / (rr[i] * rr[i] * rr[i]);
        const double ref[6] = { 2.0 * g3, 0.0, 0.0, -g3, 0.0, -g3 };

        for (k = 0; k < 6; ++k)
          e = fmax(e, fabs(out[k * 4 + i] - ref[k]) / g3);
      }

    ps_test(e > 1.0e-14, "tensor: point mass (error %.2e)", e);

    potsynth_eval_free(ew);
    potsynth_free(w);
  }

  /* V, g and T together: the first 4 outputs equal the V|G model (bitwise) */
  {
    double * F1 = malloc(10 * n * sizeof(double));
    double * F2 = malloc(4 * n * sizeof(double));
    potsynth_workspace * w2;
    potsynth_eval_workspace * ew2;

    w = potsynth_grav_alloc(0, L, L, coef, 1.0, 1.0, 0.0, 0.0,
                            POTSYNTH_GRAV_V | POTSYNTH_GRAV_G | POTSYNTH_GRAV_T, &params);
    w2 = potsynth_grav_alloc(0, L, L, coef, 1.0, 1.0, 0.0, 0.0, POTSYNTH_GRAV_V | POTSYNTH_GRAV_G, &params);
    ew = potsynth_eval_alloc(0, w);
    ew2 = potsynth_eval_alloc(0, w2);
    potsynth_eval(n, r, theta, phi, F1, ew);
    potsynth_eval(n, r, theta, phi, F2, ew2);

    ps_test(memcmp(F1, F2, 4 * n * sizeof(double)) != 0, "tensor: V, g unchanged with T (bitwise)");
    ps_test(memcmp(F1 + 4 * n, T, 6 * n * sizeof(double)) != 0, "tensor: T unchanged with V, g (bitwise)");
    ps_test_int((int) potsynth_grav_nout(POTSYNTH_GRAV_V | POTSYNTH_GRAV_G | POTSYNTH_GRAV_T), 10,
                "tensor: grav nout V|G|T");

    potsynth_eval_free(ew);
    potsynth_eval_free(ew2);
    potsynth_free(w);
    potsynth_free(w2);
    free(F1);
    free(F2);
  }

  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(Tref);
  free(T);
}

/* magnetic grad B: finite differences of B, consistency with gravity, B unchanged */
static void
test_tensor_mag(void)
{
  const double a = 6371.2, rmin = a + 250.0, rmax = a + 800.0;
  const size_t L = 60, n = 500;
  const size_t nlm = potsynth_nlm(1, L, L);
  double * gh = test_random_coef(1, L, L);
  double * coef = malloc(nlm * sizeof(double));
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * BT = malloc(9 * n * sizeof(double));
  double * B = malloc(3 * n * sizeof(double));
  double * Tg = malloc(6 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w, * wb, * wg;
  potsynth_eval_workspace * ew, * ewb, * ewg;
  size_t i, k, l, m;
  double e, tmax;

  test_random_points(n, rmin, rmax, r, theta, phi);
  params.method = POTSYNTH_METHOD_DIRECT;

  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B | POTSYNTH_MAG_GRADB, &params);
  wb = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B, &params);
  ew = potsynth_eval_alloc(0, w);
  ewb = potsynth_eval_alloc(0, wb);
  potsynth_eval(n, r, theta, phi, BT, ew);
  potsynth_eval(n, r, theta, phi, B, ewb);
  ps_test(memcmp(BT, B, 3 * n * sizeof(double)) != 0, "tensor: B unchanged with grad B (bitwise)");
  ps_test_int((int) potsynth_mag_nout(POTSYNTH_MAG_B | POTSYNTH_MAG_GRADB), 9, "tensor: mag nout B|GRADB");
  ps_test_int((int) potsynth_mag_nout(POTSYNTH_MAG_GRADB), 6, "tensor: mag nout GRADB");
  ps_test_int((int) potsynth_mag_nout(0x4), 0, "tensor: mag nout invalid flag");

  /* same potential with the gravity front-end: C = g/sqrt(2l+1), GM = a^2 -> grad B = -T */
  for (l = 1; l <= L; ++l)
    {
      for (m = 0; m <= l; ++m)
        {
          const size_t ip = potsynth_idx((long) l, (long) m, 1L, (long) L);
          const size_t im = potsynth_idx((long) l, -(long) m, 1L, (long) L);

          coef[ip] = gh[ip] / sqrt(2.0 * l + 1.0);
          if (m > 0)
            coef[im] = gh[im] / sqrt(2.0 * l + 1.0);
        }
    }

  wg = potsynth_grav_alloc(1, L, L, coef, a * a, a, rmin, rmax, POTSYNTH_GRAV_T, &params);
  ewg = potsynth_eval_alloc(0, wg);
  potsynth_eval(n, r, theta, phi, Tg, ewg);

  e = 0.0;
  tmax = 0.0;
  for (k = 0; k < 6 * n; ++k)
    {
      e = fmax(e, fabs(BT[3 * n + k] + Tg[k]));
      tmax = fmax(tmax, fabs(Tg[k]));
    }

  ps_test(e / tmax > 1.0e-13, "tensor: grad B = -T of the gravity front-end (error %.2e)", e / tmax);

  /* finite differences of B (covariant derivative formulas, B any vector field) */
  {
    const double hr = 1.0e-3, ht = 2.0e-6;
    double Fd[3 * 7];

    e = 0.0;
    tmax = 0.0;
    for (i = 20; i < 70; ++i)
      {
        const double r0 = r[i], t0 = theta[i], p0 = phi[i];
        const double st = sin(t0), ct = cos(t0);
        double rr[7] = { r0, r0 + hr, r0 - hr, r0, r0, r0, r0 };
        double tt[7] = { t0, t0, t0, t0 + ht, t0 - ht, t0, t0 };
        double pp[7] = { p0, p0, p0, p0, p0, p0 + ht, p0 - ht };
        double fd[6];

        if (st < 0.2)
          continue;

        potsynth_eval(7, rr, tt, pp, Fd, ewb);

#define TEST_TENSOR_B(c, j) Fd[(c) * 7 + (j)]
        fd[0] = (TEST_TENSOR_B(0, 1) - TEST_TENSOR_B(0, 2)) / (2.0 * hr);
        fd[1] = (TEST_TENSOR_B(1, 1) - TEST_TENSOR_B(1, 2)) / (2.0 * hr);
        fd[2] = (TEST_TENSOR_B(2, 1) - TEST_TENSOR_B(2, 2)) / (2.0 * hr);
        fd[3] = ((TEST_TENSOR_B(1, 3) - TEST_TENSOR_B(1, 4)) / (2.0 * ht) + TEST_TENSOR_B(0, 0)) / r0;
        fd[4] = (TEST_TENSOR_B(1, 5) - TEST_TENSOR_B(1, 6)) / (2.0 * ht * r0 * st) -
                TEST_TENSOR_B(2, 0) * ct / (st * r0);
        fd[5] = (TEST_TENSOR_B(2, 5) - TEST_TENSOR_B(2, 6)) / (2.0 * ht * r0 * st) +
                TEST_TENSOR_B(0, 0) / r0 + TEST_TENSOR_B(1, 0) * ct / (st * r0);
#undef TEST_TENSOR_B

        for (k = 0; k < 6; ++k)
          {
            e = fmax(e, fabs(BT[(3 + k) * n + i] - fd[k]));
            tmax = fmax(tmax, fabs(BT[(3 + k) * n + i]));
          }
      }

    ps_test(e / tmax > 1.0e-7, "tensor: grad B vs finite differences of B (error %.2e)", e / tmax);
  }

  potsynth_eval_free(ew);
  potsynth_eval_free(ewb);
  potsynth_eval_free(ewg);
  potsynth_free(w);
  potsynth_free(wb);
  potsynth_free(wg);
  free(gh);
  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(BT);
  free(B);
  free(Tg);
}

/* fast method against direct method for both front-ends, every output */
static void
test_tensor_fast(void)
{
  const double a = 6371.2, rmin = a + 250.0, rmax = a + 800.0;
  const size_t L = 90, n = 2000;
  double * gh = test_random_coef(1, L, L);
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * D = malloc(10 * n * sizeof(double));
  double * F = malloc(10 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  size_t f;

  test_random_points(n, rmin, rmax, r, theta, phi);

  for (f = 0; f < 2; ++f)
    {
      const size_t nout = (f == 0) ? 9 : 10;
      potsynth_workspace * wd, * wf;
      potsynth_eval_workspace * ewd, * ewf;
      double emax = 0.0;
      size_t k, i;

      params.method = POTSYNTH_METHOD_DIRECT;
      params.eps = 1.0e-10;
      if (f == 0)
        wd = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B | POTSYNTH_MAG_GRADB, &params);
      else
        wd = potsynth_grav_alloc(1, L, L, gh, 3.986004415e14, 6378136.3, rmin * 1.0e3, rmax * 1.0e3,
                                 POTSYNTH_GRAV_V | POTSYNTH_GRAV_G | POTSYNTH_GRAV_T, &params);

      params.method = POTSYNTH_METHOD_FAST;
      if (f == 0)
        wf = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, POTSYNTH_MAG_B | POTSYNTH_MAG_GRADB, &params);
      else
        wf = potsynth_grav_alloc(1, L, L, gh, 3.986004415e14, 6378136.3, rmin * 1.0e3, rmax * 1.0e3,
                                 POTSYNTH_GRAV_V | POTSYNTH_GRAV_G | POTSYNTH_GRAV_T, &params);

      ewd = potsynth_eval_alloc(0, wd);
      ewf = potsynth_eval_alloc(0, wf);

      if (f == 1)
        {
          for (i = 0; i < n; ++i)
            r[i] *= 1.0e3;
        }

      potsynth_eval(n, r, theta, phi, D, ewd);
      potsynth_eval(n, r, theta, phi, F, ewf);

      if (f == 1)
        {
          for (i = 0; i < n; ++i)
            r[i] *= 1.0e-3;
        }

      for (k = 0; k < nout; ++k)
        {
          double e = 0.0, vmax = 0.0;

          for (i = 0; i < n; ++i)
            {
              e = fmax(e, fabs(F[k * n + i] - D[k * n + i]));
              vmax = fmax(vmax, fabs(D[k * n + i]));
            }

          /* off-diagonal and difference outputs: relative to the tensor size */
          emax = fmax(emax, e / vmax);
        }

      ps_test(emax > 1.0e-9, "tensor: %s fast vs direct, all %zu outputs (error %.2e)",
              (f == 0) ? "mag B|GRADB" : "grav V|G|T", nout, emax);

      potsynth_eval_free(ewd);
      potsynth_eval_free(ewf);
      potsynth_free(wd);
      potsynth_free(wf);
    }

  free(gh);
  free(r);
  free(theta);
  free(phi);
  free(D);
  free(F);
}

/* frame conversions */
static void
test_tensor_frames(void)
{
  const double a = 1.0;
  size_t L, n, i, k;
  double ar, * coef, * r, * theta, * phi, * Tref;
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;

  /* NEC mapping on known values */
  {
    double T[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; /* rr rt rp tt tp pp */
    const double nec[6] = { 4.0, -5.0, 2.0, 6.0, -3.0, 1.0 }; /* NN NE NC EE EC CC */
    int ok = 1;

    potsynth_tensor_rtp2nec(1, T);
    for (k = 0; k < 6; ++k)
      ok = ok && (T[k] == nec[k]);

    ps_test(!ok, "tensor: rtp2nec mapping");
  }

  if (test_tensor_read("hess_L8_s11.bin", &L, &n, &ar, &coef, &r, &theta, &phi, &Tref) != 0)
    {
      ps_test(1, "tensor: read reference for frames");
      return;
    }

  /* the reference starts with 4 longitudes x (0, 1e-12, 1e-6, pi/2, pi - 1e-12, pi):
   * Cartesian components at theta = 0 (and pi) must not depend on phi */
  {
    double rr[8], tt[8], pp[8], out[48], ec = 0.0, tmax = 0.0;
    const double phis[4] = { 0.0, 1.0, 2.5, -2.0 };

    for (i = 0; i < 4; ++i)
      {
        rr[i] = rr[4 + i] = 1.2;
        tt[i] = 0.0;
        tt[4 + i] = M_PI;
        pp[i] = pp[4 + i] = phis[i];
      }

    params.method = POTSYNTH_METHOD_DIRECT;
    w = potsynth_grav_alloc(0, L, L, coef, 1.0, a, 0.0, 0.0, POTSYNTH_GRAV_T, &params);
    ew = potsynth_eval_alloc(0, w);
    potsynth_eval(8, rr, tt, pp, out, ew);
    potsynth_tensor_rtp2xyz(8, tt, pp, out);

    for (k = 0; k < 6; ++k)
      {
        for (i = 1; i < 4; ++i)
          {
            ec = fmax(ec, fabs(out[k * 8 + i] - out[k * 8]));
            ec = fmax(ec, fabs(out[k * 8 + 4 + i] - out[k * 8 + 4]));
          }

        for (i = 0; i < 8; ++i)
          tmax = fmax(tmax, fabs(out[k * 8 + i]));
      }

    ps_test(ec / tmax > 1.0e-14, "tensor: Cartesian components at the poles independent of phi (%.2e)",
            ec / tmax);

    /* trace and Frobenius norm are frame invariants (reference, all points) */
    {
      double * X = malloc(6 * n * sizeof(double));
      double einv = 0.0;

      memcpy(X, Tref, 6 * n * sizeof(double));
      potsynth_tensor_rtp2xyz(n, theta, phi, X);

      for (i = 0; i < n; ++i)
        {
          const double * s = Tref + i;
          const double * x = X + i;
          const double fs = s[0] * s[0] + s[3 * n] * s[3 * n] + s[5 * n] * s[5 * n] +
                            2.0 * (s[n] * s[n] + s[2 * n] * s[2 * n] + s[4 * n] * s[4 * n]);
          const double fx = x[0] * x[0] + x[3 * n] * x[3 * n] + x[5 * n] * x[5 * n] +
                            2.0 * (x[n] * x[n] + x[2 * n] * x[2 * n] + x[4 * n] * x[4 * n]);

          einv = fmax(einv, fabs(x[0] + x[3 * n] + x[5 * n]) / sqrt(fs));
          einv = fmax(einv, fabs(fx - fs) / fs);
        }

      ps_test(einv > 1.0e-14, "tensor: rtp2xyz keeps trace and norm (%.2e)", einv);
      free(X);
    }

    potsynth_eval_free(ew);
    potsynth_free(w);
  }

  free(coef);
  free(r);
  free(theta);
  free(phi);
  free(Tref);
}

static void
test_tensor(void)
{
  test_tensor_grav();
  test_tensor_mag();
  test_tensor_fast();
  test_tensor_frames();
}
