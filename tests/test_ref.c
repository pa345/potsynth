/*
 * test_ref.c
 *
 * Magnetic front-end against extended precision reference values
 * (tests/gen_reference.py), for the direct method and the fast method at
 * several accuracies. The point sets include both poles at several longitudes,
 * points 1e-300 .. 1e-6 from the poles, r = rmin and rmax, and longitudes
 * outside [0, 2pi).
 *
 * File format (little endian): int64 lmax, npts; float64 a, rmin, rmax;
 * float64 gh[nlm] (lmin = 1, mmax = lmax); float64 r, theta, phi [npts];
 * float64 B_r, B_theta, B_phi [npts]
 */

typedef struct
{
  size_t lmax;          /* maximum SH degree */
  size_t n;             /* number of points */
  double a;             /* reference radius */
  double rmin;          /* radial range of points */
  double rmax;
  double * gh;          /* Gauss coefficients */
  double * r;           /* points */
  double * theta;
  double * phi;
  double * B;           /* reference field, channel-major */
} test_ref_data;

/* read reference data file; returns 0 on success */
static int
test_ref_read(const char * filename, test_ref_data * data)
{
  FILE * fp = fopen(filename, "rb");
  int64_t hdr[2];
  double par[3];
  size_t nlm, n;
  int ok = 1;

  if (fp == NULL)
    return -1;

  ok = ok && fread(hdr, sizeof(int64_t), 2, fp) == 2;
  ok = ok && fread(par, sizeof(double), 3, fp) == 3;
  if (!ok)
    {
      fclose(fp);
      return -1;
    }

  data->lmax = (size_t) hdr[0];
  data->n = n = (size_t) hdr[1];
  data->a = par[0];
  data->rmin = par[1];
  data->rmax = par[2];
  nlm = potsynth_nlm(1, data->lmax, data->lmax);

  data->gh = malloc(nlm * sizeof(double));
  data->r = malloc(n * sizeof(double));
  data->theta = malloc(n * sizeof(double));
  data->phi = malloc(n * sizeof(double));
  data->B = malloc(3 * n * sizeof(double));

  ok = ok && fread(data->gh, sizeof(double), nlm, fp) == nlm;
  ok = ok && fread(data->r, sizeof(double), n, fp) == n;
  ok = ok && fread(data->theta, sizeof(double), n, fp) == n;
  ok = ok && fread(data->phi, sizeof(double), n, fp) == n;
  ok = ok && fread(data->B, sizeof(double), 3 * n, fp) == 3 * n;
  fclose(fp);

  return ok ? 0 : -1;
}

static void
test_ref_free(test_ref_data * data)
{
  free(data->gh);
  free(data->r);
  free(data->theta);
  free(data->phi);
  free(data->B);
}

/* evaluate the model with given method and accuracy, compare with reference */
static void
test_ref_method(const char * name, const int method, const double eps, const double tol,
                const test_ref_data * data)
{
  potsynth_parameters params = potsynth_default_parameters();
  double * B = malloc(3 * data->n * sizeof(double));
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  double emax = 0.0, bmax = 0.0;
  size_t i, nbad = 0;
  int status;

  params.method = method;
  params.eps = eps;

  w = potsynth_mag_alloc(1, data->lmax, data->lmax, data->gh, data->a, data->rmin, data->rmax, POTSYNTH_MAG_B, &params);
  ps_test(w == NULL, "ref: %s method=%d eps=%g allocation", name, method, eps);
  if (w == NULL)
    {
      free(B);
      return;
    }

  ew = potsynth_eval_alloc(0, w);
  status = potsynth_eval(data->n, data->r, data->theta, data->phi, B, ew);
  ps_test_int(status, POTSYNTH_SUCCESS, "ref: %s method=%d eps=%g evaluation", name, method, eps);

  for (i = 0; i < 3 * data->n; ++i)
    {
      if (!isfinite(B[i]))
        ++nbad;

      emax = fmax(emax, fabs(B[i] - data->B[i]));
      bmax = fmax(bmax, fabs(data->B[i]));
    }

  ps_test(nbad != 0, "ref: %s method=%d eps=%g finite outputs (%zu non-finite)", name, method, eps, nbad);
  ps_test(emax / bmax > tol, "ref: %s method=%d eps=%g (error %.2e relative to max |B|)",
          name, method, eps, emax / bmax);

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(B);
}

static void
test_ref(void)
{
  const char * files[] = { "ref_L13_s1.bin", "ref_L60_s2.bin", "ref_L185_s3.bin", "ref_L185_s4.bin" };
  size_t i;

  for (i = 0; i < sizeof(files) / sizeof(files[0]); ++i)
    {
      char filename[1024];
      test_ref_data data;
      int status;

      snprintf(filename, sizeof(filename), "%s/%s", TEST_DATA_DIR, files[i]);

      status = test_ref_read(filename, &data);
      ps_test(status, "ref: reading %s", filename);
      if (status)
        continue;

      test_ref_method(files[i], POTSYNTH_METHOD_DIRECT, 1.0e-10, 1.0e-13, &data);
      test_ref_method(files[i], POTSYNTH_METHOD_FAST, 1.0e-6, 1.0e-6, &data);
      test_ref_method(files[i], POTSYNTH_METHOD_FAST, 1.0e-10, 1.0e-10, &data);
      test_ref_method(files[i], POTSYNTH_METHOD_FAST, 1.0e-14, 1.0e-13, &data);

      test_ref_free(&data);
    }
}
