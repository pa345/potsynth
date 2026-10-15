/*
 * example_grav.c
 *
 * Evaluate a gravity field model (potential and gravitational acceleration)
 * at many scattered points with potsynth
 *
 * Usage: example_grav [file.gfc [lmax]]
 *
 * With an argument, the model is read from an ICGEM .gfc file (e.g. EGM2008,
 * EIGEN-6C4 or XGM2019e from https://icgem.gfz-potsdam.de), optionally
 * truncated at degree lmax. Without an argument, a synthetic Earth-like model
 * up to degree 360 is used.
 *
 * Build: cc -O2 example_grav.c -lpotsynth -lm -o example_grav
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <potsynth_grav.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* uniform random number in [0,1) */
static double
uniform(void)
{
  return rand() / (RAND_MAX + 1.0);
}

/* wall-clock time in seconds (C11) */
static double
now(void)
{
  struct timespec ts;

  timespec_get(&ts, TIME_UTC);

  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

/*
synthetic_model()
  Synthetic model: point mass, J2, and a Kaula spectrum (rms 1e-5/l^2 per
coefficient) for l >= 2; 4-pi fully normalized C_lm at potsynth_idx(l,m,0,lmax)
and S_lm at potsynth_idx(l,-m,0,lmax)

Inputs: lmax - maximum SH degree

Return: model, free with potsynth_gfc_free()
*/

static potsynth_gfc *
synthetic_model(const size_t lmax)
{
  potsynth_gfc * gfc = calloc(1, sizeof(potsynth_gfc));
  size_t l, m;

  gfc->lmin = 0;
  gfc->lmax = lmax;
  gfc->mmax = lmax;
  gfc->GM = 3.986004415e14;
  gfc->a = 6378136.3;
  strcpy(gfc->modelname, "synthetic");
  strcpy(gfc->tide_system, "unknown");
  gfc->coef = calloc(potsynth_nlm(0, lmax, lmax), sizeof(double));

  for (l = 2; l <= lmax; ++l)
    {
      const double sd = sqrt(3.0) * 1.0e-5 / ((double) l * l);

      for (m = 0; m <= l; ++m)
        {
          gfc->coef[potsynth_idx((long) l, (long) m, 0L, (long) lmax)] = sd * (2.0 * uniform() - 1.0);
          if (m > 0)
            gfc->coef[potsynth_idx((long) l, -(long) m, 0L, (long) lmax)] = sd * (2.0 * uniform() - 1.0);
        }
    }

  gfc->coef[potsynth_idx(0L, 0L, 0L, (long) lmax)] = 1.0;
  gfc->coef[potsynth_idx(2L, 0L, 0L, (long) lmax)] = -0.484165143790e-03;

  return gfc;
}

int
main(int argc, char * argv[])
{
  const double rmin = 6378137.0 + 200.0e3;      /* radial range of the points (m) */
  const double rmax = 6378137.0 + 500.0e3;
  const int quantities = POTSYNTH_GRAV_V | POTSYNTH_GRAV_G;
  const size_t nchan = potsynth_grav_nout(quantities);  /* V, g_r, g_theta, g_phi */
  const size_t n = 1000000;                     /* number of points */
  const size_t ndirect = 1000;                  /* points compared with the direct sum */
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * out = malloc(nchan * n * sizeof(double));
  double * outd = malloc(nchan * ndirect * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_gfc * gfc;
  potsynth_workspace * w, * wd;
  potsynth_eval_workspace * ew, * ewd;
  potsynth_info info;
  double t0;
  size_t c, i;
  int status;

  /* 1. coefficients: from a .gfc file, or synthetic */
  if (argc > 1)
    {
      const size_t lmax = (argc > 2) ? (size_t) atol(argv[2]) : 0; /* 0 = all degrees */

      gfc = potsynth_gfc_read(argv[1], lmax);
      if (gfc == NULL)
        exit(1); /* reported by the error handler */
    }
  else
    {
      gfc = synthetic_model(360);
    }

  fprintf(stderr, "model:       %s, degrees 0-%zu, GM = %.10e, a = %.4f, tide system %s\n",
          gfc->modelname, gfc->lmax, gfc->GM, gfc->a, gfc->tide_system);

  /* 2. evaluation points: random in the 200-500 km shell, colatitude in [0,pi] */
  for (i = 0; i < n; ++i)
    {
      r[i] = rmin + (rmax - rmin) * uniform();
      theta[i] = acos(2.0 * uniform() - 1.0);
      phi[i] = 2.0 * M_PI * uniform();
    }

  /* 3. set-up: done once per model and radial range */
  params.eps = 1.0e-10;   /* requested accuracy, relative to the size of each quantity */
  params.npts_hint = n;   /* lets METHOD_AUTO choose the faster method */

  t0 = now();
  w = potsynth_grav_alloc(gfc->lmin, gfc->lmax, gfc->mmax, gfc->coef, gfc->GM, gfc->a,
                          rmin, rmax, quantities, &params);
  if (w == NULL)
    exit(1);

  fprintf(stderr, "set-up:      %.3f s\n", now() - t0);

  potsynth_get_info(&info, w);
  fprintf(stderr, "method:      %s, window w = %d, radial rank = %zu, grid %.1f MB\n",
          (info.method == POTSYNTH_METHOD_FAST) ? "fast" : "direct", info.w, info.rho,
          info.grid_bytes / 1048576.0);

  /* 4. evaluation (r in m): out[c*n + i] = V, g_r, g_theta, g_phi */
  ew = potsynth_eval_alloc(0, w);

  t0 = now();
  status = potsynth_eval(n, r, theta, phi, out, ew);
  if (status)
    exit(1);

  fprintf(stderr, "evaluation:  %zu points in %.3f s\n", n, now() - t0);

  /* 5. compare the first points with the direct sum, quantity by quantity */
  params.method = POTSYNTH_METHOD_DIRECT;
  wd = potsynth_grav_alloc(gfc->lmin, gfc->lmax, gfc->mmax, gfc->coef, gfc->GM, gfc->a,
                           rmin, rmax, quantities, &params);
  ewd = potsynth_eval_alloc(0, wd);
  potsynth_eval(ndirect, r, theta, phi, outd, ewd);

  for (c = 0; c < nchan; ++c)
    {
      const char * names[] = { "V", "g_r", "g_theta", "g_phi" };
      double err = 0.0, vmax = 0.0;

      for (i = 0; i < ndirect; ++i)
        {
          err = fmax(err, fabs(out[c * n + i] - outd[c * ndirect + i]));
          vmax = fmax(vmax, fabs(outd[c * ndirect + i]));
        }

      fprintf(stderr, "fast vs direct (%zu points), %-7s: max difference %.2e (%.1e relative)\n",
              ndirect, names[c], err, err / vmax);
    }

  /* 6. first few points; the acceleration in mGal (1 mGal = 1e-5 m/s^2) */
  printf("     r [km]  colat [deg]  lon [deg]      V [m^2/s^2]      g_r [mGal]   g_theta [mGal]"
         "     g_phi [mGal]\n");
  for (i = 0; i < 4; ++i)
    {
      printf("%11.3f %12.4f %10.4f %16.4f %15.4f %16.4f %16.4f\n", r[i] / 1.0e3,
             theta[i] * 180.0 / M_PI, phi[i] * 180.0 / M_PI, out[i],
             out[n + i] * 1.0e5, out[2 * n + i] * 1.0e5, out[3 * n + i] * 1.0e5);
    }

  potsynth_eval_free(ewd);
  potsynth_free(wd);
  potsynth_eval_free(ew);
  potsynth_free(w);
  potsynth_gfc_free(gfc);
  free(r);
  free(theta);
  free(phi);
  free(out);
  free(outd);

  return 0;
}
