/*
 * example_mag.c
 *
 * Evaluate a lithospheric-type magnetic field model at many scattered points
 * with potsynth
 *
 * The model is synthetic (a dipole-dominated core field plus a flat
 * lithospheric spectrum up to degree 185, similar in size to CHAOS); replace
 * it with real Gauss coefficients in the same layout.
 *
 * Build: cc -O2 example_mag.c -lpotsynth -lm -o example_mag
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <potsynth_mag.h>

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

int
main(void)
{
  const size_t lmax = 185;                       /* maximum SH degree */
  const double a = 6371.2;                       /* reference radius (km) */
  const double rmin = a + 250.0;                 /* radial range of the points (km) */
  const double rmax = a + 800.0;
  const size_t n = 1000000;                      /* number of points */
  const size_t ndirect = 1000;                   /* points compared with the direct sum */
  double * gh = malloc(potsynth_nlm(1, lmax, lmax) * sizeof(double));
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * B = malloc(3 * n * sizeof(double));
  double * Bd = malloc(3 * ndirect * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w, * wd;
  potsynth_eval_workspace * ew, * ewd;
  potsynth_info info;
  double t0, err = 0.0, bmax = 0.0;
  size_t l, m, i, j;
  int status;

  /* 1. Gauss coefficients (nT): g_l^m at potsynth_idx(l,m,lmin,mmax), h_l^m at potsynth_idx(l,-m,lmin,mmax) */
  for (l = 1; l <= lmax; ++l)
    {
      /* rms per coefficient: core spectrum for l <= 13, flat lithospheric spectrum above */
      const double Rl = (l <= 13) ? 9.0e8 * pow(50.0 / 9.0e8, (l - 1) / 12.0) : 0.5;
      const double sd = sqrt(3.0 * Rl / ((l + 1.0) * (2.0 * l + 1.0)));

      for (m = 0; m <= l; ++m)
        {
          gh[potsynth_idx((long) l, (long) m, 1L, (long) lmax)] = sd * (2.0 * uniform() - 1.0);
          if (m > 0)
            gh[potsynth_idx((long) l, -(long) m, 1L, (long) lmax)] = sd * (2.0 * uniform() - 1.0);
        }
    }

  gh[potsynth_idx(1L, 0L, 1L, (long) lmax)] = -29400.0;

  /* 2. evaluation points: random in the 250-800 km shell, colatitude in [0,pi] */
  for (i = 0; i < n; ++i)
    {
      r[i] = rmin + (rmax - rmin) * uniform();
      theta[i] = acos(2.0 * uniform() - 1.0);
      phi[i] = 2.0 * M_PI * uniform();
    }

  /* 3. set-up: done once per model and radial range */
  params.eps = 1.0e-10; /* requested relative accuracy */

  t0 = now();
  w = potsynth_mag_alloc(1, lmax, lmax, gh, a, rmin, rmax, &params);
  if (w == NULL)
    {
      fprintf(stderr, "main: potsynth_mag_alloc failed\n");
      exit(1);
    }

  fprintf(stderr, "set-up:      %.3f s\n", now() - t0);

  potsynth_get_info(&info, w);
  fprintf(stderr, "method:      %s, window w = %d, radial rank = %zu, grid %.1f MB\n",
          (info.method == POTSYNTH_METHOD_FAST) ? "fast" : "direct", info.w, info.rho,
          info.grid_bytes / 1048576.0);

  /* 4. evaluation: one evaluation workspace per calling thread; no allocation inside */
  ew = potsynth_eval_alloc(0, w);

  t0 = now();
  status = potsynth_eval(n, r, theta, phi, B, ew);
  if (status)
    {
      fprintf(stderr, "main: potsynth_eval failed: %s\n", potsynth_strerror(status));
      exit(1);
    }

  fprintf(stderr, "evaluation:  %zu points in %.3f s\n", n, now() - t0);

  /* 5. compare the first points with the direct sum */
  params.method = POTSYNTH_METHOD_DIRECT;
  wd = potsynth_mag_alloc(1, lmax, lmax, gh, a, rmin, rmax, &params);
  ewd = potsynth_eval_alloc(0, wd);
  potsynth_eval(ndirect, r, theta, phi, Bd, ewd);

  for (i = 0; i < ndirect; ++i)
    {
      for (j = 0; j < 3; ++j)
        {
          err = fmax(err, fabs(B[j * n + i] - Bd[j * ndirect + i]));
          bmax = fmax(bmax, fabs(Bd[j * ndirect + i]));
        }
    }

  fprintf(stderr, "fast vs direct (%zu points): max difference %.2e nT (%.1e relative)\n",
          ndirect, err, err / bmax);

  /* 6. NEC components of the first few points */
  potsynth_mag_rtp2nec(n, B);

  printf("      r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]\n");
  for (i = 0; i < 4; ++i)
    {
      printf("%12.3f %12.4f %10.4f %13.4f %13.4f %13.4f\n", r[i], theta[i] * 180.0 / M_PI,
             phi[i] * 180.0 / M_PI, B[i], B[n + i], B[2 * n + i]);
    }

  potsynth_eval_free(ewd);
  potsynth_free(wd);
  potsynth_eval_free(ew);
  potsynth_free(w);
  free(gh);
  free(r);
  free(theta);
  free(phi);
  free(B);
  free(Bd);

  return 0;
}
