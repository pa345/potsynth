/*
 * example_mag.c -- evaluate a lithospheric-type magnetic field model at many
 * scattered points with potsynth
 *
 * The model is synthetic (a dipole-dominated core field plus a flat
 * lithospheric spectrum up to degree 185, similar in size to CHAOS); replace
 * it with real Gauss coefficients in the same layout.
 *
 * Build:  cc -O2 example_mag.c -lpotsynth -lm -o example_mag
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <potsynth_mag.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int
main(void)
{
  const size_t L = 185;                         /* maximum degree */
  const double a = 6371.2;                      /* reference radius (km) */
  const double rmin = a + 250.0, rmax = a + 800.0;
  const size_t npts = 1000000;
  const size_t ncoef = potsynth_ncoef(1, L, L);
  double *gh = malloc(ncoef * sizeof(double));
  double *r = malloc(npts * sizeof(double));
  double *theta = malloc(npts * sizeof(double));
  double *phi = malloc(npts * sizeof(double));
  double *B = malloc(3 * npts * sizeof(double));
  double *Bd = malloc(3 * 1000 * sizeof(double));
  potsynth_params params;
  potsynth_workspace *w, *wd;
  potsynth_eval_workspace *ew, *ewd;
  potsynth_info info;
  double t0, err = 0.0, bmax = 0.0;
  size_t n, m, i;
  int status;

  /* 1. Gauss coefficients g_n^m, h_n^m (nT), green2 layout:
   *    gh[POTSYNTH_IDX(n, m, lmin, mmax)] = g_n^m, gh[POTSYNTH_IDX(n, -m, lmin, mmax)] = h_n^m */
  for (n = 1; n <= L; ++n)
    {
      /* rms per coefficient: core spectrum for n <= 13, flat lithospheric spectrum above */
      const double Rn = (n <= 13) ? 9e8 * pow(50.0 / 9e8, (n - 1) / 12.0) : 0.5;
      const double sd = sqrt(Rn / ((n + 1.0) * (2.0 * n + 1.0)));

      for (m = 0; m <= n; ++m)
        {
          gh[POTSYNTH_IDX(n, m, 1, L)] = sd * (2.0 * uniform() - 1.0) * sqrt(3.0);
          if (m > 0)
            gh[POTSYNTH_IDX(n, -(long) m, 1, L)] = sd * (2.0 * uniform() - 1.0) * sqrt(3.0);
        }
    }
  gh[POTSYNTH_IDX(1, 0, 1, L)] = -29400.0;

  /* 2. evaluation points: random in the 250-800 km shell, colatitude in [0, pi] */
  for (i = 0; i < npts; ++i)
    {
      r[i] = rmin + (rmax - rmin) * uniform();
      theta[i] = acos(2.0 * uniform() - 1.0);
      phi[i] = 2.0 * M_PI * uniform();
    }

  /* 3. set-up: done once per model and radial range */
  potsynth_params_init(&params);
  params.eps = 1e-10;                           /* requested relative accuracy */

  t0 = now();
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &params, &status);
  if (!w)
    {
      fprintf(stderr, "potsynth_mag_alloc: %s\n", potsynth_strerror(status));
      return 1;
    }
  printf("set-up:      %.3f s\n", now() - t0);

  potsynth_get_info(w, &info);
  printf("method:      %s, window w = %d, radial rank = %zu, grid %.1f MB\n",
         info.method == POTSYNTH_METHOD_FAST ? "fast" : "direct", info.w, info.rho,
         info.grid_bytes / 1048576.0);

  /* 4. evaluation: one eval workspace per calling thread; no allocation inside */
  ew = potsynth_eval_alloc(w, 0);
  t0 = now();
  status = potsynth_eval(w, ew, npts, r, theta, phi, B);
  if (status)
    {
      fprintf(stderr, "potsynth_eval: %s\n", potsynth_strerror(status));
      return 1;
    }
  printf("evaluation:  %zu points in %.3f s\n", npts, now() - t0);

  /* 5. compare the first 1000 points with the direct sum */
  params.method = POTSYNTH_METHOD_DIRECT;
  wd = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &params, &status);
  ewd = potsynth_eval_alloc(wd, 0);
  potsynth_eval(wd, ewd, 1000, r, theta, phi, Bd);
  for (i = 0; i < 1000; ++i)
    for (n = 0; n < 3; ++n)
      {
        err = fmax(err, fabs(B[n * npts + i] - Bd[n * 1000 + i]));
        bmax = fmax(bmax, fabs(Bd[n * 1000 + i]));
      }
  printf("fast vs direct (1000 points): max difference %.2e nT (%.1e relative)\n", err,
         err / bmax);

  /* 6. NEC components of the first few points */
  potsynth_mag_rtp2nec(npts, B);
  printf("\n      r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]\n");
  for (i = 0; i < 4; ++i)
    printf("%12.3f %12.4f %10.4f %13.4f %13.4f %13.4f\n", r[i], theta[i] * 180.0 / M_PI,
           phi[i] * 180.0 / M_PI, B[i], B[npts + i], B[2 * npts + i]);

  potsynth_eval_free(ewd);
  potsynth_free(wd);
  potsynth_eval_free(ew);
  potsynth_free(w);
  free(gh); free(r); free(theta); free(phi); free(B); free(Bd);
  return 0;
}
