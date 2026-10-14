/*
 * bench.c
 *
 * Timing of set-up and evaluation (magnetic front-end)
 *
 * Usage: bench lmax n eps [nthreads] [ndirect]
 *
 * lmax     - maximum degree (random CHAOS-like model)
 * n        - number of random points in the 250-800 km shell
 * eps      - requested accuracy of the fast method
 * nthreads - number of threads (0 = all)
 * ndirect  - points for timing the direct method (scaled to n); 0 = skip
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "potsynth_mag.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned long long rng_state = 88172645463325252ULL;

/* wall-clock time in seconds (C11) */
static double
now(void)
{
  struct timespec ts;

  timespec_get(&ts, TIME_UTC);

  return ts.tv_sec + 1.0e-9 * ts.tv_nsec;
}

/* uniform random number in [0,1) (xorshift) */
static double
urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;

  return (double) (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* standard normal random number (Box-Muller) */
static double
nrand(void)
{
  const double u1 = urand() + 1.0e-300;
  const double u2 = urand();

  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int
main(int argc, char * argv[])
{
  const double a = 6371.2;
  const double rmin = a + 250.0;
  const double rmax = a + 800.0;
  const size_t lmax = (argc > 1) ? (size_t) atol(argv[1]) : 185;
  const size_t n = (argc > 2) ? (size_t) atol(argv[2]) : 1000000;
  const double eps = (argc > 3) ? atof(argv[3]) : 1.0e-10;
  const int nthreads = (argc > 4) ? atoi(argv[4]) : 0;
  const size_t ndirect = (argc > 5) ? (size_t) atol(argv[5]) : 20000;
  double * gh = malloc(potsynth_nlm(1, lmax, lmax) * sizeof(double));
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * B = malloc(3 * n * sizeof(double));
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_workspace * w;
  potsynth_eval_workspace * ew;
  potsynth_info info;
  double t0, t_setup, t_eval = 1.0e30, t_direct = 0.0;
  size_t l, m, i;
  int k;

  for (l = 1; l <= lmax; ++l)
    {
      const double Rl = (l <= 13) ? 9.0e8 * pow(50.0 / 9.0e8, (l - 1) / 12.0) : ((l <= 15) ? 5.0 : 0.5);
      const double sd = sqrt(Rl / ((l + 1.0) * (2.0 * l + 1.0)));

      for (m = 0; m <= l; ++m)
        {
          gh[potsynth_idx((long) l, (long) m, 1L, (long) lmax)] = sd * nrand();
          if (m > 0)
            gh[potsynth_idx((long) l, -(long) m, 1L, (long) lmax)] = sd * nrand();
        }
    }

  for (i = 0; i < n; ++i)
    {
      r[i] = rmin + (rmax - rmin) * urand();
      theta[i] = acos(2.0 * urand() - 1.0);
      phi[i] = 2.0 * M_PI * urand();
    }

  params.method = POTSYNTH_METHOD_FAST;
  params.eps = eps;
  params.nthreads = nthreads;

  t0 = now();
  w = potsynth_mag_alloc(1, lmax, lmax, gh, a, rmin, rmax, &params);
  t_setup = now() - t0;
  if (w == NULL)
    {
      fprintf(stderr, "main: potsynth_mag_alloc failed\n");
      exit(1);
    }

  potsynth_get_info(&info, w);
  ew = potsynth_eval_alloc(0, w);

  for (k = 0; k < 3; ++k)
    {
      t0 = now();
      potsynth_eval(n, r, theta, phi, B, ew);
      t_eval = fmin(t_eval, now() - t0);
    }

  if (ndirect > 0)
    {
      const size_t nd = (ndirect < n) ? ndirect : n;
      potsynth_workspace * wd;
      potsynth_eval_workspace * ewd;

      params.method = POTSYNTH_METHOD_DIRECT;
      wd = potsynth_mag_alloc(1, lmax, lmax, gh, a, rmin, rmax, &params);
      ewd = potsynth_eval_alloc(0, wd);

      t0 = now();
      potsynth_eval(nd, r, theta, phi, B, ewd);
      t_direct = (now() - t0) * (double) n / (double) nd;

      potsynth_eval_free(ewd);
      potsynth_free(wd);
    }

  printf("L=%zu M=%zu eps=%.0e threads=%d: w=%d rho=%zu grid=%.1f MB  set-up %.3f s  "
         "eval %.3f s (%.3f us/pt)",
         lmax, n, eps, nthreads, info.w, info.rho, info.grid_bytes / 1048576.0, t_setup, t_eval,
         1.0e6 * t_eval / n);

  if (ndirect > 0)
    printf("  direct %.3f s  speedup %.1fx", t_direct, t_direct / t_eval);

  printf("\n");

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(gh);
  free(r);
  free(theta);
  free(phi);
  free(B);

  return 0;
}
