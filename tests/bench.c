/*
 * bench.c -- timing of set-up and evaluation (magnetic front-end)
 *
 * Usage: bench L M eps [nthreads] [ndirect]
 *   L        maximum degree (random CHAOS-like model)
 *   M        number of random points in the 250-800 km shell
 *   eps      requested accuracy of the fast method
 *   nthreads 0 = all
 *   ndirect  points for timing the direct method (scaled to M); 0 = skip
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "potsynth_mag.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double
now(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* xorshift uniform in [0,1) */
static double
urand(unsigned long long *s)
{
  *s ^= *s << 13;
  *s ^= *s >> 7;
  *s ^= *s << 17;
  return (double) (*s >> 11) * (1.0 / 9007199254740992.0);
}

static double
nrand(unsigned long long *s)
{
  const double u1 = urand(s) + 1e-300, u2 = urand(s);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int
main(int argc, char **argv)
{
  const double a = 6371.2, rmin = a + 250.0, rmax = a + 800.0;
  const size_t L = (argc > 1) ? (size_t) atol(argv[1]) : 185;
  const size_t M = (argc > 2) ? (size_t) atol(argv[2]) : 1000000;
  const double eps = (argc > 3) ? atof(argv[3]) : 1e-10;
  const int nthreads = (argc > 4) ? atoi(argv[4]) : 0;
  const size_t ndirect = (argc > 5) ? (size_t) atol(argv[5]) : 20000;
  const size_t ncoef = potsynth_ncoef(1, L, L);
  unsigned long long seed = 88172645463325252ULL;
  double *gh = malloc(ncoef * sizeof(double));
  double *r = malloc(M * sizeof(double)), *th = malloc(M * sizeof(double));
  double *ph = malloc(M * sizeof(double)), *B = malloc(3 * M * sizeof(double));
  double t0, tset, teval = 1e30, tdir = 0.0;
  potsynth_params p;
  potsynth_workspace *w;
  potsynth_eval_workspace *ew;
  potsynth_info info;
  size_t n, m, i;
  int status, rep;

  for (n = 1; n <= L; ++n)
    {
      const double Rn = (n <= 13) ? 9e8 * pow(50.0 / 9e8, (n - 1) / 12.0) : ((n <= 15) ? 5.0 : 0.5);
      const double sd = sqrt(Rn / ((n + 1.0) * (2.0 * n + 1.0)));

      for (m = 0; m <= n; ++m)
        {
          gh[POTSYNTH_IDX((long) n, (long) m, 1L, (long) L)] = sd * nrand(&seed);
          if (m > 0)
            gh[POTSYNTH_IDX((long) n, -(long) m, 1L, (long) L)] = sd * nrand(&seed);
        }
    }

  for (i = 0; i < M; ++i)
    {
      r[i] = rmin + (rmax - rmin) * urand(&seed);
      th[i] = acos(2.0 * urand(&seed) - 1.0);
      ph[i] = 2.0 * M_PI * urand(&seed);
    }

  potsynth_params_init(&p);
  p.method = POTSYNTH_METHOD_FAST;
  p.eps = eps;
  p.nthreads = nthreads;

  t0 = now();
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &status);
  tset = now() - t0;
  if (!w)
    {
      printf("alloc failed: %s\n", potsynth_strerror(status));
      return 1;
    }

  potsynth_get_info(w, &info);
  ew = potsynth_eval_alloc(w, 0);

  for (rep = 0; rep < 3; ++rep)
    {
      t0 = now();
      potsynth_eval(w, ew, M, r, th, ph, B);
      teval = fmin(teval, now() - t0);
    }

  if (ndirect > 0)
    {
      potsynth_workspace *wd;
      potsynth_eval_workspace *ewd;
      const size_t nd = (ndirect < M) ? ndirect : M;

      p.method = POTSYNTH_METHOD_DIRECT;
      wd = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &status);
      ewd = potsynth_eval_alloc(wd, 0);
      t0 = now();
      potsynth_eval(wd, ewd, nd, r, th, ph, B);
      tdir = (now() - t0) * (double) M / (double) nd;
      potsynth_eval_free(ewd);
      potsynth_free(wd);
    }

  printf("L=%zu M=%zu eps=%.0e threads=%d: w=%d rho=%zu grid=%.1f MB  set-up %.3f s  "
         "eval %.3f s (%.3f us/pt)",
         L, M, eps, nthreads, info.w, info.rho, info.grid_bytes / 1048576.0, tset, teval,
         1e6 * teval / M);
  if (ndirect > 0)
    printf("  direct %.3f s  speedup %.1fx", tdir, tdir / teval);
  printf("\n");

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(gh); free(r); free(th); free(ph); free(B);
  return 0;
}
