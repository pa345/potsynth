/*
 * test_threads.c -- one shared workspace evaluated concurrently from several
 * threads, each with its own eval workspace; results must equal the serial
 * results bitwise. Also run under ThreadSanitizer (tests/check.sh).
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_mag.h"

#ifdef POTSYNTH_TEST_PTHREADS
#include <pthread.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NTHREADS 8

typedef struct
{
  const potsynth_workspace *w;
  size_t npts, chunk;
  const double *r, *th, *ph;
  double *B;
} job;

/* each thread: its own eval workspace and chunk size, all sharing w */
static void *
worker(void *arg)
{
  job *j = arg;
  potsynth_eval_workspace *et = potsynth_eval_alloc(j->w, j->chunk);
  int k;

  for (k = 0; k < 3; ++k)
    potsynth_eval(j->w, et, j->npts, j->r, j->th, j->ph, j->B);
  potsynth_eval_free(et);
  return NULL;
}

static int
run(potsynth_method method)
{
  const double a = 6371.2, rmin = a + 250.0, rmax = a + 800.0;
  const size_t L = 60, npts = 20000, ncoef = potsynth_ncoef(1, L, L);
  double *gh = malloc(ncoef * sizeof(double));
  double *r = malloc(npts * sizeof(double)), *th = malloc(npts * sizeof(double));
  double *ph = malloc(npts * sizeof(double)), *Bs = malloc(3 * npts * sizeof(double));
  double *Bt = malloc(NTHREADS * 3 * npts * sizeof(double));
  unsigned long long seed = 12345;
  potsynth_params p;
  potsynth_workspace *w;
  potsynth_eval_workspace *ew;
  int status, fail = 0, t;
  size_t i;

  for (i = 0; i < ncoef; ++i)
    {
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      gh[i] = ((double) (seed >> 11) / 9007199254740992.0 - 0.5) * 100.0;
    }
  for (i = 0; i < npts; ++i)
    {
      r[i] = rmin + (rmax - rmin) * (i % 97) / 96.0;
      th[i] = M_PI * (i % 1001) / 1000.0;
      ph[i] = 2.0 * M_PI * (i % 997) / 997.0 - 1.0;
    }

  potsynth_params_init(&p);
  p.method = method;
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &status);
  if (!w)
    {
      printf("alloc failed: %s\n", potsynth_strerror(status));
      return 1;
    }

  ew = potsynth_eval_alloc(w, 0);
  potsynth_eval(w, ew, npts, r, th, ph, Bs);
  potsynth_eval_free(ew);

  {
    job jobs[NTHREADS];
#ifdef POTSYNTH_TEST_PTHREADS
    pthread_t tid[NTHREADS];
#endif

    for (t = 0; t < NTHREADS; ++t)
      {
        jobs[t].w = w;
        jobs[t].npts = npts;
        jobs[t].chunk = 1000 + 337 * (size_t) t;
        jobs[t].r = r;
        jobs[t].th = th;
        jobs[t].ph = ph;
        jobs[t].B = Bt + (size_t) t * 3 * npts;
      }

#ifdef POTSYNTH_TEST_PTHREADS
    for (t = 0; t < NTHREADS; ++t)
      pthread_create(&tid[t], NULL, worker, &jobs[t]);
    for (t = 0; t < NTHREADS; ++t)
      pthread_join(tid[t], NULL);
#else
    for (t = 0; t < NTHREADS; ++t)
      worker(&jobs[t]);
#endif
  }

  for (t = 0; t < NTHREADS; ++t)
    if (memcmp(Bs, Bt + (size_t) t * 3 * npts, 3 * npts * sizeof(double)) != 0)
      fail = 1;

  printf("  %s: %d threads sharing one workspace: %s\n",
         (method == POTSYNTH_METHOD_FAST) ? "fast  " : "direct", NTHREADS,
         fail ? "FAIL (results differ)" : "ok");

  potsynth_free(w);
  free(gh); free(r); free(th); free(ph); free(Bs); free(Bt);
  return fail;
}

int
main(void)
{
  int nfail = 0;

#ifndef POTSYNTH_TEST_PTHREADS
  printf("no pthreads: the concurrency test runs serially\n");
#endif
  nfail += run(POTSYNTH_METHOD_FAST);
  nfail += run(POTSYNTH_METHOD_DIRECT);

  printf("%s\n", nfail ? "FAILED" : "all passed");
  return nfail != 0;
}
