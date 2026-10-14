/*
 * test_mag_ref.c -- magnetic front-end against extended-precision reference values
 * (tests/gen_reference.py), for the direct method and the fast method at several
 * accuracies. The point sets include both poles at several longitudes, points
 * 1e-300 .. 1e-6 from the poles, r = rmin and rmax, and longitudes outside [0, 2pi).
 *
 * Usage: test_mag_ref FILE...
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "potsynth_mag.h"

typedef struct
{
  int64_t lmax, npts;
  double a, rmin, rmax;
  double *gh, *r, *theta, *phi, *B;
} refdata;

static int
read_ref(const char *path, refdata *d)
{
  FILE *fp = fopen(path, "rb");
  size_t ncoef, ok = 1;
  int64_t hdr[2];
  double par[3];

  if (!fp)
    return 1;

  ok = ok && fread(hdr, sizeof(int64_t), 2, fp) == 2;
  ok = ok && fread(par, sizeof(double), 3, fp) == 3;
  if (!ok)
    {
      fclose(fp);
      return 1;
    }

  d->lmax = hdr[0];
  d->npts = hdr[1];
  d->a = par[0];
  d->rmin = par[1];
  d->rmax = par[2];
  ncoef = potsynth_ncoef(1, (size_t) d->lmax, (size_t) d->lmax);

  d->gh = malloc(ncoef * sizeof(double));
  d->r = malloc((size_t) d->npts * sizeof(double));
  d->theta = malloc((size_t) d->npts * sizeof(double));
  d->phi = malloc((size_t) d->npts * sizeof(double));
  d->B = malloc(3 * (size_t) d->npts * sizeof(double));

  ok = ok && fread(d->gh, sizeof(double), ncoef, fp) == ncoef;
  ok = ok && fread(d->r, sizeof(double), (size_t) d->npts, fp) == (size_t) d->npts;
  ok = ok && fread(d->theta, sizeof(double), (size_t) d->npts, fp) == (size_t) d->npts;
  ok = ok && fread(d->phi, sizeof(double), (size_t) d->npts, fp) == (size_t) d->npts;
  ok = ok && fread(d->B, sizeof(double), 3 * (size_t) d->npts, fp) == 3 * (size_t) d->npts;
  fclose(fp);

  return !ok;
}

static void
free_ref(refdata *d)
{
  free(d->gh);
  free(d->r);
  free(d->theta);
  free(d->phi);
  free(d->B);
}

/* max |B - Bref| / max |Bref| over all components; also report non-finite values */
static double
max_rel_err(const refdata *d, const double *B, size_t *nbad)
{
  double emax = 0.0, bmax = 0.0;
  size_t i;

  *nbad = 0;
  for (i = 0; i < 3 * (size_t) d->npts; ++i)
    {
      if (!isfinite(B[i]))
        ++*nbad;
      emax = fmax(emax, fabs(B[i] - d->B[i]));
      bmax = fmax(bmax, fabs(d->B[i]));
    }

  return emax / bmax;
}

static int
run(const refdata *d, potsynth_method method, double eps, double tol, const char *label)
{
  potsynth_params p;
  potsynth_workspace *w;
  potsynth_eval_workspace *ew;
  potsynth_info info;
  double *B = malloc(3 * (size_t) d->npts * sizeof(double));
  double err;
  size_t nbad;
  int status, fail;

  potsynth_params_init(&p);
  p.method = method;
  p.eps = eps;

  w = potsynth_mag_alloc(1, (size_t) d->lmax, (size_t) d->lmax, d->gh, d->a, d->rmin,
                         d->rmax, &p, &status);
  if (!w)
    {
      printf("  %-8s eps=%.0e  alloc failed: %s  FAIL\n", label, eps, potsynth_strerror(status));
      free(B);
      return 1;
    }

  ew = potsynth_eval_alloc(w, 0);
  status = potsynth_eval(w, ew, (size_t) d->npts, d->r, d->theta, d->phi, B);
  potsynth_get_info(w, &info);

  err = max_rel_err(d, B, &nbad);
  fail = (status != POTSYNTH_SUCCESS) || nbad || !(err <= tol);

  if (method == POTSYNTH_METHOD_FAST)
    printf("  %-8s eps=%.0e  w=%2d rho=%2zu ncheb=%2zu  err %.2e (tol %.0e)  %s\n", label, eps,
           info.w, info.rho, info.ncheb, err, tol, fail ? "FAIL" : "ok");
  else
    printf("  %-8s                              err %.2e (tol %.0e)  %s\n", label, err, tol,
           fail ? "FAIL" : "ok");

  if (status)
    printf("    eval status: %s\n", potsynth_strerror(status));
  if (nbad)
    printf("    %zu non-finite outputs\n", nbad);

  potsynth_eval_free(ew);
  potsynth_free(w);
  free(B);
  return fail;
}

int
main(int argc, char **argv)
{
  int i, nfail = 0;

  if (argc < 2)
    {
      fprintf(stderr, "usage: %s FILE...\n", argv[0]);
      return 2;
    }

  for (i = 1; i < argc; ++i)
    {
      refdata d;

      if (read_ref(argv[i], &d))
        {
          printf("cannot read %s\n", argv[i]);
          ++nfail;
          continue;
        }

      printf("%s: L=%d, %d points, r in [%.1f, %.1f]\n", argv[i], (int) d.lmax, (int) d.npts,
             d.rmin, d.rmax);

      nfail += run(&d, POTSYNTH_METHOD_DIRECT, 1e-10, 1e-13, "direct");
      nfail += run(&d, POTSYNTH_METHOD_FAST, 1e-6, 1e-6, "fast");
      nfail += run(&d, POTSYNTH_METHOD_FAST, 1e-10, 1e-10, "fast");
      nfail += run(&d, POTSYNTH_METHOD_FAST, 1e-14, 1e-13, "fast");

      free_ref(&d);
    }

  printf("%s\n", nfail ? "FAILED" : "all passed");
  return nfail != 0;
}
