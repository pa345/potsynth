/*
 * test_core.c -- API contract and edge cases; fast method checked against the
 * direct method
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_mag.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int nfail = 0;

#define CHECK(cond, ...)                                                        \
  do {                                                                          \
    if (!(cond))                                                                \
      {                                                                         \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);                             \
        printf(__VA_ARGS__);                                                    \
        printf("\n");                                                           \
        ++nfail;                                                                \
      }                                                                         \
  } while (0)

static unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;

static double
urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (double) (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* random coefficients with a decaying spectrum, green2 layout */
static double *
random_coef(size_t lmin, size_t lmax, size_t mmax)
{
  const size_t nc = potsynth_ncoef(lmin, lmax, mmax);
  double *c = malloc(nc * sizeof(double));
  size_t n, m;

  for (n = lmin; n <= lmax; ++n)
    {
      const size_t M = (n < mmax) ? n : mmax;
      const double sd = 1000.0 * pow(0.7, (double) n);

      for (m = 0; m <= M; ++m)
        {
          c[POTSYNTH_IDX((long) n, (long) m, (long) lmin, (long) mmax)] = sd * (2.0 * urand() - 1.0);
          if (m > 0)
            c[POTSYNTH_IDX((long) n, -(long) m, (long) lmin, (long) mmax)] = sd * (2.0 * urand() - 1.0);
        }
    }

  return c;
}

/* random points in [rmin, rmax] plus poles and seams */
static size_t
make_points(size_t n, double rmin, double rmax, double *r, double *th, double *ph)
{
  const double specials[][2] = { { 0.0, 0.0 }, { M_PI, 0.0 }, { 0.0, 2.0 }, { M_PI, -1.0 },
                                 { 1e-12, 3.0 }, { M_PI - 1e-12, 5.0 }, { M_PI / 2, 0.0 },
                                 { 1.0, 2.0 * M_PI - 1e-14 }, { 2.0, -7.0 }, { 0.5, 1e6 } };
  const size_t ns = sizeof(specials) / sizeof(specials[0]);
  size_t i;

  for (i = 0; i < n; ++i)
    {
      r[i] = rmin + (rmax - rmin) * urand();
      if (i < ns)
        {
          th[i] = specials[i][0];
          ph[i] = specials[i][1];
          r[i] = (i & 1) ? rmax : rmin;
        }
      else
        {
          th[i] = acos(2.0 * urand() - 1.0);
          ph[i] = 2.0 * M_PI * urand();
        }
    }

  return n;
}

static double
max_rel_diff(size_t n, const double *a, const double *b)
{
  double e = 0.0, s = 0.0;
  size_t i;

  for (i = 0; i < n; ++i)
    {
      e = fmax(e, fabs(a[i] - b[i]));
      s = fmax(s, fabs(b[i]));
    }

  return (s > 0.0) ? e / s : e;
}

/* evaluate the same model with the fast and direct methods and compare */
static void
fast_vs_direct(const char *label, size_t lmin, size_t lmax, size_t mmax, double rmin,
               double rmax, size_t nchan, const potsynth_channel *chan, double eps, double tol)
{
  const double a = 6371.2;
  const size_t npts = 3000;
  double *coef = random_coef(lmin, lmax, mmax);
  double *r = malloc(npts * sizeof(double)), *th = malloc(npts * sizeof(double));
  double *ph = malloc(npts * sizeof(double));
  double *Bf = malloc(nchan * npts * sizeof(double)), *Bd = malloc(nchan * npts * sizeof(double));
  potsynth_params p;
  potsynth_workspace *wf, *wd;
  potsynth_eval_workspace *ef, *ed;
  int sf, sd;
  double err;

  make_points(npts, rmin, rmax, r, th, ph);

  potsynth_params_init(&p);
  p.eps = eps;
  p.method = POTSYNTH_METHOD_FAST;
  wf = chan ? potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, &p, &sf)
            : potsynth_mag_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, &p, &sf);
  p.method = POTSYNTH_METHOD_DIRECT;
  wd = chan ? potsynth_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, nchan, chan, &p, &sd)
            : potsynth_mag_alloc(lmin, lmax, mmax, coef, a, rmin, rmax, &p, &sd);

  CHECK(wf && wd, "%s: alloc failed (%s, %s)", label, potsynth_strerror(sf), potsynth_strerror(sd));
  if (wf && wd)
    {
      ef = potsynth_eval_alloc(wf, 0);
      ed = potsynth_eval_alloc(wd, 0);
      sf = potsynth_eval(wf, ef, npts, r, th, ph, Bf);
      sd = potsynth_eval(wd, ed, npts, r, th, ph, Bd);
      CHECK(sf == 0 && sd == 0, "%s: eval failed", label);
      err = max_rel_diff(nchan * npts, Bf, Bd);
      CHECK(err <= tol, "%s: fast vs direct %.2e > %.0e", label, err, tol);
      printf("  %-36s fast vs direct %.2e\n", label, err);
      potsynth_eval_free(ef);
      potsynth_eval_free(ed);
    }

  potsynth_free(wf);
  potsynth_free(wd);
  free(coef); free(r); free(th); free(ph); free(Bf); free(Bd);
}

static void
test_models(void)
{
  const double a = 6371.2;
  double rfac_dg[121], rfac_v[121];
  potsynth_channel grav[2];
  size_t n;

  printf("fast vs direct:\n");
  fast_vs_direct("mag L=60", 1, 60, 60, a + 250, a + 800, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag lmin=16 L=80", 16, 80, 80, a + 250, a + 800, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag mmax=10 L=50", 1, 50, 10, a + 250, a + 800, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag lmin=5 > mmax=3 L=40", 5, 40, 3, a + 250, a + 800, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag thin shell rmin=rmax", 1, 60, 60, a + 450, a + 450, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag L=1", 1, 1, 1, a, a + 1000, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag L=2", 1, 2, 2, a, a + 1000, 3, NULL, 1e-12, 1e-11);
  fast_vs_direct("mag L=30 eps=1e-6", 1, 30, 30, a, a + 1000, 3, NULL, 1e-6, 2e-6);

  /* gravity-like: potential (P, u^(n+1)) and radial derivative (u^(n+2)), lmin = 0 */
  for (n = 0; n <= 120; ++n)
    {
      rfac_v[n] = 1.0;
      rfac_dg[n] = n + 1.0;
    }
  grav[0].rpow = 1; grav[0].rfac = rfac_v;  grav[0].theta = POTSYNTH_THETA_P; grav[0].dphi = 0;
  grav[1].rpow = 2; grav[1].rfac = rfac_dg; grav[1].theta = POTSYNTH_THETA_P; grav[1].dphi = 0;
  fast_vs_direct("core lmin=0 V + dV/dr L=120", 0, 120, 120, a, a + 500, 2, grav, 1e-12, 1e-11);
}

static void
test_contract(void)
{
  const double a = 6371.2, rmin = a + 250, rmax = a + 800;
  const size_t L = 40, npts = 5000;
  double *gh = random_coef(1, L, L);
  double *r = malloc(npts * sizeof(double)), *th = malloc(npts * sizeof(double));
  double *ph = malloc(npts * sizeof(double));
  double *B1 = malloc(3 * npts * sizeof(double)), *B2 = malloc(3 * npts * sizeof(double));
  potsynth_params p;
  potsynth_workspace *w, *w1;
  potsynth_eval_workspace *ew, *ew1, *ewsmall;
  potsynth_info info;
  double x[1], y[1], z[1], o[3];
  int s;
  size_t i;

  printf("API contract:\n");
  make_points(npts, rmin, rmax, r, th, ph);

  potsynth_params_init(&p);
  p.method = POTSYNTH_METHOD_FAST;
  w = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &s);
  CHECK(w != NULL, "alloc: %s", potsynth_strerror(s));
  ew = potsynth_eval_alloc(w, 0);
  ewsmall = potsynth_eval_alloc(w, 777);

  /* chunking does not change results (bitwise) */
  s = potsynth_eval(w, ew, npts, r, th, ph, B1);
  CHECK(s == 0, "eval");
  s = potsynth_eval(w, ewsmall, npts, r, th, ph, B2);
  CHECK(s == 0 && memcmp(B1, B2, 3 * npts * sizeof(double)) == 0, "chunked results differ");

  /* thread count does not change results (bitwise) */
  p.nthreads = 1;
  w1 = potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &s);
  ew1 = potsynth_eval_alloc(w1, 0);
  s = potsynth_eval(w1, ew1, npts, r, th, ph, B2);
  CHECK(s == 0 && memcmp(B1, B2, 3 * npts * sizeof(double)) == 0, "1-thread results differ");
  potsynth_eval_free(ew1);
  potsynth_free(w1);

  /* longitude periodicity */
  for (i = 0; i < npts; ++i)
    ph[i] += 2.0 * M_PI * 1000.0;
  s = potsynth_eval(w, ew, npts, r, th, ph, B2);
  CHECK(s == 0 && max_rel_diff(3 * npts, B2, B1) < 1e-9, "phi + 2000 pi differs: %.2e",
        max_rel_diff(3 * npts, B2, B1));
  for (i = 0; i < npts; ++i)
    ph[i] -= 2.0 * M_PI * 1000.0;

  /* NaN inputs give NaN outputs for that point only */
  r[5] = NAN;
  th[7] = INFINITY;
  s = potsynth_eval(w, ew, npts, r, th, ph, B2);
  CHECK(s == 0, "NaN input should not be an error");
  for (i = 0; i < 3; ++i)
    {
      CHECK(isnan(B2[i * npts + 5]) && isnan(B2[i * npts + 7]), "NaN not propagated");
      CHECK(B2[i * npts + 6] == B1[i * npts + 6], "neighbour of NaN point changed");
    }
  r[5] = rmin;
  th[7] = 1.0;

  /* domain errors leave the output untouched */
  for (i = 0; i < 3 * npts; ++i)
    B2[i] = -12345.0;
  th[100] = -1e-9;
  CHECK(potsynth_eval(w, ew, npts, r, th, ph, B2) == POTSYNTH_EDOM, "theta < 0 accepted");
  th[100] = M_PI + 1e-12;
  CHECK(potsynth_eval(w, ew, npts, r, th, ph, B2) == POTSYNTH_EDOM, "theta > pi accepted");
  th[100] = 1.0;
  r[100] = rmax * (1.0 + 1e-6);
  CHECK(potsynth_eval(w, ew, npts, r, th, ph, B2) == POTSYNTH_EDOM, "r > rmax accepted");
  r[100] = 0.0;
  CHECK(potsynth_eval(w, ew, npts, r, th, ph, B2) == POTSYNTH_EDOM, "r = 0 accepted");
  r[100] = rmin;
  CHECK(B2[0] == -12345.0 && B2[3 * npts - 1] == -12345.0, "output written on error");

  /* empty input; mismatched eval workspace; NULL arrays */
  CHECK(potsynth_eval(w, ew, 0, NULL, NULL, NULL, NULL) == 0, "npts = 0");
  CHECK(potsynth_eval(w, NULL, 1, r, th, ph, B2) == POTSYNTH_EINVAL, "NULL eval workspace");
  CHECK(potsynth_eval(w, ew, 1, r, NULL, ph, B2) == POTSYNTH_EINVAL, "NULL theta");

  /* AUTO picks the direct method at low degree; direct accepts any r > 0 */
  {
    potsynth_workspace *wa;
    potsynth_eval_workspace *ea;

    potsynth_params_init(&p);
    wa = potsynth_mag_alloc(1, 13, 13, gh, a, 0.0, 0.0, &p, &s);
    CHECK(wa != NULL, "AUTO low degree alloc: %s", potsynth_strerror(s));
    potsynth_get_info(wa, &info);
    CHECK(info.method == POTSYNTH_METHOD_DIRECT, "AUTO did not choose direct at L=13");
    ea = potsynth_eval_alloc(wa, 0);
    x[0] = 10.0 * a;
    y[0] = 0.0;
    z[0] = 0.0;
    CHECK(potsynth_eval(wa, ea, 1, x, y, z, o) == 0 && isfinite(o[0]) && isfinite(o[2]),
          "direct at pole, r = 10a");
    potsynth_eval_free(ea);
    potsynth_free(wa);
  }

  /* invalid arguments */
  potsynth_params_init(&p);
  CHECK(potsynth_mag_alloc(0, L, L, gh, a, rmin, rmax, &p, &s) == NULL && s == POTSYNTH_EINVAL,
        "lmin = 0 accepted by mag");
  CHECK(potsynth_mag_alloc(1, L, L + 1, gh, a, rmin, rmax, &p, &s) == NULL, "mmax > lmax");
  p.method = POTSYNTH_METHOD_FAST;
  CHECK(potsynth_mag_alloc(1, L, L, gh, a, rmax, rmin, &p, &s) == NULL, "rmin > rmax");
  CHECK(potsynth_mag_alloc(1, 3000, 3000, gh, a, rmin, rmax, &p, &s) == NULL &&
        s == POTSYNTH_EUNSUP, "lmax = 3000 should be unsupported");
  p.eps = 0.0;
  CHECK(potsynth_mag_alloc(1, L, L, gh, a, rmin, rmax, &p, &s) == NULL, "eps = 0");

  /* NEC conversion */
  o[0] = 1.0; o[1] = 2.0; o[2] = 3.0;
  potsynth_mag_rtp2nec(1, o);
  CHECK(o[0] == -2.0 && o[1] == 3.0 && o[2] == -1.0, "rtp2nec");

  potsynth_eval_free(ew);
  potsynth_eval_free(ewsmall);
  potsynth_free(w);
  free(gh); free(r); free(th); free(ph); free(B1); free(B2);
}

int
main(void)
{
  test_models();
  test_contract();

  printf("%s\n", nfail ? "FAILED" : "all passed");
  return nfail != 0;
}
