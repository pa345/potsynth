/*
 * test_alf.c
 */

/*
test_alf_ref()
  Long double 4-pi P(l,m), dP/dtheta and P/sin by the standard unscaled
recursions (adequate for lmax <= 185: no underflow in long double)

Inputs: lmax  - maximum degree
        theta - colatitude
        P     - (output) P(l,m), M-major
        D     - (output) dP(l,m)/dtheta, M-major
        PS    - (output) P(l,m)/sin(theta), M-major
*/

static void
test_alf_ref(const size_t lmax, const long double theta, long double * P, long double * D,
             long double * PS)
{
  const long double x = cosl(theta);
  const long double s = sinl(theta);
  long double pmm = 1.0L, dmm = 0.0L, qmm = 0.0L; /* P(m,m), dP(m,m), P(m,m)/s */
  size_t m, l;

  for (m = 0; m <= lmax; ++m)
    {
      long double p1, p2 = 0.0L, d1, d2 = 0.0L, q1, q2 = 0.0L;

      if (m == 1)
        {
          pmm = sqrtl(3.0L) * s;
          dmm = sqrtl(3.0L) * x;
          qmm = sqrtl(3.0L);
        }
      else if (m >= 2)
        {
          const long double f = sqrtl(1.0L + 0.5L / m);

          dmm = f * (s * dmm + x * pmm);
          pmm *= f * s;
          qmm *= f * s;
        }

      p1 = pmm;
      d1 = dmm;
      q1 = qmm;

      for (l = m; l <= lmax; ++l)
        {
          const size_t idx = potsynth_alf_idx(l, m, lmax);

          if (l > m)
            {
              long double p, d, q;

              if (l == m + 1)
                {
                  const long double c = sqrtl(2.0L * m + 3.0L);

                  p = c * x * p1;
                  d = c * (x * d1 - s * p1);
                  q = c * x * q1;
                }
              else
                {
                  const long double a = sqrtl((2.0L * l + 1) * (2.0L * l - 1) /
                                              ((long double) (l + m) * (l - m)));
                  const long double b = -sqrtl((long double) (l + m - 1) * (l - m - 1) * (2.0L * l + 1) /
                                               ((long double) (l + m) * (l - m) * (2.0L * l - 3)));

                  p = a * x * p1 + b * p2;
                  d = a * (x * d1 - s * p1) + b * d2;
                  q = a * x * q1 + b * q2;
                }

              p2 = p1;
              d2 = d1;
              q2 = q1;
              p1 = p;
              d1 = d;
              q1 = q;
            }

          P[idx] = p1;
          D[idx] = d1;
          PS[idx] = q1;
        }
    }
}

/* unscale a column value: v * s^p * 2^scale */
static double
test_alf_unscale(const double v, const double s, const size_t p, const int scale)
{
  double mant;
  int expo;

  potsynth_pow_scaled(s, p, &mant, &expo);

  return ldexp(v * mant, expo + scale);
}

/*
test_alf_columns()
  Compare the scaled columns with the long double reference at lmax = 185,
at and near the poles; errors are relative to the largest value of each
quantity (near the poles the recursion error grows like eps * L^2)
*/

static void
test_alf_columns(void)
{
  const size_t L = 185;
  const size_t nlm = (L + 1) * (L + 2) / 2;
  const double thetas[] = { 0.0, 1.0e-300, 1.0e-8, 0.05, 0.3, 1.2, M_PI / 2.0, 2.5,
                            M_PI - 1.0e-6, M_PI };
  long double * P = malloc(nlm * sizeof(long double));
  long double * D = malloc(nlm * sizeof(long double));
  long double * PS = malloc(nlm * sizeof(long double));
  double * Q = malloc((L + 1) * POTSYNTH_NB * sizeof(double));
  double * E = malloc((L + 1) * POTSYNTH_NB * sizeof(double));
  potsynth_alf_workspace * w = potsynth_alf_alloc(L);
  size_t it, j, m, l;

  for (it = 0; it < sizeof(thetas) / sizeof(thetas[0]); ++it)
    {
      const double theta = thetas[it];
      double x[POTSYNTH_NB], s[POTSYNTH_NB];
      double scP = 1.0, scD = 1.0, scS = 1.0;
      double errP = 0.0, errD = 0.0, errS = 0.0;
      int nbad = 0;

      test_alf_ref(L, (long double) theta, P, D, PS);

      for (j = 0; j < POTSYNTH_NB; ++j)
        {
          x[j] = cos(theta);
          s[j] = sin(theta);
        }

      for (m = 0; m <= L; ++m)
        {
          for (l = m; l <= L; ++l)
            {
              const size_t idx = potsynth_alf_idx(l, m, L);

              scP = fmax(scP, (double) fabsl(P[idx]));
              scD = fmax(scD, (double) fabsl(D[idx]));
              if (m > 0)
                scS = fmax(scS, (double) fabsl(PS[idx]));
            }
        }

      for (m = 0; m <= L; ++m)
        {
          potsynth_alf_column(m, x, s, Q, E, w);

          for (l = m; l <= L; ++l)
            {
              const size_t idx = potsynth_alf_idx(l, m, L);
              const double q = Q[(l - m) * POTSYNTH_NB];
              const double e = E[(l - m) * POTSYNTH_NB];
              double p, d;

              if (m == 0)
                {
                  p = q;
                  d = e;
                }
              else
                {
                  const double ps = test_alf_unscale(q, s[0], m - 1, w->scale[m]);

                  p = test_alf_unscale(q, s[0], m, w->scale[m]);
                  d = test_alf_unscale(e, s[0], m - 1, w->scale[m]);
                  errS = fmax(errS, (double) fabsl(ps - PS[idx]) / scS);

                  if (!isfinite(ps))
                    ++nbad;
                }

              if (!isfinite(p) || !isfinite(d))
                ++nbad;

              errP = fmax(errP, (double) fabsl(p - P[idx]) / scP);
              errD = fmax(errD, (double) fabsl(d - D[idx]) / scD);
            }
        }

      ps_test(nbad, "alf: columns L=%zu theta=%g finite", L, theta);
      ps_test(errP > 2.0e-12, "alf: P L=%zu theta=%g (error %.2e)", L, theta, errP);
      ps_test(errD > 2.0e-12, "alf: dP/dtheta L=%zu theta=%g (error %.2e)", L, theta, errD);
      ps_test(errS > 2.0e-12, "alf: P/sin L=%zu theta=%g (error %.2e)", L, theta, errS);
    }

  /* analytic values at the north pole: P(l,0) = sqrt(2l+1), P(l,1)/sin = sqrt((2l+1) l (l+1) / 2) */
  {
    double x[POTSYNTH_NB], s[POTSYNTH_NB];

    for (j = 0; j < POTSYNTH_NB; ++j)
      {
        x[j] = 1.0;
        s[j] = 0.0;
      }

    potsynth_alf_column(0, x, s, Q, E, w);
    for (l = 0; l <= L; ++l)
      ps_test_rel(Q[l * POTSYNTH_NB], sqrt(2.0 * l + 1.0), 5.0e-13, "alf: pole P(%zu,0)", l);

    potsynth_alf_column(1, x, s, Q, E, w);
    for (l = 1; l <= L; ++l)
      ps_test_rel(ldexp(Q[(l - 1) * POTSYNTH_NB], w->scale[1]),
                  sqrt((2.0 * l + 1.0) * l * (l + 1.0) / 2.0), 5.0e-13, "alf: pole P(%zu,1)/sin", l);
  }

  potsynth_alf_free(w);
  free(P);
  free(D);
  free(PS);
  free(Q);
  free(E);
}

/* column scaling: none below L ~ 1370, active at 2190, supported up to POTSYNTH_LMAX */
static void
test_alf_scale(void)
{
  potsynth_alf_workspace * w;
  size_t m;
  int nscaled;

  w = potsynth_alf_alloc(1300);
  nscaled = 0;
  for (m = 0; m <= 1300; ++m)
    nscaled += (w->scale[m] > 0);
  ps_test_int(nscaled, 0, "alf: no column scaling at L=1300");
  potsynth_alf_free(w);

  w = potsynth_alf_alloc(2190);
  nscaled = 0;
  for (m = 0; m <= 2190; ++m)
    nscaled += (w->scale[m] > 0);
  ps_test(nscaled == 0, "alf: column scaling at L=2190");
  potsynth_alf_free(w);

  w = potsynth_alf_alloc(POTSYNTH_LMAX);
  ps_test(w == NULL, "alf: allocation at POTSYNTH_LMAX");
  if (w)
    potsynth_alf_free(w);

  w = potsynth_alf_alloc(POTSYNTH_LMAX + 100);
  ps_test(w != NULL, "alf: allocation beyond POTSYNTH_LMAX must fail");
  ps_test_int(test_last_error(), POTSYNTH_EUNSUP, "alf: beyond POTSYNTH_LMAX error code");
}

/* potsynth_alf_accum() against long double sums, random sizes and start indices */
static void
test_alf_accum(void)
{
  double maxerr = 0.0;
  size_t trial;

  for (trial = 0; trial < 200; ++trial)
    {
      const size_t nn = 1 + (size_t) (test_urand() * 60);
      const size_t npair = 1 + (size_t) (test_urand() * 5);
      const size_t ldw = nn + 3;
      double * W = malloc(2 * npair * ldw * sizeof(double));
      double * Q = malloc(nn * POTSYNTH_NB * sizeof(double));
      double * E = malloc(nn * POTSYNTH_NB * sizeof(double));
      double * acc = malloc(4 * npair * POTSYNTH_NB * sizeof(double));
      unsigned char * src = malloc(npair);
      size_t * n0 = malloc(npair * sizeof(size_t));
      size_t i, k, j, l;

      for (i = 0; i < 2 * npair * ldw; ++i)
        W[i] = 2.0 * test_urand() - 1.0;

      for (i = 0; i < nn * POTSYNTH_NB; ++i)
        {
          Q[i] = 2.0 * test_urand() - 1.0;
          E[i] = 2.0 * test_urand() - 1.0;
        }

      for (k = 0; k < npair; ++k)
        {
          src[k] = test_urand() < 0.5;
          n0[k] = (size_t) (test_urand() * (nn + 1)); /* includes n0 = nn (empty) */
        }

      potsynth_alf_accum(nn, npair, W, ldw, src, n0, Q, E, acc);

      for (k = 0; k < npair; ++k)
        {
          for (j = 0; j < POTSYNTH_NB; ++j)
            {
              const double * V = src[k] ? E : Q;
              long double ref[4] = { 0.0L, 0.0L, 0.0L, 0.0L };

              for (l = n0[k]; l < nn; ++l)
                {
                  ref[0 + (l & 1)] += (long double) W[2 * k * ldw + l] * V[l * POTSYNTH_NB + j];
                  ref[2 + (l & 1)] += (long double) W[(2 * k + 1) * ldw + l] * V[l * POTSYNTH_NB + j];
                }

              for (i = 0; i < 4; ++i)
                maxerr = fmax(maxerr, (double) fabsl(acc[(4 * k + i) * POTSYNTH_NB + j] - ref[i]));
            }
        }

      free(W);
      free(Q);
      free(E);
      free(acc);
      free(src);
      free(n0);
    }

  ps_test(maxerr > 1.0e-13, "alf: accumulation kernel (error %.2e)", maxerr);
}

static void
test_alf(void)
{
  test_alf_columns();
  test_alf_scale();
  test_alf_accum();
}
