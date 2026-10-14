/*
 * test_quad.c
 */

/* Gauss-Legendre rule must integrate x^d exactly for d <= 2n-1 */
static void
test_gauss_legendre(void)
{
  const size_t ns[] = { 1, 2, 3, 10, 57, 100, 228 };
  size_t i, j, d;

  for (i = 0; i < sizeof(ns) / sizeof(ns[0]); ++i)
    {
      const size_t n = ns[i];
      double * x = malloc(n * sizeof(double));
      double * w = malloc(n * sizeof(double));
      double maxerr = 0.0;
      int nbad = 0;

      potsynth_gauss_legendre(n, x, w);

      /* d <= 60 to stay well conditioned */
      for (d = 0; d <= 2 * n - 1 && d <= 60; ++d)
        {
          const double ref = (d & 1) ? 0.0 : 2.0 / (d + 1.0);
          long double sum = 0.0L;

          for (j = 0; j < n; ++j)
            sum += (long double) w[j] * powl(x[j], (long double) d);

          maxerr = fmax(maxerr, fabs((double) sum - ref));
        }

      for (j = 0; j < n; ++j)
        {
          if (fabs(x[j] + x[n - 1 - j]) > 1.0e-15 || (j > 0 && x[j] <= x[j - 1]))
            ++nbad;
        }

      ps_test(maxerr > 1.0e-14, "quad: Gauss-Legendre n=%zu moments (max error %.2e)", n, maxerr);
      ps_test(nbad, "quad: Gauss-Legendre n=%zu node symmetry/order", n);

      free(x);
      free(w);
    }
}

/*
window_ft_ref()
  phi_hat(k) by tanh-sinh (double exponential) quadrature in z, in long
double: converges exponentially despite the endpoint singularity of phi, and
is independent of the library's substitution + Gauss-Legendre rule
*/

static long double
window_ft_ref(const double k, const int w, const double beta, const double h)
{
  const long double alpha = 0.5L * w * h;
  const long double step = 1.0L / 64.0L;
  long double sum = 0.0L;
  int i;

  for (i = -64 * 6; i <= 64 * 6; ++i)
    {
      const long double tau = i * step;
      const long double sh = 0.5L * M_PI * sinhl(tau);
      const long double z = tanhl(sh);
      const long double ch2 = coshl(sh) * coshl(sh);
      const long double wgt = 0.5L * M_PI * coshl(tau) / ch2;
      const long double omz2 = 1.0L / ch2; /* 1 - z^2 */

      sum += wgt * expl(beta * (sqrtl(omz2) - 1.0L)) * cosl(k * alpha * z);
    }

  return alpha * sum * step;
}

static void
test_window_ft(void)
{
  const int ws[] = { 4, 7, 11, 16 };
  size_t i, k;

  for (i = 0; i < sizeof(ws) / sizeof(ws[0]); ++i)
    {
      const int w = ws[i];
      const double sigma = 2.0;
      const double beta = M_PI * (1.0 - 1.0 / (2.0 * sigma)) * w;
      const size_t L = 185;
      const size_t nf = potsynth_next_smooth((size_t) (sigma * (2 * L + 1)), 2);
      const double h = 2.0 * M_PI / nf;
      double * ft = malloc((L + 1) * sizeof(double));
      double maxerr = 0.0;
      int nbad = 0;

      potsynth_window_ft(L + 1, w, beta, h, ft);

      for (k = 0; k <= L; k += 5)
        {
          const long double ref = window_ft_ref((double) k, w, beta, h);
          maxerr = fmax(maxerr, (double) fabsl((ft[k] - ref) / ref));
        }

      /* phi_hat must be positive and decreasing over the band */
      for (k = 1; k <= L; ++k)
        {
          if (!(ft[k] > 0.0 && ft[k] < ft[k - 1]))
            ++nbad;
        }

      ps_test(maxerr > 1.0e-13, "quad: window_ft w=%d (max relative error %.2e)", w, maxerr);
      ps_test(nbad, "quad: window_ft w=%d positive and decreasing", w);

      free(ft);
    }
}

static void
test_quad(void)
{
  test_gauss_legendre();
  test_window_ft();

  ps_test_int(potsynth_window_choose(1.0e-6, 2.0), 7, "quad: window width eps=1e-6");
  ps_test_int(potsynth_window_choose(1.0e-10, 2.0), 11, "quad: window width eps=1e-10");
  ps_test_int(potsynth_window_choose(1.0e-30, 2.0), POTSYNTH_WMAX, "quad: window width eps=1e-30");
  ps_test_int(potsynth_window_choose(1.0, 2.0), 4, "quad: window width eps=1");
}
