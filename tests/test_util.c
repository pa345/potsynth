/*
 * test_util.c
 */

/* compare potsynth_next_smooth() with a brute force search */
static void
test_next_smooth(void)
{
  size_t n, mult, k;

  for (mult = 1; mult <= 4; mult *= 2)
    {
      int nfail = 0;

      for (n = 1; n <= 3000; ++n)
        {
          const size_t r = potsynth_next_smooth(n, mult);
          size_t t = r;

          while (t % 2 == 0)
            t /= 2;
          while (t % 3 == 0)
            t /= 3;
          while (t % 5 == 0)
            t /= 5;

          if (r < n || r % mult != 0 || t != 1)
            ++nfail;

          /* minimal: no smaller valid length in [n, r) */
          for (k = n; k < r; ++k)
            {
              size_t q = k;

              if (k % mult)
                continue;

              while (q % 2 == 0)
                q /= 2;
              while (q % 3 == 0)
                q /= 3;
              while (q % 5 == 0)
                q /= 5;

              if (q == 1)
                ++nfail;
            }
        }

      ps_test(nfail, "util: next_smooth mult=%zu (%d failures)", mult, nfail);
    }
}

/*
test_pow_scaled()
  Compare potsynth_pow_scaled() with long double logarithms (the value
itself may be far outside the double range); the relative error is bounded
by about k*eps, as for a running product
*/

static void
test_pow_scaled(void)
{
  const double svals[] = { 0.0, 1.0, -1.0, 0.5, -0.7, 0.999999, 1.0e-3, -1.0e-3, 1.0e-100,
                           1.0e-300, 3.0e-310, 0.389, 1.0 / 3.0 };
  const size_t ns = sizeof(svals) / sizeof(svals[0]);
  double maxerr = 0.0;
  size_t i, k;

  for (i = 0; i < ns; ++i)
    {
      for (k = 0; k <= 3000; k += (k < 20) ? 1 : 37)
        {
          const double s = svals[i];
          double mant;
          int expo;

          potsynth_pow_scaled(s, k, &mant, &expo);

          if (s == 0.0)
            {
              ps_test_rel(mant, (k == 0) ? 1.0 : 0.0, 0.0, "util: pow_scaled(0,%zu)", k);
              continue;
            }

          {
            const long double ref = (long double) k * log2l(fabsl((long double) s));
            const long double got = log2l(fabsl((long double) mant)) + expo;
            const int negative = (s < 0.0 && (k & 1));
            const double err = (double) (fabsl(got - ref) * M_LN2) / ((k + 1.0) * DBL_EPSILON);

            ps_test((mant < 0) != negative, "util: pow_scaled sign s=%g k=%zu", s, k);
            ps_test(!((k == 0 && mant == 1.0 && expo == 0) || (fabs(mant) >= 0.5 && fabs(mant) < 1.0)),
                    "util: pow_scaled mantissa s=%g k=%zu", s, k);

            if (err > maxerr)
              maxerr = err;
          }
        }
    }

  ps_test(maxerr > 1.0, "util: pow_scaled error %.2f (k+1) eps", maxerr);
}

/* compare potsynth_sincos_m() with a long double reference using the exact product m*phi */
static void
test_sincos_m(void)
{
  const double phis[] = { 0.0, 1.0e-10, 0.1, 1.0, 2.0, M_PI, 4.0, 2.0 * M_PI - 1.0e-12, 6.2 };
  const size_t nphi = sizeof(phis) / sizeof(phis[0]);
  double maxerr = 0.0;
  size_t i, m;

  for (i = 0; i < nphi; ++i)
    {
      for (m = 0; m <= 2800; m += (m < 50) ? 1 : 13)
        {
          const long double a = (long double) m * (long double) phis[i];
          const long double da = fmal((long double) m, (long double) phis[i], -a);
          const long double cref = cosl(a) - da * sinl(a);
          const long double sref = sinl(a) + da * cosl(a);
          double c, s;

          potsynth_sincos_m(m, phis[i], &c, &s);

          maxerr = fmax(maxerr, (double) fabsl((long double) c - cref));
          maxerr = fmax(maxerr, (double) fabsl((long double) s - sref));
        }
    }

  ps_test(maxerr > 4.0 * DBL_EPSILON, "util: sincos_m error %.2e", maxerr);
}

static void
test_util(void)
{
  test_next_smooth();
  test_pow_scaled();
  test_sincos_m();
}
