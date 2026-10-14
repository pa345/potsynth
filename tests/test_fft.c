/*
 * test_fft.c
 */

/*
test_fft()
  Test the pocketfft conventions relied on by the set-up (fast.c):

forward:  F_k = sum_j c_j e^{-2 pi i jk/N}
backward: c_j = sum_k F_k e^{+2 pi i jk/N}
real backward, half-complex input r0, r1, i1, r2, i2, ...:
  x_k = r0 + 2 sum_m (r_m cos - i_m sin)(2 pi mk/N) + r_{N/2} (-1)^k
*/

static void
test_fft(void)
{
  const size_t ns[] = { 12, 30, 64, 375, 750 };
  size_t in;

  for (in = 0; in < sizeof(ns) / sizeof(ns[0]); ++in)
    {
      const size_t N = ns[in];
      double * c = malloc(2 * N * sizeof(double));
      double * c0 = malloc(2 * N * sizeof(double));
      double * r = malloc(N * sizeof(double));
      double * r0 = malloc(N * sizeof(double));
      cfft_plan cp = make_cfft_plan(N);
      rfft_plan rp = make_rfft_plan(N);
      double err_fwd = 0.0, err_inv = 0.0, err_real = 0.0;
      size_t j, k, m;

      for (j = 0; j < 2 * N; ++j)
        c[j] = c0[j] = 2.0 * test_urand() - 1.0;

      cfft_forward(cp, c, 1.0);
      for (k = 0; k < N; ++k)
        {
          long double re = 0.0L, im = 0.0L;

          for (j = 0; j < N; ++j)
            {
              const long double a = -2.0L * M_PI * (long double) ((j * k) % N) / N;

              re += c0[2 * j] * cosl(a) - c0[2 * j + 1] * sinl(a);
              im += c0[2 * j] * sinl(a) + c0[2 * j + 1] * cosl(a);
            }

          err_fwd = fmax(err_fwd, (double) fmaxl(fabsl(c[2 * k] - re), fabsl(c[2 * k + 1] - im)) / N);
        }

      cfft_backward(cp, c, 1.0 / N);
      for (j = 0; j < 2 * N; ++j)
        err_inv = fmax(err_inv, fabs(c[j] - c0[j]));

      for (j = 0; j < N; ++j)
        r[j] = r0[j] = 2.0 * test_urand() - 1.0;

      rfft_backward(rp, r, 1.0);
      for (k = 0; k < N; ++k)
        {
          long double sum = r0[0];

          for (m = 1; 2 * m < N; ++m)
            {
              const long double a = 2.0L * M_PI * (long double) ((m * k) % N) / N;
              sum += 2.0L * (r0[2 * m - 1] * cosl(a) - r0[2 * m] * sinl(a));
            }

          if (N % 2 == 0)
            sum += r0[N - 1] * ((k & 1) ? -1.0L : 1.0L);

          err_real = fmax(err_real, (double) fabsl(r[k] - sum) / N);
        }

      ps_test(err_fwd > 1.0e-14, "fft: forward N=%zu (error %.2e)", N, err_fwd);
      ps_test(err_inv > 1.0e-14, "fft: backward N=%zu (error %.2e)", N, err_inv);
      ps_test(err_real > 1.0e-14, "fft: real backward N=%zu (error %.2e)", N, err_real);

      destroy_cfft_plan(cp);
      destroy_rfft_plan(rp);
      free(c);
      free(c0);
      free(r);
      free(r0);
    }
}
