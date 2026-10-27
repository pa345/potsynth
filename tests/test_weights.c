/*
 * test_weights.c
 *
 * Mean squares of the channel operators (the radial weights of the fast
 * method) against numerical integration over the sphere of the channels
 * computed by the direct method
 */

/*
test_op_msq()
  For each operator (theta operator, phi derivative) and each (l, m), integrate
the square of the channel of one 4-pi normalized harmonic over the sphere
(Gauss-Legendre in cos(theta), uniform in phi: exact for these trigonometric
polynomials) and compare with potsynth_op_msq(). Also check that the two
families have m-independent sums, l(l+1) and (l-1)l(l+1)(l+2)
*/

static void
test_op_msq(void)
{
  const size_t L = 16;
  const size_t nt = 2 * L + 6, np = 2 * L + 6;
  const size_t n = nt * np;
  const int ops[10] = { POTSYNTH_THETA_P, POTSYNTH_THETA_P, POTSYNTH_THETA_DP, POTSYNTH_THETA_DP,
                        POTSYNTH_THETA_PSIN, POTSYNTH_THETA_PSIN, POTSYNTH_THETA_S2A,
                        POTSYNTH_THETA_S2A, POTSYNTH_THETA_S2B, POTSYNTH_THETA_S2B };
  const size_t nlm = potsynth_nlm(0, L, L);
  potsynth_channel chan[10];
  potsynth_parameters params = potsynth_default_parameters();
  double * x = malloc(nt * sizeof(double));
  double * wq = malloc(nt * sizeof(double));
  double * r = malloc(n * sizeof(double));
  double * theta = malloc(n * sizeof(double));
  double * phi = malloc(n * sizeof(double));
  double * out = malloc(10 * n * sizeof(double));
  double * coef = calloc(nlm, sizeof(double));
  double maxerr = 0.0, maxfam1 = 0.0, maxfam2 = 0.0;
  size_t lmin = 0; /* a variable: potsynth_idx() with a constant 0 trips -Wtype-limits */
  size_t c, i, j, l, m;

  params.method = POTSYNTH_METHOD_DIRECT;
  potsynth_gauss_legendre(nt, x, wq);

  for (c = 0; c < 10; ++c)
    {
      chan[c].rpow = 1;
      chan[c].rfac = NULL;
      chan[c].theta = ops[c];
      chan[c].dphi = (int) (c & 1);
    }

  for (i = 0; i < nt; ++i)
    {
      for (j = 0; j < np; ++j)
        {
          r[i * np + j] = 1.0;
          theta[i * np + j] = acos(x[i]);
          phi[i * np + j] = 2.0 * M_PI * j / np;
        }
    }

  for (l = 0; l <= L; ++l)
    {
      for (m = 0; m <= l; ++m)
        {
          potsynth_workspace * w;
          potsynth_eval_workspace * ew;
          double msq[10];

          coef[potsynth_idx(l, m, lmin, L)] = 1.0;
          w = potsynth_alloc(0, L, L, coef, 1.0, 0.0, 0.0, 10, chan, &params);
          ew = potsynth_eval_alloc(1, w);
          potsynth_eval(n, r, theta, phi, out, ew);
          coef[potsynth_idx(l, m, lmin, L)] = 0.0;

          for (c = 0; c < 10; ++c)
            {
              const double ref = potsynth_op_msq(ops[c], (int) (c & 1), l, m);
              long double sum = 0.0L;

              for (i = 0; i < nt; ++i)
                {
                  for (j = 0; j < np; ++j)
                    {
                      const double v = out[c * n + i * np + j];
                      sum += (long double) wq[i] * v * v;
                    }
                }

              msq[c] = (double) (sum / (2.0L * np));
              maxerr = fmax(maxerr, fabs(msq[c] - ref) / fmax(1.0, ref));
            }

          {
            const double ll = (double) l * (l + 1.0);
            const double lam2 = (l < 2) ? 0.0 : ((double) l - 1.0) * ll * (l + 2.0);

            maxfam1 = fmax(maxfam1, fabs(msq[2] + msq[5] - ll) / fmax(1.0, ll));
            maxfam2 = fmax(maxfam2, fabs(msq[6] + 4.0 * msq[9] - lam2) / fmax(1.0, lam2));
          }

          potsynth_eval_free(ew);
          potsynth_free(w);
        }
    }

  ps_test(maxerr > 1.0e-12, "weights: operator mean squares, L=%zu (max rel error %.2e)", L, maxerr);
  ps_test(maxfam1 > 1.0e-12, "weights: dP^2 + (Psin d/dphi)^2 = l(l+1) (max rel error %.2e)", maxfam1);
  ps_test(maxfam2 > 1.0e-12, "weights: S2A^2 + 4 (S2B d/dphi)^2 = (l-1)l(l+1)(l+2) (max rel error %.2e)",
          maxfam2);

  free(x);
  free(wq);
  free(r);
  free(theta);
  free(phi);
  free(out);
  free(coef);
}

static void
test_weights(void)
{
  test_op_msq();
}
