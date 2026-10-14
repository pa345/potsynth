/*
 * test_svd.c
 */

/* random n-by-n orthogonal matrix (column-major) from Householder reflections */
static void
test_random_orthogonal(const size_t n, double * Q)
{
  double * v = malloc(n * sizeof(double));
  size_t i, j, k, r;

  for (i = 0; i < n * n; ++i)
    Q[i] = 0.0;
  for (i = 0; i < n; ++i)
    Q[i * n + i] = 1.0;

  for (r = 0; r < n; ++r)
    {
      double nrm = 0.0;

      for (i = 0; i < n; ++i)
        {
          v[i] = 2.0 * test_urand() - 1.0;
          nrm += v[i] * v[i];
        }

      nrm = sqrt(nrm);
      for (i = 0; i < n; ++i)
        v[i] /= nrm;

      /* Q = (I - 2 v v^T) Q */
      for (j = 0; j < n; ++j)
        {
          double d = 0.0;

          for (k = 0; k < n; ++k)
            d += v[k] * Q[j * n + k];

          for (k = 0; k < n; ++k)
            Q[j * n + k] -= 2.0 * d * v[k];
        }
    }

  free(v);
}

/*
test_svd()
  SVD of X = U S W^T with known graded singular values (1 to 1e-15),
including rank deficient and wide matrices
*/

static void
test_svd(void)
{
  const size_t cases[][2] = { { 40, 10 }, { 300, 25 }, { 7, 7 }, { 5, 9 } };
  size_t ic;

  for (ic = 0; ic < sizeof(cases) / sizeof(cases[0]); ++ic)
    {
      const size_t n = cases[ic][0];
      const size_t p = cases[ic][1];
      const size_t r = (n < p) ? n : p;
      double * U = malloc(n * n * sizeof(double));
      double * W = malloc(p * p * sizeof(double));
      double * X = malloc(n * p * sizeof(double));
      double * X0 = malloc(n * p * sizeof(double));
      double * V = malloc(p * p * sizeof(double));
      double * sv = malloc(p * sizeof(double));
      double * sref = malloc(p * sizeof(double));
      double sverr = 0.0, orth = 0.0, rec = 0.0, xnorm = 0.0;
      size_t i, j, k;
      int status, sorted = 1;

      for (k = 0; k < p; ++k)
        sref[k] = (k < r) ? pow(10.0, -15.0 * k / (double) (r > 1 ? r - 1 : 1)) : 0.0;

      if (ic == 2)
        sref[5] = sref[6] = 0.0; /* rank deficient: must still converge */

      test_random_orthogonal(n, U);
      test_random_orthogonal(p, W);
      for (j = 0; j < p; ++j)
        {
          for (i = 0; i < n; ++i)
            {
              long double sum = 0.0L;

              for (k = 0; k < r; ++k)
                sum += (long double) U[k * n + i] * sref[k] * W[k * p + j];

              X[j * n + i] = X0[j * n + i] = (double) sum;
              xnorm = fmax(xnorm, fabs((double) sum));
            }
        }

      status = potsynth_svd_jacobi(n, p, X, V, sv);
      ps_test_int(status, POTSYNTH_SUCCESS, "svd: %zux%zu converged", n, p);

      for (k = 0; k < p; ++k)
        {
          if (k > 0 && sv[k] > sv[k - 1])
            sorted = 0;

          /* one-sided Jacobi: absolute accuracy ~ eps * s_1 */
          sverr = fmax(sverr, fabs(sv[k] - sref[k]));
        }

      /* V orthogonal */
      for (i = 0; i < p; ++i)
        {
          for (j = 0; j < p; ++j)
            {
              double d = 0.0;

              for (k = 0; k < p; ++k)
                d += V[i * p + k] * V[j * p + k];

              orth = fmax(orth, fabs(d - (i == j ? 1.0 : 0.0)));
            }
        }

      /* X0 = X V^T (X now holds U*S) */
      for (j = 0; j < p; ++j)
        {
          for (i = 0; i < n; ++i)
            {
              double d = 0.0;

              for (k = 0; k < p; ++k)
                d += X[k * n + i] * V[k * p + j];

              rec = fmax(rec, fabs(d - X0[j * n + i]));
            }
        }

      ps_test(!sorted, "svd: %zux%zu sorted", n, p);
      ps_test(sverr > 1.0e-14, "svd: %zux%zu singular values (error %.2e)", n, p, sverr);
      ps_test(orth > 1.0e-13, "svd: %zux%zu orthogonality (error %.2e)", n, p, orth);
      ps_test(rec > 1.0e-14 * xnorm * p, "svd: %zux%zu reconstruction (error %.2e)", n, p, rec);

      free(U);
      free(W);
      free(X);
      free(X0);
      free(V);
      free(sv);
      free(sref);
    }
}
