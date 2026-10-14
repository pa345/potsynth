/*
 * test_radial.c
 */

/* evaluate radial basis function i at u */
static double
test_radial_eval(const size_t i, const double u, const potsynth_radial_workspace * w)
{
  const double t = (w->ud > 0.0) ? (u - w->uc) / w->ud : 0.0;
  double T0 = 1.0, T1 = t, sum = w->basis[i * w->ncheb];
  size_t k;

  if (w->ncheb > 1)
    sum += w->basis[i * w->ncheb + 1] * T1;

  for (k = 2; k < w->ncheb; ++k)
    {
      const double T2 = 2.0 * t * T1 - T0;

      sum += w->basis[i * w->ncheb + k] * T2;
      T0 = T1;
      T1 = T2;
    }

  return sum;
}

/*
test_radial_error()
  Weighted approximation error of u^(l+rpow) over a fine grid, relative to
the largest weighted function value
*/

static double
test_radial_error(const double u0, const double u1, const size_t lmin, const size_t lmax,
                  const size_t nchan, const int * rpow, const double * wn,
                  const potsynth_radial_workspace * w)
{
  const size_t L1 = lmax + 1;
  const size_t ng = 500;
  double emax = 0.0, fmaxv = 0.0;
  size_t c, l, g, i;

  for (c = 0; c < nchan; ++c)
    {
      for (l = lmin; l <= lmax; ++l)
        {
          for (g = 0; g <= ng; ++g)
            {
              const double u = u0 + (u1 - u0) * g / (double) ng;
              const double f = pow(u, (double) l + rpow[c]);
              double approx = 0.0;

              for (i = 0; i < w->rho; ++i)
                approx += w->beta[(c * w->rho + i) * L1 + l] * test_radial_eval(i, u, w);

              emax = fmax(emax, wn[c * L1 + l] * fabs(approx - f));
              fmaxv = fmax(fmaxv, wn[c * L1 + l] * fabs(f));
            }
        }
    }

  return emax / fmaxv;
}

static void
test_radial(void)
{
  const double a = 6371.2;
  const double rmin[] = { a + 250.0, a, a, a + 450.0 };
  const double rmax[] = { a + 800.0, 2.0 * a, a + 1000.0, a + 450.0 };
  const size_t lmin[] = { 1, 1, 0, 1 };
  const size_t lmax[] = { 185, 185, 13, 60 };
  const double tols[] = { 1.0e-6, 1.0e-10, 1.0e-14 };
  const int rpow[2] = { 2, 1 };
  size_t ic, it, l;

  for (ic = 0; ic < sizeof(lmax) / sizeof(lmax[0]); ++ic)
    {
      const size_t L = lmax[ic];
      const size_t L1 = L + 1;
      const double u0 = a / rmax[ic];
      const double u1 = a / rmin[ic];
      double * wn = malloc(2 * L1 * sizeof(double));
      size_t lastrho = 0;

      for (l = 0; l <= L; ++l)
        {
          wn[l] = (l + 1.0) * pow(0.8, (double) l); /* channel 0 */
          wn[L1 + l] = pow(0.8, (double) l);        /* channel 1 */
        }

      for (it = 0; it < sizeof(tols) / sizeof(tols[0]); ++it)
        {
          potsynth_radial_workspace * w = potsynth_radial_alloc(u0, u1, lmin[ic], L, 2, rpow, wn, tols[it]);
          double err;

          ps_test(w == NULL, "radial: case %zu tol=%g allocation", ic, tols[it]);
          if (w == NULL)
            continue;

          err = test_radial_error(u0, u1, lmin[ic], L, 2, rpow, wn, w);

          ps_test(err > 20.0 * tols[it] + 1.0e-15,
                  "radial: case %zu tol=%g rho=%zu ncheb=%zu (error %.2e)",
                  ic, tols[it], w->rho, w->ncheb, err);
          ps_test(w->rho < lastrho, "radial: case %zu rank increases as tol decreases", ic);

          lastrho = w->rho;
          potsynth_radial_free(w);
        }

      free(wn);
    }
}
