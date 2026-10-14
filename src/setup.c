/*
 * setup.c
 *
 * Model set-up: parameters, coefficients, method selection
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"

/*
potsynth_default_parameters()
  Return default parameters
*/

potsynth_parameters
potsynth_default_parameters(void)
{
  potsynth_parameters params;

  params.eps = 1.0e-10;
  params.method = POTSYNTH_METHOD_AUTO;
  params.sigma = 0.0;
  params.w = 0;
  params.rank_tol = 0.0;
  params.nthreads = 0;

  return params;
}

/*
potsynth_nlm()
  Number of SH coefficients for degrees lmin..lmax and orders up to mmax,
i.e. the length of a coefficient vector indexed by potsynth_idx()

Inputs: lmin - minimum SH degree
        lmax - maximum SH degree
        mmax - maximum SH order
*/

size_t
potsynth_nlm(const size_t lmin, const size_t lmax, const size_t mmax)
{
  size_t l, sum = 0;

  for (l = lmin; l <= lmax; ++l)
    {
      const size_t M = (l < mmax) ? l : mmax;
      sum += 2 * M + 1;
    }

  return sum;
}

/*
potsynth_alloc()
  Allocate a model workspace with user-defined channels

Inputs: lmin   - minimum SH degree (>= 0)
        lmax   - maximum SH degree (<= POTSYNTH_LMAX)
        mmax   - maximum SH order (<= lmax)
        coef   - 4-pi fully normalized coefficients, indexed by potsynth_idx(l,m,lmin,mmax),
                 length potsynth_nlm(lmin,lmax,mmax); copied
        a      - reference radius (same unit as r)
        rmin   - minimum radius of evaluation points (fast method; ignored by direct method)
        rmax   - maximum radius of evaluation points (fast method; ignored by direct method)
        nchan  - number of channels
        chan   - channel definitions, length nchan; rfac arrays are copied
        params - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error

Notes:
1) Choose [rmin,rmax] as tight as the data allow: a narrower shell needs fewer
radial functions, so evaluation is faster
*/

potsynth_workspace *
potsynth_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * coef,
               const double a, const double rmin, const double rmax,
               const size_t nchan, const potsynth_channel * chan, const potsynth_parameters * params)
{
  const potsynth_parameters p = (params != NULL) ? *params : potsynth_default_parameters();
  const size_t L1 = lmax + 1;
  const size_t nlm = L1 * (lmax + 2) / 2;
  potsynth_workspace * w;
  size_t c, l, m;
  int status;

  if (lmin > lmax)
    {
      POTSYNTH_ERROR_NULL ("lmin must be <= lmax", POTSYNTH_EINVAL);
    }
  else if (mmax > lmax)
    {
      POTSYNTH_ERROR_NULL ("mmax must be <= lmax", POTSYNTH_EINVAL);
    }
  else if (lmax > POTSYNTH_LMAX)
    {
      POTSYNTH_ERROR_NULL ("lmax exceeds POTSYNTH_LMAX", POTSYNTH_EUNSUP);
    }
  else if (coef == NULL || chan == NULL || nchan == 0)
    {
      POTSYNTH_ERROR_NULL ("coef and channels must be given", POTSYNTH_EINVAL);
    }
  else if (!(a > 0.0) || !isfinite(a))
    {
      POTSYNTH_ERROR_NULL ("reference radius must be positive", POTSYNTH_EINVAL);
    }
  else if (!(p.eps > 0.0) || p.sigma < 0.0 || p.w < 0 || p.rank_tol < 0.0 || p.nthreads < 0)
    {
      POTSYNTH_ERROR_NULL ("invalid parameters", POTSYNTH_EINVAL);
    }
  else if (p.method != POTSYNTH_METHOD_AUTO && p.method != POTSYNTH_METHOD_FAST &&
           p.method != POTSYNTH_METHOD_DIRECT)
    {
      POTSYNTH_ERROR_NULL ("invalid method", POTSYNTH_EINVAL);
    }

  for (c = 0; c < nchan; ++c)
    {
      if (chan[c].theta != POTSYNTH_THETA_P && chan[c].theta != POTSYNTH_THETA_DP &&
          chan[c].theta != POTSYNTH_THETA_PSIN)
        {
          POTSYNTH_ERROR_NULL ("invalid theta operator", POTSYNTH_EINVAL);
        }
      else if (chan[c].dphi != 0 && chan[c].dphi != 1)
        {
          POTSYNTH_ERROR_NULL ("dphi must be 0 or 1", POTSYNTH_EINVAL);
        }
      else if ((long) lmin + chan[c].rpow < 0)
        {
          POTSYNTH_ERROR_NULL ("lmin + rpow must be >= 0", POTSYNTH_EINVAL);
        }
    }

  w = calloc(1, sizeof(potsynth_workspace));
  if (w == NULL)
    {
      POTSYNTH_ERROR_NULL ("unable to allocate workspace", POTSYNTH_ENOMEM);
    }

  w->lmin = lmin;
  w->lmax = lmax;
  w->mmax = mmax;
  w->a = a;
  w->rmin = rmin;
  w->rmax = rmax;
  w->nchan = nchan;
  w->params = p;
  w->nthreads = potsynth_max_threads(p.nthreads);

  if (p.method == POTSYNTH_METHOD_AUTO)
    w->method = (lmax >= POTSYNTH_AUTO_LMIN) ? POTSYNTH_METHOD_FAST : POTSYNTH_METHOD_DIRECT;
  else
    w->method = p.method;

  if (w->method == POTSYNTH_METHOD_FAST && !(rmin > 0.0 && rmin <= rmax && isfinite(rmax)))
    {
      potsynth_free(w);
      POTSYNTH_ERROR_NULL ("fast method needs 0 < rmin <= rmax", POTSYNTH_EINVAL);
    }

  w->rpow = malloc(nchan * sizeof(int));
  w->theta_op = malloc(nchan * sizeof(int));
  w->dphi = malloc(nchan * sizeof(int));
  w->rfac = potsynth_aligned_calloc(nchan * L1 * sizeof(double));
  w->C = potsynth_aligned_calloc(nlm * sizeof(double));
  w->S = potsynth_aligned_calloc(nlm * sizeof(double));
  if (!w->rpow || !w->theta_op || !w->dphi || !w->rfac || !w->C || !w->S)
    {
      potsynth_free(w);
      POTSYNTH_ERROR_NULL ("unable to allocate model arrays", POTSYNTH_ENOMEM);
    }

  for (c = 0; c < nchan; ++c)
    {
      w->rpow[c] = chan[c].rpow;
      w->theta_op[c] = chan[c].theta;
      w->dphi[c] = chan[c].dphi;

      for (l = 0; l <= lmax; ++l)
        w->rfac[c * L1 + l] = (chan[c].rfac != NULL) ? chan[c].rfac[l] : 1.0;
    }

  /* public layout (green2 convention) -> internal M-major */
  for (l = lmin; l <= lmax; ++l)
    {
      const size_t M = (l < mmax) ? l : mmax;

      for (m = 0; m <= M; ++m)
        {
          const size_t idx = potsynth_alf_idx(l, m, lmax);
          const long ll = (long) l, lm = (long) m, llmin = (long) lmin, lmmax = (long) mmax;

          w->C[idx] = coef[potsynth_idx(ll, lm, llmin, lmmax)];
          if (m > 0)
            w->S[idx] = coef[potsynth_idx(ll, -lm, llmin, lmmax)];
        }
    }

  w->alf_workspace_p = potsynth_alf_alloc(lmax);
  if (w->alf_workspace_p == NULL)
    {
      potsynth_free(w);
      return NULL; /* reported by potsynth_alf_alloc() */
    }

  if (w->method == POTSYNTH_METHOD_FAST)
    status = potsynth_fast_init(w);
  else
    status = potsynth_direct_init(w);

  if (status)
    {
      potsynth_free(w);
      return NULL; /* reported above */
    }

  return w;
}

/*
potsynth_free()
  Free model workspace
*/

void
potsynth_free(potsynth_workspace * w)
{
  if (w->rpow)
    free(w->rpow);

  if (w->theta_op)
    free(w->theta_op);

  if (w->dphi)
    free(w->dphi);

  if (w->rfac)
    potsynth_aligned_free(w->rfac);

  if (w->C)
    potsynth_aligned_free(w->C);

  if (w->S)
    potsynth_aligned_free(w->S);

  if (w->alf_workspace_p)
    potsynth_alf_free(w->alf_workspace_p);

  if (w->grid)
    potsynth_aligned_free(w->grid);

  if (w->radial_workspace_p)
    potsynth_radial_free(w->radial_workspace_p);

  potsynth_direct_free(w);

  free(w);
}

/*
potsynth_get_info()
  Return parameters chosen during set-up

Inputs: info - (output) parameters
        w    - workspace

Notes:
1) The fast method fields are 0 for the direct method
*/

int
potsynth_get_info(potsynth_info * info, const potsynth_workspace * w)
{
  memset(info, 0, sizeof(potsynth_info));

  info->lmin = w->lmin;
  info->lmax = w->lmax;
  info->mmax = w->mmax;
  info->nchan = w->nchan;
  info->method = w->method;

  if (w->method == POTSYNTH_METHOD_FAST)
    {
      info->w = w->w;
      info->sigma = w->sigma;
      info->rho = w->radial_workspace_p->rho;
      info->ncheb = w->radial_workspace_p->ncheb;
      info->nf1 = w->nf1;
      info->nf2 = w->nf2;
      info->nrows = w->nrows;
      info->grid_bytes = w->nrows * w->ncols * w->nchp * sizeof(double);
    }

  return POTSYNTH_SUCCESS;
}
