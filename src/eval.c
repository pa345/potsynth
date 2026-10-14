/*
 * eval.c -- evaluation at scattered points (Step 5 of the method)
 *
 * Fast method, per point:
 *   - window weights in theta and phi (2w evaluations);
 *   - w x w weighted sum over the grid of all channels at once; thanks to the
 *     periodic padding columns each window row is one contiguous block of
 *     w * nchp values, so the inner loop is a plain vectorisable multiply-add;
 *   - radial values b_i(a/r) from their Chebyshev series, and the radial sum.
 * Points are processed in chunks, counting-sorted by grid block so that
 * consecutive points read neighbouring memory, and distributed over threads.
 *
 * Robustness:
 *   - all inputs are validated before anything is written to the output;
 *   - non-finite inputs (NaN/Inf) give NaN outputs for that point;
 *   - grid indices are clamped, so no input can cause an out-of-bounds read;
 *   - the poles theta = 0, pi need no special treatment: the grid is a smooth
 *     periodic function on the doubled sphere and windows straddling a pole read
 *     the stored margin rows.
 *
 * Evaluation performs no memory allocation.
 *
 * Copyright (c) 2026, Patrick Alken and Jonathan Burchill. BSD 3-Clause License.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "potsynth_internal.h"

#define PS_SORT_BLOCK   16          /* grid cells per sort bin, each direction */
#define PS_PAR_BLOCK    256         /* points per parallel work unit */
#define PS_CHUNK_DEF    (1u << 18)  /* default points per chunk */

static size_t
round8(size_t n)
{
  return (n + 7) / 8 * 8;
}

/* fast-method per-thread scratch: k1[w], k2[w], acc[nchp], tmp[nchp], T[ncheb], b[rho] */
static size_t
fast_scratch_size(const potsynth_workspace *w)
{
  return 2 * round8((size_t) w->w) + 2 * round8(w->nchp) + round8(w->rad.ncheb) +
         round8(w->rad.rho);
}

potsynth_eval_workspace *
potsynth_eval_alloc(const potsynth_workspace *w, size_t chunk)
{
  potsynth_eval_workspace *ew;
  size_t tsize;

  if (!w)
    return NULL;

  ew = calloc(1, sizeof(potsynth_eval_workspace));
  if (!ew)
    return NULL;

  ew->w = w;
  ew->nthreads = w->nthreads;
  ew->chunk = (chunk > 0) ? chunk : PS_CHUNK_DEF;
  if (ew->chunk > UINT32_MAX)
    ew->chunk = UINT32_MAX;

  if (w->method == POTSYNTH_METHOD_FAST)
    {
      const size_t nbr = w->nrows / PS_SORT_BLOCK + 1;
      const size_t nbc = w->ncols / PS_SORT_BLOCK + 1;

      tsize = fast_scratch_size(w);
      ew->nbins = nbr * nbc + 1;              /* last bin: non-finite points */
      ew->key = malloc(ew->chunk * sizeof(uint32_t));
      ew->order = malloc(ew->chunk * sizeof(uint32_t));
      ew->count = malloc((ew->nbins + 1) * sizeof(size_t));
      if (!ew->key || !ew->order || !ew->count)
        {
          potsynth_eval_free(ew);
          return NULL;
        }
    }
  else
    {
      tsize = ps_direct_scratch_size(w);
    }

  ew->tstride = round8(tsize);
  ew->tscratch = ps_malloc((size_t) ew->nthreads * ew->tstride * sizeof(double));
  if (!ew->tscratch)
    {
      potsynth_eval_free(ew);
      return NULL;
    }

  return ew;
}

void
potsynth_eval_free(potsynth_eval_workspace *ew)
{
  if (!ew)
    return;

  free(ew->key);
  free(ew->order);
  free(ew->count);
  ps_free(ew->tscratch);
  free(ew);
}

static int
point_finite(double r, double theta, double phi)
{
  return isfinite(r) && isfinite(theta) && isfinite(phi);
}

/* phi reduced to [0, 2 pi) */
static double
reduce_phi(double phi)
{
  double p = fmod(phi, 2.0 * M_PI);

  if (p < 0.0)
    p += 2.0 * M_PI;
  if (p >= 2.0 * M_PI)
    p = 0.0;

  return p;
}

static long
clampl(long v, long lo, long hi)
{
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

/* first window row and column of a point (always inside the stored grid) */
static void
window_origin(const potsynth_workspace *w, double theta, double phi, long *l1, long *l2)
{
  *l1 = clampl((long) ceil(theta * w->inv_h1 - w->half), w->row_lo, w->row_hi - w->w + 1);
  *l2 = clampl((long) ceil(phi * w->inv_h2 - w->half), w->col_lo, w->col_hi - w->w + 1);
}

/* validate all points before any output is written */
static int
validate(const potsynth_workspace *w, size_t npts, const double *r, const double *theta,
         const double *phi)
{
  const double tmax = 1.0 + 1e-10;
  size_t i;

  for (i = 0; i < npts; ++i)
    {
      if (!point_finite(r[i], theta[i], phi[i]))
        continue;                         /* NaN output, not an error */

      if (theta[i] < 0.0 || theta[i] > M_PI || !(r[i] > 0.0))
        return POTSYNTH_EDOM;

      if (w->method == POTSYNTH_METHOD_FAST)
        {
          const double u = w->a / r[i];

          if (w->rad.ud > 0.0)
            {
              if (fabs((u - w->rad.uc) / w->rad.ud) > tmax)
                return POTSYNTH_EDOM;
            }
          else if (fabs(u - w->rad.uc) > 1e-12 * w->rad.uc)
            {
              return POTSYNTH_EDOM;
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

/*
 * acc[q] = sum_{a,b < w} k1[a] k2[b] g[a*rstride + b*nchp + q],  q < nchp
 *
 * Channels are processed in blocks of CB with NACC independent partial sums
 * (interleaved over b), so that enough independent multiply-add chains are in
 * flight to hide the FMA latency; the fixed sizes let the compiler keep all
 * accumulators in vector registers. Each window row is a contiguous run of
 * w*nchp values.
 */
#if defined(__GNUC__) || defined(__clang__)

/* 4 doubles; aligned(8) permits unaligned access (the grid is in fact 32-byte
 * aligned here, since nchp is a multiple of 4 and the grid 64-byte aligned) */
typedef double ps_v4 __attribute__((vector_size(32), aligned(8)));

static inline void
window_sum(const double *restrict g0, size_t rstride, size_t nchp, size_t ww,
           const double *restrict k1, const double *restrict k2, double *restrict acc)
{
  const ps_v4 zero = { 0.0, 0.0, 0.0, 0.0 };
  size_t q = 0, a, b;

  /* blocks of 8 channels: 2 vectors x 4 partial sums (over b mod 4) = 8
   * independent multiply-add chains */
  for (; q + 8 <= nchp; q += 8)
    {
      ps_v4 s0a = zero, s0b = zero, s1a = zero, s1b = zero;
      ps_v4 s2a = zero, s2b = zero, s3a = zero, s3b = zero;

      for (a = 0; a < ww; ++a)
        {
          const double wa = k1[a];
          const double *g = g0 + a * rstride + q;

          for (b = 0; b + 4 <= ww; b += 4)
            {
              const ps_v4 *p0 = (const ps_v4 *) (g + b * nchp);
              const ps_v4 *p1 = (const ps_v4 *) (g + (b + 1) * nchp);
              const ps_v4 *p2 = (const ps_v4 *) (g + (b + 2) * nchp);
              const ps_v4 *p3 = (const ps_v4 *) (g + (b + 3) * nchp);
              const double w0 = wa * k2[b], w1 = wa * k2[b + 1];
              const double w2 = wa * k2[b + 2], w3 = wa * k2[b + 3];

              s0a += w0 * p0[0];
              s0b += w0 * p0[1];
              s1a += w1 * p1[0];
              s1b += w1 * p1[1];
              s2a += w2 * p2[0];
              s2b += w2 * p2[1];
              s3a += w3 * p3[0];
              s3b += w3 * p3[1];
            }

          for (; b < ww; ++b)
            {
              const ps_v4 *p0 = (const ps_v4 *) (g + b * nchp);
              const double w0 = wa * k2[b];

              s0a += w0 * p0[0];
              s0b += w0 * p0[1];
            }
        }

      *(ps_v4 *) (acc + q) = (s0a + s1a) + (s2a + s3a);
      *(ps_v4 *) (acc + q + 4) = (s0b + s1b) + (s2b + s3b);
    }

  /* remaining block of 4 channels: 4 partial sums */
  for (; q + 4 <= nchp; q += 4)
    {
      ps_v4 s0 = zero, s1 = zero, s2 = zero, s3 = zero;

      for (a = 0; a < ww; ++a)
        {
          const double wa = k1[a];
          const double *g = g0 + a * rstride + q;

          for (b = 0; b + 4 <= ww; b += 4)
            {
              s0 += (wa * k2[b]) * *(const ps_v4 *) (g + b * nchp);
              s1 += (wa * k2[b + 1]) * *(const ps_v4 *) (g + (b + 1) * nchp);
              s2 += (wa * k2[b + 2]) * *(const ps_v4 *) (g + (b + 2) * nchp);
              s3 += (wa * k2[b + 3]) * *(const ps_v4 *) (g + (b + 3) * nchp);
            }

          for (; b < ww; ++b)
            s0 += (wa * k2[b]) * *(const ps_v4 *) (g + b * nchp);
        }

      *(ps_v4 *) (acc + q) = (s0 + s1) + (s2 + s3);
    }
}

#else

static void
window_sum(const double *restrict g0, size_t rstride, size_t nchp, size_t ww,
           const double *restrict k1, const double *restrict k2, double *restrict acc)
{
  size_t q, a, b;

  for (q = 0; q < nchp; ++q)
    acc[q] = 0.0;

  for (a = 0; a < ww; ++a)
    for (b = 0; b < ww; ++b)
      {
        const double wab = k1[a] * k2[b];
        const double *g = g0 + a * rstride + b * nchp;

        for (q = 0; q < nchp; ++q)
          acc[q] += wab * g[q];
      }
}

#endif

static void
fast_point(const potsynth_workspace *w, double *scratch, double r, double theta, double phi,
           double *out, size_t ldo)
{
  const int ww = w->w;
  const size_t nchp = w->nchp, ncols = w->ncols, rho = w->rad.rho, ncheb = w->rad.ncheb;
  double *k1 = scratch;
  double *k2 = k1 + round8((size_t) ww);
  double *acc = k2 + round8((size_t) ww);
  double *T = acc + 2 * round8(nchp);
  double *b = T + round8(ncheb);
  const double ph = reduce_phi(phi);
  const double zt = theta * w->inv_h1, zp = ph * w->inv_h2;
  const double ihalf = 1.0 / w->half;
  double t;
  long l1, l2;
  size_t a, i, k, c;

  if (!point_finite(r, theta, phi))
    {
      for (c = 0; c < w->nchan; ++c)
        out[c * ldo] = NAN;
      return;
    }

  window_origin(w, theta, ph, &l1, &l2);

  for (a = 0; a < (size_t) ww; ++a)
    {
      k1[a] = ps_window((zt - (double) (l1 + (long) a)) * ihalf, w->beta);
      k2[a] = ps_window((zp - (double) (l2 + (long) a)) * ihalf, w->beta);
    }

  window_sum(w->grid + ((size_t) (l1 - w->row_lo) * ncols + (size_t) (l2 - w->col_lo)) * nchp,
             ncols * nchp, nchp, (size_t) ww, k1, k2, acc);

  /* radial values */
  if (w->rad.ud > 0.0)
    {
      t = (w->a / r - w->rad.uc) / w->rad.ud;
      t = (t > 1.0) ? 1.0 : ((t < -1.0) ? -1.0 : t);
    }
  else
    {
      t = 0.0;
    }

  T[0] = 1.0;
  if (ncheb > 1)
    T[1] = t;
  for (k = 2; k < ncheb; ++k)
    T[k] = 2.0 * t * T[k - 1] - T[k - 2];

  for (i = 0; i < rho; ++i)
    {
      const double *bas = w->rad.basis + i * ncheb;
      double sum = 0.0;

      for (k = 0; k < ncheb; ++k)
        sum += bas[k] * T[k];
      b[i] = sum;
    }

  for (c = 0; c < w->nchan; ++c)
    {
      double sum = 0.0;

      for (i = 0; i < rho; ++i)
        sum += b[i] * acc[c * rho + i];
      out[c * ldo] = sum;
    }
}

static int
eval_fast(const potsynth_workspace *w, potsynth_eval_workspace *ew, size_t npts,
          const double *r, const double *theta, const double *phi, double *out)
{
  const size_t nbc = w->ncols / PS_SORT_BLOCK + 1;
  size_t s0;

  for (s0 = 0; s0 < npts; s0 += ew->chunk)
    {
      const size_t n = (npts - s0 < ew->chunk) ? npts - s0 : ew->chunk;
      const double *rc = r + s0, *tc = theta + s0, *pc = phi + s0;
      size_t i, k;
      long blk, nblk;

      /* counting sort by grid block */
      memset(ew->count, 0, (ew->nbins + 1) * sizeof(size_t));
      for (i = 0; i < n; ++i)
        {
          size_t key;

          if (point_finite(rc[i], tc[i], pc[i]))
            {
              long l1, l2;

              window_origin(w, tc[i], reduce_phi(pc[i]), &l1, &l2);
              key = (size_t) (l1 - w->row_lo) / PS_SORT_BLOCK * nbc +
                    (size_t) (l2 - w->col_lo) / PS_SORT_BLOCK;
            }
          else
            {
              key = ew->nbins - 1;
            }

          ew->key[i] = (uint32_t) key;
          ++ew->count[key + 1];
        }

      for (k = 0; k < ew->nbins; ++k)
        ew->count[k + 1] += ew->count[k];

      for (i = 0; i < n; ++i)
        ew->order[ew->count[ew->key[i]]++] = (uint32_t) i;

      nblk = (long) ((n + PS_PAR_BLOCK - 1) / PS_PAR_BLOCK);

#pragma omp parallel for schedule(dynamic, 4) num_threads(ew->nthreads)
      for (blk = 0; blk < nblk; ++blk)
        {
#ifdef _OPENMP
          double *scratch = ew->tscratch + (size_t) omp_get_thread_num() * ew->tstride;
#else
          double *scratch = ew->tscratch;
#endif
          const size_t j0 = (size_t) blk * PS_PAR_BLOCK;
          const size_t j1 = (j0 + PS_PAR_BLOCK < n) ? j0 + PS_PAR_BLOCK : n;
          size_t jj;

          for (jj = j0; jj < j1; ++jj)
            {
              const size_t idx = ew->order[jj];
              fast_point(w, scratch, rc[idx], tc[idx], pc[idx], out + s0 + idx, npts);
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

static int
eval_direct(const potsynth_workspace *w, potsynth_eval_workspace *ew, size_t npts,
            const double *r, const double *theta, const double *phi, double *out)
{
  const long nblk = (long) ((npts + PS_NB - 1) / PS_NB);
  long blk;

#pragma omp parallel for schedule(dynamic, 16) num_threads(ew->nthreads)
  for (blk = 0; blk < nblk; ++blk)
    {
#ifdef _OPENMP
      double *scratch = ew->tscratch + (size_t) omp_get_thread_num() * ew->tstride;
#else
      double *scratch = ew->tscratch;
#endif
      const size_t j0 = (size_t) blk * PS_NB;
      const size_t nb = (j0 + PS_NB < npts) ? PS_NB : npts - j0;
      double rb[PS_NB], tb[PS_NB], pb[PS_NB];
      int bad[PS_NB];
      size_t j, c;

      /* non-finite points are replaced by a harmless one and flagged */
      for (j = 0; j < nb; ++j)
        {
          bad[j] = !point_finite(r[j0 + j], theta[j0 + j], phi[j0 + j]);
          rb[j] = bad[j] ? w->a : r[j0 + j];
          tb[j] = bad[j] ? 0.5 * M_PI : theta[j0 + j];
          pb[j] = bad[j] ? 0.0 : phi[j0 + j];
        }

      ps_direct_block(w, scratch, nb, rb, tb, pb, out + j0, npts);

      for (j = 0; j < nb; ++j)
        if (bad[j])
          for (c = 0; c < w->nchan; ++c)
            out[c * npts + j0 + j] = NAN;
    }

  return POTSYNTH_SUCCESS;
}

int
potsynth_eval(const potsynth_workspace *w, potsynth_eval_workspace *ew, size_t npts,
              const double *r, const double *theta, const double *phi, double *out)
{
  int status;

  if (!w || !ew || ew->w != w)
    return POTSYNTH_EINVAL;
  if (npts == 0)
    return POTSYNTH_SUCCESS;
  if (!r || !theta || !phi || !out)
    return POTSYNTH_EINVAL;

  status = validate(w, npts, r, theta, phi);
  if (status)
    return status;

  if (w->method == POTSYNTH_METHOD_FAST)
    return eval_fast(w, ew, npts, r, theta, phi, out);

  return eval_direct(w, ew, npts, r, theta, phi, out);
}
