/*
 * eval.c
 *
 * Evaluation at scattered points (step 5 of the fast method, and the direct
 * method)
 *
 * Fast method, for each point:
 *
 * 1) window weights in theta and phi (2w evaluations)
 *
 * 2) w-by-w weighted sum over the grid of all channels at once; thanks to the
 *    periodic padding columns each window row is one contiguous block of
 *    w*nchp values
 *
 * 3) radial values b_i(a/r) from their Chebyshev series, and the radial sum
 *
 * Points are processed in chunks, counting-sorted by grid block so that
 * consecutive points read neighboring memory, and distributed over threads.
 *
 * All inputs are validated before anything is written to the output;
 * non-finite inputs (NaN/Inf) give NaN outputs; grid indices are clamped, so no
 * input can read out of bounds. The poles need no special treatment: the grid
 * is a smooth periodic function on the doubled sphere, and windows straddling a
 * pole read the stored margin rows. Evaluation performs no memory allocation.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "potsynth_internal.h"

#define POTSYNTH_SORT_BLOCK          16          /* grid cells per sort bin, each direction */
#define POTSYNTH_PAR_BLOCK           256         /* points per parallel work unit */
#define POTSYNTH_CHUNK_DEFAULT       (1u << 18)  /* default points per chunk */

static size_t round8(const size_t n);
static size_t fast_scratch_size(const potsynth_workspace * w);
static int point_finite(const double r, const double theta, const double phi);
static double reduce_phi(const double phi);
static void window_origin(const double theta, const double phi, long * l1, long * l2,
                          const potsynth_workspace * w);
static int validate(const size_t n, const double * r, const double * theta, const double * phi,
                    const potsynth_workspace * w);
static void window_sum(const double * g0, const size_t rstride, const size_t nchp, const size_t ww,
                       const double * k1, const double * k2, double * acc);
static void fast_point(const double r, const double theta, const double phi, const size_t ldo,
                       double * out, double * scratch, const potsynth_workspace * w);
static int eval_fast(const size_t n, const double * r, const double * theta, const double * phi,
                     double * out, potsynth_eval_workspace * ew);
static void eval_outputs(const size_t n, double * out, potsynth_eval_workspace * ew);
static int eval_direct(const size_t n, const double * r, const double * theta, const double * phi,
                       double * out, potsynth_eval_workspace * ew);

/*
potsynth_eval_alloc()
  Allocate an evaluation workspace

Inputs: chunk - points per chunk (0 = default); any number of points may be
                passed to potsynth_eval()
        w     - model workspace

Return: pointer to workspace

Notes:
1) Each thread calling potsynth_eval() concurrently needs its own evaluation
workspace; the model workspace w may be shared
*/

potsynth_eval_workspace *
potsynth_eval_alloc(const size_t chunk, const potsynth_workspace * w)
{
  potsynth_eval_workspace * ew;
  size_t tsize;

  ew = calloc(1, sizeof(potsynth_eval_workspace));
  if (ew == NULL)
    {
      POTSYNTH_ERROR_NULL ("unable to allocate evaluation workspace", POTSYNTH_ENOMEM);
    }

  ew->w = w;
  ew->nthreads = w->nthreads;
  ew->chunk = (chunk > 0) ? chunk : POTSYNTH_CHUNK_DEFAULT;
  if (ew->chunk > UINT32_MAX)
    ew->chunk = UINT32_MAX;

  if (w->method == POTSYNTH_METHOD_FAST)
    {
      const size_t nbr = w->nrows / POTSYNTH_SORT_BLOCK + 1;
      const size_t nbc = w->ncols / POTSYNTH_SORT_BLOCK + 1;

      tsize = fast_scratch_size(w);
      ew->nbins = nbr * nbc + 1; /* last bin: non-finite points */
      ew->key = malloc(ew->chunk * sizeof(uint32_t));
      ew->order = malloc(ew->chunk * sizeof(uint32_t));
      ew->count = malloc((ew->nbins + 1) * sizeof(size_t));
      if (!ew->key || !ew->order || !ew->count)
        {
          potsynth_eval_free(ew);
          POTSYNTH_ERROR_NULL ("unable to allocate sort arrays", POTSYNTH_ENOMEM);
        }
    }
  else
    {
      tsize = potsynth_direct_scratch_size(w);
    }

  ew->tstride = round8(tsize);
  ew->tscratch = potsynth_aligned_alloc((size_t) ew->nthreads * ew->tstride * sizeof(double));
  if (ew->tscratch == NULL)
    {
      potsynth_eval_free(ew);
      POTSYNTH_ERROR_NULL ("unable to allocate thread scratch", POTSYNTH_ENOMEM);
    }

  return ew;
}

void
potsynth_eval_free(potsynth_eval_workspace * ew)
{
  if (ew->key)
    free(ew->key);

  if (ew->order)
    free(ew->order);

  if (ew->count)
    free(ew->count);

  if (ew->tscratch)
    potsynth_aligned_free(ew->tscratch);

  free(ew);
}

/*
potsynth_eval()
  Evaluate all channels of the model at n points

Inputs: n     - number of points
        r     - radius, length n
        theta - colatitude in [0,pi] (radians), length n
        phi   - longitude (radians), any finite value, length n
        out   - (output) output-major, out[k*n + i], length nout*n (nout = number
                of channels, or the outputs defined by a front-end)
        ew    - evaluation workspace

Return: success or error

Notes:
1) All points are validated before anything is written. POTSYNTH_EDOM is
returned if any point has theta outside [0,pi], r <= 0, or (fast method)
r outside [rmin,rmax]

2) Points with a NaN or infinite coordinate are not an error: their outputs
are set to NaN

3) The results are bitwise identical for any number of threads and chunk size
*/

int
potsynth_eval(const size_t n, const double * r, const double * theta, const double * phi,
              double * out, potsynth_eval_workspace * ew)
{
  if (n == 0)
    {
      return POTSYNTH_SUCCESS;
    }
  else if (r == NULL || theta == NULL || phi == NULL || out == NULL)
    {
      POTSYNTH_ERROR ("NULL input or output array", POTSYNTH_EINVAL);
    }
  else
    {
      int status = validate(n, r, theta, phi, ew->w);

      if (status)
        {
          POTSYNTH_ERROR ("theta outside [0,pi], r <= 0, or (fast method) r outside [rmin,rmax]", status);
        }

      if (ew->w->method == POTSYNTH_METHOD_FAST)
        status = eval_fast(n, r, theta, phi, out, ew);
      else
        status = eval_direct(n, r, theta, phi, out, ew);

      if (status == POTSYNTH_SUCCESS && ew->w->omap != NULL)
        eval_outputs(n, out, ew);

      return status;
    }
}

/*
eval_outputs()
  Replace the channel values by the outputs of the front-end map, in place

Inputs: n   - number of points
        out - channel values out[c*n + i] on input, outputs out[k*n + i] on output
        ew  - evaluation workspace

Notes:
1) Each point is independent: its channel values are read before any of its
outputs are written
*/

static void
eval_outputs(const size_t n, double * out, potsynth_eval_workspace * ew)
{
  const potsynth_workspace * w = ew->w;
  const size_t nchan = w->nchan;
  const size_t nout = w->nout;
  long ii;

#pragma omp parallel for private(ii) schedule(static) num_threads(ew->nthreads) if (n > 100000)
  for (ii = 0; ii < (long) n; ++ii)
    {
      const size_t i = (size_t) ii;
      double v[POTSYNTH_OMAP_MAX];
      size_t c, k;

      for (c = 0; c < nchan; ++c)
        v[c] = out[c * n + i];

      for (k = 0; k < nout; ++k)
        {
          const double * mk = w->omap + k * nchan;
          double sum = 0.0;

          for (c = 0; c < nchan; ++c)
            {
              if (mk[c] != 0.0)
                sum += mk[c] * v[c];
            }

          out[k * n + i] = sum;
        }
    }
}

static size_t
round8(const size_t n)
{
  return (n + 7) / 8 * 8;
}

/* fast method per-thread scratch: k1[w], k2[w], acc[nchp], (spare)[nchp], T[ncheb], b[rho] */
static size_t
fast_scratch_size(const potsynth_workspace * w)
{
  return 2 * round8((size_t) w->w) + 2 * round8(w->nchp) +
         round8(w->radial_workspace_p->ncheb) + round8(w->radial_workspace_p->rho);
}

static int
point_finite(const double r, const double theta, const double phi)
{
  return isfinite(r) && isfinite(theta) && isfinite(phi);
}

/* phi reduced to [0, 2 pi) */
static double
reduce_phi(const double phi)
{
  double p = fmod(phi, 2.0 * M_PI);

  if (p < 0.0)
    p += 2.0 * M_PI;

  if (p >= 2.0 * M_PI)
    p = 0.0;

  return p;
}

/*
window_origin()
  First window row and column of a point, clamped to the stored grid

Inputs: theta - colatitude (radians)
        phi   - longitude in [0,2pi) (radians)
        l1    - (output) first row
        l2    - (output) first column
        w     - workspace
*/

static void
window_origin(const double theta, const double phi, long * l1, long * l2, const potsynth_workspace * w)
{
  const long r1 = (long) ceil(theta * w->inv_h1 - w->half);
  const long c1 = (long) ceil(phi * w->inv_h2 - w->half);
  const long rmax = w->row_hi - w->w + 1;
  const long cmax = w->col_hi - w->w + 1;

  *l1 = (r1 < w->row_lo) ? w->row_lo : ((r1 > rmax) ? rmax : r1);
  *l2 = (c1 < w->col_lo) ? w->col_lo : ((c1 > cmax) ? cmax : c1);
}

/* validate all points before any output is written */
static int
validate(const size_t n, const double * r, const double * theta, const double * phi,
         const potsynth_workspace * w)
{
  const double tmax = 1.0 + 1.0e-10;
  size_t i;

  for (i = 0; i < n; ++i)
    {
      if (!point_finite(r[i], theta[i], phi[i]))
        continue; /* NaN output, not an error */

      if (theta[i] < 0.0 || theta[i] > M_PI || !(r[i] > 0.0))
        return POTSYNTH_EDOM;

      if (w->method == POTSYNTH_METHOD_FAST)
        {
          const potsynth_radial_workspace * rad = w->radial_workspace_p;
          const double u = w->a / r[i];

          if (rad->ud > 0.0)
            {
              if (fabs((u - rad->uc) / rad->ud) > tmax)
                return POTSYNTH_EDOM;
            }
          else if (fabs(u - rad->uc) > 1.0e-12 * rad->uc)
            {
              return POTSYNTH_EDOM;
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

#ifdef POTSYNTH_HAVE_VECEXT

/* 4 doubles; aligned(8) permits unaligned access */
typedef double potsynth_v4 __attribute__((vector_size(32), aligned(8)));

/*
window_sum()
  acc[q] = sum_{a,b < w} k1[a] k2[b] g[a*rstride + b*nchp + q], q < nchp

Inputs: g0      - first grid value of the window
        rstride - grid row stride
        nchp    - padded number of channels (multiple of 4)
        ww      - window width
        k1      - window weights in theta, length ww
        k2      - window weights in phi, length ww
        acc     - (output) sums, length nchp

Notes:
1) Channels are processed in blocks of 8 with 4 independent partial sums each
(over b mod 4), so that 8 multiply-add chains hide the FMA latency; all
accumulators are in vector registers
*/

static void
window_sum(const double * restrict g0, const size_t rstride, const size_t nchp, const size_t ww,
           const double * restrict k1, const double * restrict k2, double * restrict acc)
{
  const potsynth_v4 zero = { 0.0, 0.0, 0.0, 0.0 };
  size_t q = 0, a, b;

  /* blocks of 8 channels */
  for (; q + 8 <= nchp; q += 8)
    {
      potsynth_v4 s0a = zero, s0b = zero, s1a = zero, s1b = zero;
      potsynth_v4 s2a = zero, s2b = zero, s3a = zero, s3b = zero;

      for (a = 0; a < ww; ++a)
        {
          const double wa = k1[a];
          const double * g = g0 + a * rstride + q;

          for (b = 0; b + 4 <= ww; b += 4)
            {
              const potsynth_v4 * p0 = (const potsynth_v4 *) (g + b * nchp);
              const potsynth_v4 * p1 = (const potsynth_v4 *) (g + (b + 1) * nchp);
              const potsynth_v4 * p2 = (const potsynth_v4 *) (g + (b + 2) * nchp);
              const potsynth_v4 * p3 = (const potsynth_v4 *) (g + (b + 3) * nchp);
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
              const potsynth_v4 * p0 = (const potsynth_v4 *) (g + b * nchp);
              const double w0 = wa * k2[b];

              s0a += w0 * p0[0];
              s0b += w0 * p0[1];
            }
        }

      *(potsynth_v4 *) (acc + q) = (s0a + s1a) + (s2a + s3a);
      *(potsynth_v4 *) (acc + q + 4) = (s0b + s1b) + (s2b + s3b);
    }

  /* remaining block of 4 channels */
  for (; q + 4 <= nchp; q += 4)
    {
      potsynth_v4 s0 = zero, s1 = zero, s2 = zero, s3 = zero;

      for (a = 0; a < ww; ++a)
        {
          const double wa = k1[a];
          const double * g = g0 + a * rstride + q;

          for (b = 0; b + 4 <= ww; b += 4)
            {
              s0 += (wa * k2[b]) * *(const potsynth_v4 *) (g + b * nchp);
              s1 += (wa * k2[b + 1]) * *(const potsynth_v4 *) (g + (b + 1) * nchp);
              s2 += (wa * k2[b + 2]) * *(const potsynth_v4 *) (g + (b + 2) * nchp);
              s3 += (wa * k2[b + 3]) * *(const potsynth_v4 *) (g + (b + 3) * nchp);
            }

          for (; b < ww; ++b)
            s0 += (wa * k2[b]) * *(const potsynth_v4 *) (g + b * nchp);
        }

      *(potsynth_v4 *) (acc + q) = (s0 + s1) + (s2 + s3);
    }
}

#else

/* portable version of window_sum() */
static void
window_sum(const double * restrict g0, const size_t rstride, const size_t nchp, const size_t ww,
           const double * restrict k1, const double * restrict k2, double * restrict acc)
{
  size_t q, a, b;

  for (q = 0; q < nchp; ++q)
    acc[q] = 0.0;

  for (a = 0; a < ww; ++a)
    {
      for (b = 0; b < ww; ++b)
        {
          const double wab = k1[a] * k2[b];
          const double * g = g0 + a * rstride + b * nchp;

          for (q = 0; q < nchp; ++q)
            acc[q] += wab * g[q];
        }
    }
}

#endif /* POTSYNTH_HAVE_VECEXT */

/*
fast_point()
  Evaluate all channels at one point with the fast method

Inputs: r       - radius
        theta   - colatitude (radians)
        phi     - longitude (radians)
        ldo     - output stride between channels
        out     - (output) out[c*ldo]
        scratch - per-thread scratch space
        w       - workspace
*/

static void
fast_point(const double r, const double theta, const double phi, const size_t ldo,
           double * out, double * scratch, const potsynth_workspace * w)
{
  const potsynth_radial_workspace * rad = w->radial_workspace_p;
  const size_t ww = (size_t) w->w;
  const size_t nchp = w->nchp;
  const size_t rho = rad->rho;
  const size_t ncheb = rad->ncheb;
  const double ihalf = 1.0 / w->half;
  double * k1 = scratch;
  double * k2 = k1 + round8(ww);
  double * acc = k2 + round8(ww);
  double * T = acc + 2 * round8(nchp);
  double * b = T + round8(ncheb);
  double ph, zt, zp, t;
  long l1, l2;
  size_t a, i, k, c;

  if (!point_finite(r, theta, phi))
    {
      for (c = 0; c < w->nchan; ++c)
        out[c * ldo] = NAN;

      return;
    }

  ph = reduce_phi(phi);
  zt = theta * w->inv_h1;
  zp = ph * w->inv_h2;

  window_origin(theta, ph, &l1, &l2, w);

  /* window weights */
  for (a = 0; a < ww; ++a)
    {
      k1[a] = potsynth_window((zt - (double) (l1 + (long) a)) * ihalf, w->beta);
      k2[a] = potsynth_window((zp - (double) (l2 + (long) a)) * ihalf, w->beta);
    }

  /* w-by-w window sum, all channels */
  window_sum(w->grid + ((size_t) (l1 - w->row_lo) * w->ncols + (size_t) (l2 - w->col_lo)) * nchp,
             w->ncols * nchp, nchp, ww, k1, k2, acc);

  /* radial values b_i(t) */
  if (rad->ud > 0.0)
    {
      t = (w->a / r - rad->uc) / rad->ud;
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
      const double * basis = rad->basis + i * ncheb;
      double sum = 0.0;

      for (k = 0; k < ncheb; ++k)
        sum += basis[k] * T[k];

      b[i] = sum;
    }

  /* radial sum */
  for (c = 0; c < w->nchan; ++c)
    {
      double sum = 0.0;

      for (i = 0; i < rho; ++i)
        sum += b[i] * acc[c * rho + i];

      out[c * ldo] = sum;
    }
}

/*
eval_fast()
  Fast method evaluation: chunks of points, counting sort by grid block,
parallel over blocks of sorted points
*/

static int
eval_fast(const size_t n, const double * r, const double * theta, const double * phi,
          double * out, potsynth_eval_workspace * ew)
{
  const potsynth_workspace * w = ew->w;
  const size_t nbc = w->ncols / POTSYNTH_SORT_BLOCK + 1;
  size_t s0;

  for (s0 = 0; s0 < n; s0 += ew->chunk)
    {
      const size_t nc = (n - s0 < ew->chunk) ? n - s0 : ew->chunk;
      const double * rc = r + s0;
      const double * tc = theta + s0;
      const double * pc = phi + s0;
      const long nblk = (long) ((nc + POTSYNTH_PAR_BLOCK - 1) / POTSYNTH_PAR_BLOCK);
      size_t i, k;
      long blk;

      /* counting sort by grid block */
      memset(ew->count, 0, (ew->nbins + 1) * sizeof(size_t));
      for (i = 0; i < nc; ++i)
        {
          size_t key;

          if (point_finite(rc[i], tc[i], pc[i]))
            {
              long l1, l2;

              window_origin(tc[i], reduce_phi(pc[i]), &l1, &l2, w);
              key = (size_t) (l1 - w->row_lo) / POTSYNTH_SORT_BLOCK * nbc +
                    (size_t) (l2 - w->col_lo) / POTSYNTH_SORT_BLOCK;
            }
          else
            {
              key = ew->nbins - 1;
            }

          ew->key[i] = (uint32_t) key;
          ++(ew->count[key + 1]);
        }

      for (k = 0; k < ew->nbins; ++k)
        ew->count[k + 1] += ew->count[k];

      for (i = 0; i < nc; ++i)
        ew->order[ew->count[ew->key[i]]++] = (uint32_t) i;

#pragma omp parallel for private(blk) schedule(dynamic, 4) num_threads(ew->nthreads)
      for (blk = 0; blk < nblk; ++blk)
        {
#ifdef _OPENMP
          double * scratch = ew->tscratch + (size_t) omp_get_thread_num() * ew->tstride;
#else
          double * scratch = ew->tscratch;
#endif
          const size_t j0 = (size_t) blk * POTSYNTH_PAR_BLOCK;
          const size_t j1 = (j0 + POTSYNTH_PAR_BLOCK < nc) ? j0 + POTSYNTH_PAR_BLOCK : nc;
          size_t jj;

          for (jj = j0; jj < j1; ++jj)
            {
              const size_t idx = ew->order[jj];
              fast_point(rc[idx], tc[idx], pc[idx], n, out + s0 + idx, scratch, w);
            }
        }
    }

  return POTSYNTH_SUCCESS;
}

/*
eval_direct()
  Direct method evaluation: blocks of POTSYNTH_NB points in parallel
*/

static int
eval_direct(const size_t n, const double * r, const double * theta, const double * phi,
            double * out, potsynth_eval_workspace * ew)
{
  const potsynth_workspace * w = ew->w;
  const long nblk = (long) ((n + POTSYNTH_NB - 1) / POTSYNTH_NB);
  long blk;

#pragma omp parallel for private(blk) schedule(dynamic, 16) num_threads(ew->nthreads)
  for (blk = 0; blk < nblk; ++blk)
    {
#ifdef _OPENMP
      double * scratch = ew->tscratch + (size_t) omp_get_thread_num() * ew->tstride;
#else
      double * scratch = ew->tscratch;
#endif
      const size_t j0 = (size_t) blk * POTSYNTH_NB;
      const size_t nb = (j0 + POTSYNTH_NB < n) ? POTSYNTH_NB : n - j0;
      double rb[POTSYNTH_NB], tb[POTSYNTH_NB], pb[POTSYNTH_NB];
      int bad[POTSYNTH_NB];
      size_t j, c;

      /* non-finite points are replaced by a harmless one and flagged */
      for (j = 0; j < nb; ++j)
        {
          bad[j] = !point_finite(r[j0 + j], theta[j0 + j], phi[j0 + j]);
          rb[j] = bad[j] ? w->a : r[j0 + j];
          tb[j] = bad[j] ? 0.5 * M_PI : theta[j0 + j];
          pb[j] = bad[j] ? 0.0 : phi[j0 + j];
        }

      potsynth_direct_block(nb, rb, tb, pb, n, out + j0, scratch, w);

      for (j = 0; j < nb; ++j)
        {
          if (bad[j])
            {
              for (c = 0; c < w->nchan; ++c)
                out[c * n + j0 + j] = NAN;
            }
        }
    }

  return POTSYNTH_SUCCESS;
}
