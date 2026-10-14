/*
 * gfc.c
 *
 * Reader for the ICGEM gravity field coefficient format (.gfc, static models)
 *
 * The file consists of:
 *
 * 1) a header, ending with a line "end_of_head". Lines starting with a
 *    keyword define the model; other lines are free text:
 *
 *    product_type            gravity_field
 *    modelname               EGM2008
 *    earth_gravity_constant  0.3986004415E+15   (m^3/s^2)
 *    radius                  0.63781363E+07     (m)
 *    max_degree              2190
 *    norm                    fully_normalized   (default; or unnormalized)
 *    tide_system             tide_free
 *    errors                  calibrated         (ignored)
 *
 * 2) one line per coefficient: gfc l m C S [sigma_C sigma_S]. Numbers may use
 *    Fortran exponents (1.0D-09)
 *
 * Only static models are supported: files with time-variable terms (gfct,
 * trnd, dot, acos, asin) are rejected. Coefficients absent from the file are
 * zero (models often omit degrees 0 and 1).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_grav.h"

#define POTSYNTH_GFC_LINE            4096 /* maximum line length */
#define POTSYNTH_GFC_TOKEN           128  /* maximum token length */

static int gfc_read_line(FILE * fp, char * line, size_t * lineno);
static const char * gfc_token(const char * p, char * tok);
static int gfc_strieq(const char * a, const char * b);
static int gfc_double(const char * tok, double * x);
static double gfc_unnorm_factor(const size_t l, const size_t m);

/*
potsynth_gfc_read()
  Read a static gravity field model from an ICGEM .gfc file

Inputs: filename - .gfc file
        lmax     - maximum degree to read (truncation), or 0 for max_degree of
                   the file

Return: pointer to model, or NULL on error

Notes:
1) The coefficients are stored in gfc->coef, 4-pi fully normalized (unnormalized
files are converted), indexed by potsynth_idx(l,m,0,lmax), ready for
potsynth_grav_alloc()

2) The coefficients are used as given: the tide system is reported in
gfc->tide_system but not converted
*/

potsynth_gfc *
potsynth_gfc_read(const char * filename, const size_t lmax)
{
  char line[POTSYNTH_GFC_LINE];
  char tok[POTSYNTH_GFC_TOKEN];
  char msg[POTSYNTH_GFC_LINE + 256];
  potsynth_gfc * gfc;
  unsigned char * seen;
  size_t lineno = 0, nlm;
  long max_degree = -1;
  double GM = -1.0, a = -1.0;
  int unnormalized = 0, end_of_head = 0, status;
  char modelname[128] = "", tide_system[64] = "unknown";
  FILE * fp;

  fp = fopen(filename, "r");
  if (fp == NULL)
    {
      snprintf(msg, sizeof(msg), "unable to open gfc file %s", filename);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  /* header */
  while ((status = gfc_read_line(fp, line, &lineno)) == 0)
    {
      const char * p = gfc_token(line, tok);
      char val[POTSYNTH_GFC_TOKEN];
      const char * err = NULL;

      if (gfc_strieq(tok, "end_of_head"))
        {
          end_of_head = 1;
          break;
        }

      gfc_token(p, val);

      if (gfc_strieq(tok, "product_type"))
        {
          if (!gfc_strieq(val, "gravity_field"))
            err = "product_type is not gravity_field";
        }
      else if (gfc_strieq(tok, "modelname"))
        {
          snprintf(modelname, sizeof(modelname), "%.127s", val);
        }
      else if (gfc_strieq(tok, "earth_gravity_constant"))
        {
          if (gfc_double(val, &GM) || !(GM > 0.0) || !isfinite(GM))
            err = "invalid earth_gravity_constant";
        }
      else if (gfc_strieq(tok, "radius"))
        {
          if (gfc_double(val, &a) || !(a > 0.0) || !isfinite(a))
            err = "invalid radius";
        }
      else if (gfc_strieq(tok, "max_degree"))
        {
          char * end;

          max_degree = strtol(val, &end, 10);
          if (*val == '\0' || *end != '\0' || max_degree < 0)
            err = "invalid max_degree";
        }
      else if (gfc_strieq(tok, "norm"))
        {
          if (gfc_strieq(val, "unnormalized"))
            unnormalized = 1;
          else if (!gfc_strieq(val, "fully_normalized"))
            err = "norm must be fully_normalized or unnormalized";
        }
      else if (gfc_strieq(tok, "tide_system"))
        {
          snprintf(tide_system, sizeof(tide_system), "%.63s", val);
        }

      if (err != NULL)
        {
          fclose(fp);
          snprintf(msg, sizeof(msg), "%s:%zu: %s", filename, lineno, err);
          POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
        }
    }

  if (status > 0 || !end_of_head)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: %s", filename,
               (status > 0) ? "line too long in header" : "end_of_head not found");
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }
  else if (GM < 0.0 || a < 0.0 || max_degree < 0)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: header lacks earth_gravity_constant, radius or max_degree",
               filename);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }
  else if ((long) lmax > max_degree)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: requested lmax = %zu exceeds max_degree = %ld",
               filename, lmax, max_degree);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }
  else if (lmax == 0 && max_degree > POTSYNTH_LMAX)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: max_degree = %ld exceeds POTSYNTH_LMAX; give a smaller lmax",
               filename, max_degree);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EUNSUP);
    }
  else if (lmax > POTSYNTH_LMAX)
    {
      fclose(fp);
      POTSYNTH_ERROR_NULL ("lmax exceeds POTSYNTH_LMAX", POTSYNTH_EUNSUP);
    }

  gfc = calloc(1, sizeof(potsynth_gfc));
  if (gfc == NULL)
    {
      fclose(fp);
      POTSYNTH_ERROR_NULL ("unable to allocate gfc model", POTSYNTH_ENOMEM);
    }

  gfc->lmin = 0;
  gfc->lmax = (lmax > 0) ? lmax : (size_t) max_degree;
  gfc->mmax = gfc->lmax;
  gfc->GM = GM;
  gfc->a = a;
  snprintf(gfc->modelname, sizeof(gfc->modelname), "%s", modelname);
  snprintf(gfc->tide_system, sizeof(gfc->tide_system), "%s", tide_system);

  nlm = potsynth_nlm(gfc->lmin, gfc->lmax, gfc->mmax);
  gfc->coef = calloc(nlm, sizeof(double));
  seen = calloc(nlm, 1);
  if (gfc->coef == NULL || seen == NULL)
    {
      fclose(fp);
      free(seen);
      potsynth_gfc_free(gfc);
      POTSYNTH_ERROR_NULL ("unable to allocate gfc coefficients", POTSYNTH_ENOMEM);
    }

  /* coefficient lines */
  while ((status = gfc_read_line(fp, line, &lineno)) == 0)
    {
      const char * p = gfc_token(line, tok);
      const char * err = NULL;
      char tl[POTSYNTH_GFC_TOKEN], tm[POTSYNTH_GFC_TOKEN], tc[POTSYNTH_GFC_TOKEN], ts[POTSYNTH_GFC_TOKEN];
      double C, S;
      long l = -1, m = -1;
      int unsup = 0;

      if (gfc_strieq(tok, "gfct") || gfc_strieq(tok, "trnd") || gfc_strieq(tok, "dot") ||
          gfc_strieq(tok, "acos") || gfc_strieq(tok, "asin"))
        {
          unsup = 1;
          err = "time-variable models are not supported; only static models (gfc)";
        }
      else if (!gfc_strieq(tok, "gfc"))
        {
          err = "unknown keyword (expected gfc)";
        }
      else
        {
          char * end1, * end2;

          p = gfc_token(p, tl);
          p = gfc_token(p, tm);
          p = gfc_token(p, tc);
          gfc_token(p, ts);

          l = strtol(tl, &end1, 10);
          m = strtol(tm, &end2, 10);

          if (*tl == '\0' || *end1 != '\0' || *tm == '\0' || *end2 != '\0')
            err = "expected 'gfc l m C S'";
          else if (gfc_double(tc, &C) || gfc_double(ts, &S) || !isfinite(C) || !isfinite(S))
            err = "invalid coefficient";
          else if (l < 0 || l > max_degree || m < 0 || m > l)
            err = "degree/order outside 0 <= m <= l <= max_degree";
        }

      if (err == NULL && l <= (long) gfc->lmax)
        {
          const size_t ip = potsynth_idx(l, m, 0L, (long) gfc->mmax);
          const size_t im = potsynth_idx(l, -m, 0L, (long) gfc->mmax);
          const double f = unnormalized ? gfc_unnorm_factor((size_t) l, (size_t) m) : 1.0;

          if (seen[ip])
            {
              err = "duplicate coefficient";
            }
          else if (m == 0 && S != 0.0)
            {
              err = "nonzero S coefficient for m = 0";
            }
          else
            {
              seen[ip] = 1;
              gfc->coef[ip] = f * C;
              if (m > 0)
                gfc->coef[im] = f * S;
            }
        }

      if (err != NULL)
        {
          fclose(fp);
          free(seen);
          potsynth_gfc_free(gfc);
          snprintf(msg, sizeof(msg), "%s:%zu: %s", filename, lineno, err);
          POTSYNTH_ERROR_NULL (msg, unsup ? POTSYNTH_EUNSUP : POTSYNTH_EINVAL);
        }
    }

  fclose(fp);
  free(seen);

  if (status > 0)
    {
      potsynth_gfc_free(gfc);
      snprintf(msg, sizeof(msg), "%s:%zu: line too long", filename, lineno);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  return gfc;
}

void
potsynth_gfc_free(potsynth_gfc * gfc)
{
  if (gfc->coef)
    free(gfc->coef);

  free(gfc);
}

/*
potsynth_grav_alloc_gfc()
  Read a static model from an ICGEM .gfc file and allocate the gravity field
model

Inputs: filename   - .gfc file
        lmax       - maximum degree (truncation), or 0 for all degrees of the file
        rmin       - minimum radius of evaluation points (m), fast method
        rmax       - maximum radius of evaluation points (m), fast method
        quantities - POTSYNTH_GRAV_xxx flags
        params     - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error

Notes:
1) GM and the reference radius are taken from the file
*/

potsynth_workspace *
potsynth_grav_alloc_gfc(const char * filename, const size_t lmax, const double rmin, const double rmax,
                        const int quantities, const potsynth_parameters * params)
{
  potsynth_gfc * gfc = potsynth_gfc_read(filename, lmax);
  potsynth_workspace * w;

  if (gfc == NULL)
    return NULL; /* reported by potsynth_gfc_read() */

  w = potsynth_grav_alloc(gfc->lmin, gfc->lmax, gfc->mmax, gfc->coef, gfc->GM, gfc->a,
                          rmin, rmax, quantities, params);

  potsynth_gfc_free(gfc);

  return w;
}

/*
gfc_read_line()
  Read the next line which is not blank

Inputs: fp     - file pointer
        line   - (output) line, length POTSYNTH_GFC_LINE
        lineno - (input/output) line number of the line read

Return: 0 on success, -1 at end of file, 1 if a line is too long
*/

static int
gfc_read_line(FILE * fp, char * line, size_t * lineno)
{
  while (fgets(line, POTSYNTH_GFC_LINE, fp) != NULL)
    {
      const size_t len = strlen(line);
      const char * p = line;

      ++(*lineno);

      if (len == POTSYNTH_GFC_LINE - 1 && line[len - 1] != '\n' && !feof(fp))
        return 1;

      while (*p && isspace((unsigned char) *p))
        ++p;

      if (*p == '\0')
        continue;

      return 0;
    }

  return -1;
}

/*
gfc_token()
  Extract the next whitespace-separated token

Inputs: p   - position in line
        tok - (output) token, length POTSYNTH_GFC_TOKEN, truncated if longer;
              empty at end of line

Return: position after the token
*/

static const char *
gfc_token(const char * p, char * tok)
{
  size_t n = 0;

  while (*p && isspace((unsigned char) *p))
    ++p;

  while (*p && !isspace((unsigned char) *p))
    {
      if (n < POTSYNTH_GFC_TOKEN - 1)
        tok[n++] = *p;

      ++p;
    }

  tok[n] = '\0';

  return p;
}

/* case-insensitive string equality */
static int
gfc_strieq(const char * a, const char * b)
{
  while (*a && *b)
    {
      if (tolower((unsigned char) *a) != tolower((unsigned char) *b))
        return 0;

      ++a;
      ++b;
    }

  return (*a == '\0' && *b == '\0');
}

/*
gfc_double()
  Convert a token to double, accepting Fortran exponents (D or d)

Return: 0 on success, 1 if the token is not a number
*/

static int
gfc_double(const char * tok, double * x)
{
  char buf[POTSYNTH_GFC_TOKEN];
  char * end;
  size_t i;

  for (i = 0; tok[i] != '\0' && i < POTSYNTH_GFC_TOKEN - 1; ++i)
    buf[i] = (tok[i] == 'D' || tok[i] == 'd') ? 'e' : tok[i];

  buf[i] = '\0';

  if (buf[0] == '\0')
    return 1;

  *x = strtod(buf, &end);

  return (*end != '\0');
}

/*
gfc_unnorm_factor()
  Factor converting an unnormalized coefficient to 4-pi fully normalized:

C_lm(normalized) = C_lm(unnormalized) / N_lm,
N_lm = sqrt( (2 - delta_m0) (2l + 1) (l-m)! / (l+m)! )
*/

static double
gfc_unnorm_factor(const size_t l, const size_t m)
{
  const double k = (m == 0) ? 1.0 : 2.0;
  const double logN = 0.5 * (log(k * (2.0 * l + 1.0)) + lgamma(l - m + 1.0) - lgamma(l + m + 1.0));

  return exp(-logN);
}
