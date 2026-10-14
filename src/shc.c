/*
 * shc.c
 *
 * Reader for the Swarm SHC coefficient file format (static models)
 *
 * The file consists of:
 *
 * 1) comment lines starting with '#' (anywhere in the file)
 *
 * 2) a header line: lmin lmax ntimes [spline_order nstep]
 *
 * 3) a line with the ntimes epochs (decimal years)
 *
 * 4) one line per coefficient: l m value(s), with m >= 0 for g_l^m and
 *    m < 0 for h_l^|m| (Schmidt semi-normalized, nT)
 *
 * Only static models (ntimes = 1, e.g. lithospheric models such as LCS-1)
 * are supported: time-dependent (core field) models are of low degree and are
 * best evaluated with a direct sum.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_mag.h"

#define POTSYNTH_SHC_LINE            4096 /* maximum line length */

static int shc_read_line(FILE * fp, char * line, size_t * lineno);
static int shc_is_blank(const char * line);

/*
potsynth_shc_read()
  Read a static model from a Swarm SHC coefficient file

Inputs: filename - SHC file

Return: pointer to model, or NULL on error

Notes:
1) All coefficients for lmin <= l <= lmax, |m| <= l must be present exactly once

2) The coefficients are stored in shc->gh, indexed by
potsynth_idx(l,m,lmin,lmax), ready for potsynth_mag_alloc()
*/

potsynth_shc *
potsynth_shc_read(const char * filename)
{
  char line[POTSYNTH_SHC_LINE];
  char msg[POTSYNTH_SHC_LINE + 128];
  potsynth_shc * shc;
  unsigned char * seen;
  size_t lineno = 0, nread = 0, nlm;
  long hdr[5];
  FILE * fp;
  int nh, status;

  fp = fopen(filename, "r");
  if (fp == NULL)
    {
      snprintf(msg, sizeof(msg), "unable to open SHC file %s", filename);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  /* header line */
  status = shc_read_line(fp, line, &lineno);
  if (status)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: header line not found", filename);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  nh = sscanf(line, "%ld %ld %ld %ld %ld", &hdr[0], &hdr[1], &hdr[2], &hdr[3], &hdr[4]);
  if (nh < 3 || hdr[0] < 1 || hdr[1] < hdr[0] || hdr[2] < 1)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s:%zu: invalid header line (expected lmin lmax ntimes ...)",
               filename, lineno);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }
  else if (hdr[1] > POTSYNTH_LMAX)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg), "%s: lmax = %ld exceeds POTSYNTH_LMAX", filename, hdr[1]);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EUNSUP);
    }
  else if (hdr[2] != 1)
    {
      fclose(fp);
      snprintf(msg, sizeof(msg),
               "%s: time-dependent SHC files (%ld epochs) are not supported; only static models",
               filename, hdr[2]);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EUNSUP);
    }

  shc = calloc(1, sizeof(potsynth_shc));
  if (shc == NULL)
    {
      fclose(fp);
      POTSYNTH_ERROR_NULL ("unable to allocate SHC model", POTSYNTH_ENOMEM);
    }

  shc->lmin = (size_t) hdr[0];
  shc->lmax = (size_t) hdr[1];
  shc->mmax = shc->lmax;
  nlm = potsynth_nlm(shc->lmin, shc->lmax, shc->mmax);
  shc->gh = malloc(nlm * sizeof(double));
  seen = calloc(nlm, 1);
  if (shc->gh == NULL || seen == NULL)
    {
      fclose(fp);
      free(seen);
      potsynth_shc_free(shc);
      POTSYNTH_ERROR_NULL ("unable to allocate SHC coefficients", POTSYNTH_ENOMEM);
    }

  /* epoch line */
  status = shc_read_line(fp, line, &lineno);
  if (status || sscanf(line, "%lf", &(shc->epoch)) != 1 || !isfinite(shc->epoch))
    {
      fclose(fp);
      free(seen);
      potsynth_shc_free(shc);
      snprintf(msg, sizeof(msg), "%s:%zu: invalid or missing epoch line", filename, lineno);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  /* coefficient lines */
  while ((status = shc_read_line(fp, line, &lineno)) == 0)
    {
      long l, m;
      double value;
      char extra;
      int c = sscanf(line, "%ld %ld %lf %c", &l, &m, &value, &extra);
      size_t idx;
      const char * err = NULL;

      if (c != 3)
        err = (c == 4) ? "unexpected extra column" : "expected 'l m value'";
      else if (l < (long) shc->lmin || l > (long) shc->lmax || labs(m) > l)
        err = "degree/order outside the range of the header";
      else if (!isfinite(value))
        err = "non-finite coefficient";

      if (err == NULL)
        {
          idx = potsynth_idx(l, m, (long) shc->lmin, (long) shc->mmax);

          if (seen[idx])
            {
              err = "duplicate coefficient";
            }
          else
            {
              seen[idx] = 1;
              shc->gh[idx] = value;
              ++nread;
            }
        }

      if (err != NULL)
        {
          fclose(fp);
          free(seen);
          potsynth_shc_free(shc);
          snprintf(msg, sizeof(msg), "%s:%zu: %s", filename, lineno, err);
          POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
        }
    }

  fclose(fp);
  free(seen);

  if (status > 0)
    {
      potsynth_shc_free(shc);
      snprintf(msg, sizeof(msg), "%s:%zu: line too long", filename, lineno);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }
  else if (nread != nlm)
    {
      potsynth_shc_free(shc);
      snprintf(msg, sizeof(msg), "%s: %zu of %zu coefficients present", filename, nread, nlm);
      POTSYNTH_ERROR_NULL (msg, POTSYNTH_EINVAL);
    }

  return shc;
}

void
potsynth_shc_free(potsynth_shc * shc)
{
  if (shc->gh)
    free(shc->gh);

  free(shc);
}

/*
potsynth_mag_alloc_shc()
  Read a static model from a Swarm SHC file and allocate the magnetic field
model (channels B_r, B_theta, B_phi)

Inputs: filename - SHC file
        a        - reference radius (km), usually 6371.2
        rmin     - minimum radius of evaluation points (km), fast method
        rmax     - maximum radius of evaluation points (km), fast method
        params   - parameters, or NULL for defaults

Return: pointer to workspace, or NULL on error
*/

potsynth_workspace *
potsynth_mag_alloc_shc(const char * filename, const double a, const double rmin, const double rmax,
                       const potsynth_parameters * params)
{
  potsynth_shc * shc = potsynth_shc_read(filename);
  potsynth_workspace * w;

  if (shc == NULL)
    return NULL; /* reported by potsynth_shc_read() */

  w = potsynth_mag_alloc(shc->lmin, shc->lmax, shc->mmax, shc->gh, a, rmin, rmax, params);

  potsynth_shc_free(shc);

  return w;
}

/*
shc_read_line()
  Read the next line which is not a comment or blank

Inputs: fp     - file pointer
        line   - (output) line, length POTSYNTH_SHC_LINE
        lineno - (input/output) line number of the line read

Return: 0 on success, -1 at end of file, 1 if a line is too long
*/

static int
shc_read_line(FILE * fp, char * line, size_t * lineno)
{
  while (fgets(line, POTSYNTH_SHC_LINE, fp) != NULL)
    {
      const size_t len = strlen(line);

      ++(*lineno);

      if (len == POTSYNTH_SHC_LINE - 1 && line[len - 1] != '\n' && !feof(fp))
        return 1;

      if (line[0] == '#' || shc_is_blank(line))
        continue;

      return 0;
    }

  return -1;
}

static int
shc_is_blank(const char * line)
{
  while (*line)
    {
      if (!isspace((unsigned char) *line))
        return 0;

      ++line;
    }

  return 1;
}
