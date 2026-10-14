/*
 * test_common.c
 *
 * Test helpers with the same interface and behavior as gsl_test (GSL):
 * failures are always printed, passes only if the environment variable
 * POTSYNTH_TEST_VERBOSE is set; ps_test_summary() returns the exit status.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

#include "test_common.h"

static unsigned int tests = 0;
static unsigned int passed = 0;
static unsigned int failed = 0;
static int verbose = -1;
static int last_error = 0;
static unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;

static int
test_verbose(void)
{
  if (verbose < 0)
    verbose = (getenv("POTSYNTH_TEST_VERBOSE") != NULL);

  return verbose;
}

/* record and report the result of one test */
static void
test_update(const int status, const char * prefix, const char * fmt, va_list ap)
{
  ++tests;

  if (status == 0)
    ++passed;
  else
    ++failed;

  if (status || test_verbose())
    {
      printf("%s: ", status ? "FAIL" : "PASS");
      if (prefix)
        printf("%s ", prefix);
      vprintf(fmt, ap);
      printf("\n");
      fflush(stdout);
    }
}

void
ps_test(const int status, const char * test_description, ...)
{
  va_list ap;

  va_start(ap, test_description);
  test_update(status, NULL, test_description, ap);
  va_end(ap);
}

void
ps_test_rel(const double result, const double expected, const double relative_error,
            const char * test_description, ...)
{
  char prefix[128];
  va_list ap;
  int status;

  if (isnan(result) || isnan(expected))
    status = isnan(result) != isnan(expected);
  else if (isinf(result) || isinf(expected))
    status = (result != expected);
  else if (expected != 0.0)
    status = (fabs(result - expected) / fabs(expected) > relative_error);
  else
    status = (fabs(result) > relative_error);

  snprintf(prefix, sizeof(prefix), "(%.18g observed vs %.18g expected)", result, expected);

  va_start(ap, test_description);
  test_update(status, status ? prefix : NULL, test_description, ap);
  va_end(ap);
}

void
ps_test_abs(const double result, const double expected, const double absolute_error,
            const char * test_description, ...)
{
  char prefix[128];
  va_list ap;
  int status;

  if (isnan(result) || isnan(expected))
    status = isnan(result) != isnan(expected);
  else if (isinf(result) || isinf(expected))
    status = (result != expected);
  else
    status = (fabs(result - expected) > absolute_error);

  snprintf(prefix, sizeof(prefix), "(%.18g observed vs %.18g expected)", result, expected);

  va_start(ap, test_description);
  test_update(status, status ? prefix : NULL, test_description, ap);
  va_end(ap);
}

void
ps_test_int(const int result, const int expected, const char * test_description, ...)
{
  char prefix[64];
  va_list ap;
  const int status = (result != expected);

  snprintf(prefix, sizeof(prefix), "(%d observed vs %d expected)", result, expected);

  va_start(ap, test_description);
  test_update(status, status ? prefix : NULL, test_description, ap);
  va_end(ap);
}

/* return EXIT_SUCCESS if all tests passed */
int
ps_test_summary(void)
{
  if (test_verbose())
    printf("%u tests, passed %u, failed %u.\n", tests, passed, failed);

  if (failed != 0 || tests != passed + failed)
    {
      if (!test_verbose())
        printf("%u tests, passed %u, failed %u.\n", tests, passed, failed);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}

/* uniform random number in [0,1) (xorshift) */
double
test_urand(void)
{
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;

  return (double) (rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* standard normal random number (Box-Muller) */
double
test_nrand(void)
{
  const double u1 = test_urand() + 1.0e-300;
  const double u2 = test_urand();

  return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

/* potsynth error handler for the tests: record the error code, print nothing */
void
test_error_handler(const char * reason, const char * file, int line, int potsynth_errno)
{
  (void) reason;
  (void) file;
  (void) line;

  last_error = potsynth_errno;
}

/* return and clear the last recorded error code */
int
test_last_error(void)
{
  const int e = last_error;

  last_error = 0;

  return e;
}
