/*
 * test_common.h
 *
 * Test helpers with the same interface and behavior as gsl_test (GSL), so
 * that the tests do not depend on GSL
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_test_common_h
#define INCLUDED_test_common_h

#include <stddef.h>

#ifdef __GNUC__
#define TEST_PRINTF_ATTR(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define TEST_PRINTF_ATTR(fmt_idx, arg_idx)
#endif

/*
 * Prototypes
 */

void ps_test(const int status, const char * test_description, ...) TEST_PRINTF_ATTR(2, 3);
void ps_test_rel(const double result, const double expected, const double relative_error,
                 const char * test_description, ...) TEST_PRINTF_ATTR(4, 5);
void ps_test_abs(const double result, const double expected, const double absolute_error,
                 const char * test_description, ...) TEST_PRINTF_ATTR(4, 5);
void ps_test_int(const int result, const int expected, const char * test_description, ...)
  TEST_PRINTF_ATTR(3, 4);
int ps_test_summary(void);

double test_urand(void);
double test_nrand(void);
void test_error_handler(const char * reason, const char * file, int line, int potsynth_errno);
int test_last_error(void);

#endif /* INCLUDED_test_common_h */
