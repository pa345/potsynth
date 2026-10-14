/*
 * test.c
 *
 * potsynth test suite. The test is compiled from the library sources so that
 * internal components can be tested directly (white-box), as well as the
 * public interface:
 *
 * test_util.c    - scaled powers, cos/sin(m phi), FFT lengths
 * test_quad.c    - Gauss-Legendre quadrature, window transform, window width
 * test_fft.c     - pocketfft conventions relied on by the set-up
 * test_svd.c     - one-sided Jacobi SVD
 * test_radial.c  - radial basis
 * test_alf.c     - Legendre columns, accumulation kernel, column scaling
 * test_alf_gsl.c - Legendre columns against GSL (if HAVE_GSL)
 * test_ref.c     - magnetic field against extended precision reference values
 * test_eval.c    - fast method against direct method
 * test_api.c     - API contract (NaN inputs, domain errors, reproducibility)
 * test_threads.c - one workspace evaluated concurrently from several threads
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

#ifdef HAVE_PTHREADS
#include <pthread.h>
#endif

#ifdef HAVE_GSL
#include <gsl/gsl_sf_alf.h>
#endif

#include "potsynth_internal.h"
#include "potsynth_mag.h"
#include "pocketfft.h"

#include "test_common.h"

/* directory containing reference data files; normally set by CMake */
#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "data"
#endif

#include "test_util.c"
#include "test_quad.c"
#include "test_fft.c"
#include "test_svd.c"
#include "test_radial.c"
#include "test_alf.c"
#include "test_alf_gsl.c"
#include "test_ref.c"
#include "test_eval.c"
#include "test_api.c"
#include "test_threads.c"

int
main(void)
{
  /* errors are expected in some tests: record them instead of printing */
  potsynth_set_error_handler(test_error_handler);

  test_util();
  test_quad();
  test_fft();
  test_svd();
  test_radial();
  test_alf();
  test_alf_gsl();
  test_ref();
  test_eval();
  test_api();
  test_threads();

  exit (ps_test_summary());
}
