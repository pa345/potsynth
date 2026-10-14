/*
 * test_shc.c
 *
 * SHC reader: valid files (comments, blank lines, lmin > 1) and malformed
 * files (missing, duplicate, out of range or extra entries, time-dependent
 * files, missing file)
 */

#define TEST_SHC_FILE                "test_shc_tmp.shc"

/* write a file with the given contents */
static void
test_shc_write(const char * contents)
{
  FILE * fp = fopen(TEST_SHC_FILE, "w");

  fputs(contents, fp);
  fclose(fp);
}

/* reading the current file must fail with the given error code */
static void
test_shc_fail(const char * desc, const int expected)
{
  potsynth_shc * shc;

  test_last_error();
  shc = potsynth_shc_read(TEST_SHC_FILE);

  ps_test(shc != NULL, "shc: %s rejected", desc);
  ps_test_int(test_last_error(), expected, "shc: %s error code", desc);

  if (shc)
    potsynth_shc_free(shc);
}

/*
test_shc_valid()
  Read a small valid file with lmin = 2 and compare the model built from the
file with the model built from the coefficient vector
*/

static void
test_shc_valid(void)
{
  const char * contents =
    "# test model\n"
    "# with comments\n"
    "  2 3   1   1   1\n"
    "\n"
    "          2015.0\n"
    "  2    0  1.5\n"
    "# comment in the middle\n"
    "  2    1  2.5\n"
    "  2   -1 -3.5\n"
    "  2    2  4.0e-1\n"
    "  2   -2 -5.0E-1\n"
    "  3    0  0.25\n"
    "  3    1  0.125\n"
    "  3   -1 -0.0625\n"
    "  3    2  1\n"
    "  3   -2 -1\n"
    "  3    3  2\n"
    "  3   -3 -2\n";
  const double a = 6371.2;
  const size_t n = 50;
  double r[50], theta[50], phi[50], B1[150], B2[150];
  potsynth_parameters params = potsynth_default_parameters();
  potsynth_shc * shc;
  potsynth_workspace * w1, * w2;
  potsynth_eval_workspace * ew1, * ew2;
  size_t i;

  test_shc_write(contents);

  shc = potsynth_shc_read(TEST_SHC_FILE);
  ps_test(shc == NULL, "shc: valid file read");
  if (shc == NULL)
    return;

  ps_test_int((int) shc->lmin, 2, "shc: lmin");
  ps_test_int((int) shc->lmax, 3, "shc: lmax");
  ps_test_rel(shc->epoch, 2015.0, 0.0, "shc: epoch");
  ps_test_rel(shc->gh[potsynth_idx(2L, 0L, 2L, 3L)], 1.5, 0.0, "shc: g20");
  ps_test_rel(shc->gh[potsynth_idx(2L, -1L, 2L, 3L)], -3.5, 0.0, "shc: h21");
  ps_test_rel(shc->gh[potsynth_idx(2L, -2L, 2L, 3L)], -0.5, 0.0, "shc: h22");
  ps_test_rel(shc->gh[potsynth_idx(3L, 3L, 2L, 3L)], 2.0, 0.0, "shc: g33");
  ps_test_rel(shc->gh[potsynth_idx(3L, -3L, 2L, 3L)], -2.0, 0.0, "shc: h33");

  /* model from the file = model from the vector (bitwise) */
  for (i = 0; i < n; ++i)
    {
      r[i] = a + 100.0 * test_urand();
      theta[i] = M_PI * test_urand();
      phi[i] = 2.0 * M_PI * test_urand();
    }

  params.method = POTSYNTH_METHOD_DIRECT;
  w1 = potsynth_mag_alloc_shc(TEST_SHC_FILE, a, a, a + 100.0, &params);
  w2 = potsynth_mag_alloc(shc->lmin, shc->lmax, shc->mmax, shc->gh, a, a, a + 100.0, &params);
  ew1 = potsynth_eval_alloc(0, w1);
  ew2 = potsynth_eval_alloc(0, w2);
  potsynth_eval(n, r, theta, phi, B1, ew1);
  potsynth_eval(n, r, theta, phi, B2, ew2);
  ps_test(memcmp(B1, B2, sizeof(B1)) != 0, "shc: model from file equals model from vector");

  potsynth_eval_free(ew1);
  potsynth_eval_free(ew2);
  potsynth_free(w1);
  potsynth_free(w2);
  potsynth_shc_free(shc);
}

static void
test_shc(void)
{
  test_shc_valid();

  /* h11 missing */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 1.0\n1 1 2.0\n");
  test_shc_fail("missing coefficient", POTSYNTH_EINVAL);

  /* g10 twice */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 1.0\n1 1 2.0\n1 -1 3.0\n1 0 1.0\n");
  test_shc_fail("duplicate coefficient", POTSYNTH_EINVAL);

  /* degree 2 in an lmax = 1 file */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 1.0\n1 1 2.0\n1 -1 3.0\n2 0 1.0\n");
  test_shc_fail("degree out of range", POTSYNTH_EINVAL);

  /* |m| > l */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 1.0\n1 2 2.0\n1 -1 3.0\n");
  test_shc_fail("order out of range", POTSYNTH_EINVAL);

  /* two value columns in a one-epoch file */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 1.0 2.0\n1 1 2.0\n1 -1 3.0\n");
  test_shc_fail("extra column", POTSYNTH_EINVAL);

  /* non-numeric value */
  test_shc_write("1 1 1 1 1\n2015.0\n1 0 abc\n1 1 2.0\n1 -1 3.0\n");
  test_shc_fail("non-numeric value", POTSYNTH_EINVAL);

  /* time-dependent model */
  test_shc_write("1 1 2 2 1\n2015.0 2020.0\n1 0 1.0 1.1\n1 1 2.0 2.1\n1 -1 3.0 3.1\n");
  test_shc_fail("time-dependent file", POTSYNTH_EUNSUP);

  /* header with lmax < lmin */
  test_shc_write("3 1 1 1 1\n2015.0\n");
  test_shc_fail("invalid header", POTSYNTH_EINVAL);

  /* no header */
  test_shc_write("# only comments\n");
  test_shc_fail("missing header", POTSYNTH_EINVAL);

  remove(TEST_SHC_FILE);

  test_last_error();
  ps_test(potsynth_shc_read("nonexistent_file.shc") != NULL, "shc: nonexistent file rejected");
  ps_test_int(test_last_error(), POTSYNTH_EINVAL, "shc: nonexistent file error code");
}
