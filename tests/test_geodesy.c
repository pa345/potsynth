/*
 * test_geodesy.c
 *
 * Geodetic helpers: normal zonals and normal gravity of WGS84 and GRS80
 * against published / 40-digit values, geodetic to geocentric coordinates
 * (special points, round trip), subtraction of the normal field
 */

/* geocentric (r, theta) -> geodetic (lat, h) by iteration, for the round trip */
static void
test_geodesy_inverse(const double r, const double theta, const potsynth_ellipsoid * ell,
                     double * lat, double * h)
{
  const double e2 = ell->f * (2.0 - ell->f);
  const double p = r * sin(theta), z = r * cos(theta);
  double phi = atan2(z, p * (1.0 - e2)), N = ell->a;
  int k;

  for (k = 0; k < 20; ++k)
    {
      N = ell->a / sqrt(1.0 - e2 * sin(phi) * sin(phi));
      phi = atan2(z + e2 * N * sin(phi), p);
    }

  N = ell->a / sqrt(1.0 - e2 * sin(phi) * sin(phi));
  *lat = phi;
  *h = (fabs(cos(phi)) > 0.1) ? p / cos(phi) - N : z / sin(phi) - N * (1.0 - e2);
}

static void
test_geodesy(void)
{
  /* C_2n,0, n = 1..5 (40-digit evaluation of the Moritz formulas; the WGS84 values
   * equal the published ones of NIMA TR8350.2 to all 12 digits) */
  const double wgs[5] = { -0.0004841667749850007, 7.9030373351132032e-7, -1.6872496115141694e-9,
                          3.4605246839422827e-12, -2.6500222574691766e-15 };
  const double grs[5] = { -0.00048416685489572681, 7.9030407288168203e-7, -1.687251175642666e-9,
                          3.460532397805222e-12, -2.6500621766633348e-15 };
  double C[25];
  size_t k;

  /* normal zonals */
  ps_test_int(potsynth_normal_zonals(&potsynth_wgs84, 24, C), POTSYNTH_SUCCESS, "geodesy: WGS84 zonals");
  ps_test_rel(C[0], 1.0, 0.0, "geodesy: C_00 = 1");
  for (k = 1; k <= 5; ++k)
    ps_test_rel(C[2 * k], wgs[k - 1], 1.0e-12, "geodesy: WGS84 C_%zu,0", 2 * k);
  ps_test_rel(C[3], 0.0, 0.0, "geodesy: odd zonal is zero");
  ps_test_rel(C[22], 0.0, 0.0, "geodesy: zonals above degree 20 are zero");

  potsynth_normal_zonals(&potsynth_grs80, 10, C);
  for (k = 1; k <= 5; ++k)
    ps_test_rel(C[2 * k], grs[k - 1], 1.0e-12, "geodesy: GRS80 C_%zu,0", 2 * k);
  ps_test_rel(-C[2] * sqrt(5.0), 108263.0e-8, 1.0e-12, "geodesy: GRS80 J2 (defining value)");

  /* normal gravity at the equator and the poles (published values) */
  {
    const double lat[3] = { 0.0, 0.5 * M_PI, -0.5 * M_PI };
    double g[3];

    potsynth_normal_gravity(3, lat, &potsynth_wgs84, g);
    ps_test_rel(g[0], 9.7803253359, 1.0e-11, "geodesy: WGS84 gamma_e");
    ps_test_rel(g[1], 9.8321849378, 1.0e-11, "geodesy: WGS84 gamma_p (north)");
    ps_test_rel(g[2], 9.8321849378, 1.0e-11, "geodesy: WGS84 gamma_p (south)");

    potsynth_normal_gravity(3, lat, &potsynth_grs80, g);
    ps_test_rel(g[0], 9.7803267715, 1.0e-11, "geodesy: GRS80 gamma_e");
    ps_test_rel(g[1], 9.8321863685, 1.0e-11, "geodesy: GRS80 gamma_p");
  }

  /* geodetic -> geocentric: equator, poles, round trip */
  {
    const potsynth_ellipsoid * ell = &potsynth_wgs84;
    const double b = ell->a * (1.0 - ell->f);
    double lat[4] = { 0.0, 0.5 * M_PI, -0.5 * M_PI, 0.0 }, h[4] = { 0.0, 0.0, 0.0, 1000.0 };
    double r[4], th[4], elat = 0.0, eh = 0.0;
    size_t i;

    potsynth_geodetic2geocentric(4, lat, h, ell, r, th);
    ps_test_rel(r[0], ell->a, 1.0e-15, "geodesy: equator r = a");
    ps_test_rel(th[0], 0.5 * M_PI, 1.0e-15, "geodesy: equator theta = pi/2");
    ps_test_rel(r[1], b, 1.0e-15, "geodesy: north pole r = b");
    ps_test_abs(th[1], 0.0, 1.0e-15, "geodesy: north pole theta = 0");
    ps_test_rel(th[2], M_PI, 1.0e-15, "geodesy: south pole theta = pi");
    ps_test_rel(r[3], ell->a + 1000.0, 1.0e-15, "geodesy: equator, h = 1 km");

    for (i = 0; i < 200; ++i)
      {
        double la = M_PI * (test_urand() - 0.5), hh = -500.0 + 1.0e6 * test_urand(), rr, tt, la2, h2;

        potsynth_geodetic2geocentric(1, &la, &hh, ell, &rr, &tt);
        test_geodesy_inverse(rr, tt, ell, &la2, &h2);
        elat = fmax(elat, fabs(la2 - la));
        eh = fmax(eh, fabs(h2 - hh));
      }

    ps_test(elat > 1.0e-14, "geodesy: round trip latitude (error %.2e rad)", elat);
    ps_test(eh > 1.0e-7, "geodesy: round trip height (error %.2e m)", eh);

    /* in place */
    potsynth_geodetic2geocentric(4, lat, h, ell, lat, h);
    ps_test(lat[0] != r[0] || h[1] != th[1], "geodesy: in-place conversion");
  }

  /* subtracting the normal field */
  {
    const size_t L = 6;
    const double GM = 3.986004415e14, a = 6378136.3;
    double * coef = test_random_coef(0, L, L);
    double * c0 = malloc(potsynth_nlm(0, L, L) * sizeof(double));
    double * c2 = test_random_coef(2, L, L);
    double * c2b = malloc(potsynth_nlm(2, L, L) * sizeof(double));
    const double q = potsynth_wgs84.a / a;
    int ok = 1;
    size_t i, l;

    memcpy(c0, coef, potsynth_nlm(0, L, L) * sizeof(double));
    potsynth_normal_zonals(&potsynth_wgs84, L, C);
    ps_test_int(potsynth_grav_subtract_normal(0, L, L, coef, GM, a, &potsynth_wgs84), POTSYNTH_SUCCESS,
                "geodesy: subtract normal field");

    for (i = 0; i < potsynth_nlm(0, L, L); ++i)
      {
        double expect = c0[i];

        for (l = 0; l <= L; l += 2)
          {
            if (i == (size_t) potsynth_idx((long) l, 0L, 0L, (long) L))
              expect -= C[l] * (potsynth_wgs84.GM / GM) * pow(q, (double) l);
          }

        ok = ok && fabs(coef[i] - expect) <= 1.0e-15 * fmax(1.0, fabs(expect));
      }

    ps_test(!ok, "geodesy: normal zonals rescaled to the model GM and a");

    /* lmin = 2: degree 0 is not present and must not be written */
    memcpy(c2b, c2, potsynth_nlm(2, L, L) * sizeof(double));
    potsynth_grav_subtract_normal(2, L, L, c2, GM, a, &potsynth_wgs84);
    ok = 1;
    for (i = 0; i < potsynth_nlm(2, L, L); ++i)
      {
        const int zonal = (i == (size_t) potsynth_idx(2L, 0L, 2L, (long) L) ||
                           i == (size_t) potsynth_idx(4L, 0L, 2L, (long) L) ||
                           i == (size_t) potsynth_idx(6L, 0L, 2L, (long) L));

        ok = ok && (zonal ? c2[i] != c2b[i] : c2[i] == c2b[i]);
      }

    ps_test(!ok, "geodesy: lmin = 2, only even zonals changed");

    test_last_error();
    ps_test(potsynth_grav_subtract_normal(0, L, L, coef, -GM, a, &potsynth_wgs84) != POTSYNTH_EINVAL,
            "geodesy: GM < 0 rejected");
    test_last_error();

    free(coef);
    free(c0);
    free(c2);
    free(c2b);
  }
}
