/*
 * geodesy.c
 *
 * Small geodetic helpers for the gravity front-end: reference ellipsoids,
 * geodetic to geocentric coordinates, the normal field (even zonal
 * coefficients) and normal gravity on the ellipsoid
 *
 * The derived constants of an ellipsoid (J2, normal gravity at the equator and
 * the poles) follow from its four defining parameters (a, f, GM, omega) by the
 * closed formulas of Moritz, Geodetic Reference System 1980, J. Geodesy 74
 * (2000), 128-133:
 *
 * e^2 = 2f - f^2, e'^2 = e^2/(1 - e^2), b = a (1 - f), m = omega^2 a^2 b / GM,
 * q0  = ((1 + 3/e'^2) atan(e') - 3/e') / 2
 *     = sum_{n>=1} (-1)^(n+1) 2n e'^(2n+1) / ((2n+1)(2n+3)),
 * q0' = 3 (1 + 1/e'^2) (1 - atan(e')/e') - 1
 *     = sum_{n>=1} (-1)^(n+1) 6 e'^(2n) / ((2n+1)(2n+3)),
 * J2  = e^2/3 (1 - 2/15 m e'/q0),
 * gamma_e = GM/(a b) (1 - m - m/6 e' q0'/q0),  gamma_p = GM/a^2 (1 + m/3 e' q0'/q0)
 *
 * The closed forms of q0 and q0' cancel (they lose about 4 digits in double
 * precision); the series are used instead.
 *
 * and the zonal coefficients of the normal gravitational potential are
 *
 * J_2n = (-1)^(n+1) 3 e^2n / ((2n+1)(2n+3)) (1 - n + 5n J2/e^2),
 * C_2n,0 = -J_2n / sqrt(4n+1)    (4-pi normalized)
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <math.h>

#include "potsynth_internal.h"
#include "potsynth_grav.h"

#define POTSYNTH_NORMAL_LMAX         20 /* highest degree of the normal field (J_2n below 1e-20) */

const potsynth_ellipsoid potsynth_wgs84 = { 6378137.0, 1.0 / 298.257223563, 3.986004418e14, 7.292115e-5 };
const potsynth_ellipsoid potsynth_grs80 = { 6378137.0, 1.0 / 298.257222101, 3.986005e14, 7.292115e-5 };

static int geodesy_check(const potsynth_ellipsoid * ell);
static void geodesy_derived(const potsynth_ellipsoid * ell, double * J2, double * gamma_e,
                            double * gamma_p);

/*
potsynth_geodetic2geocentric()
  Convert geodetic latitude and ellipsoidal height to geocentric radius and
colatitude (the longitude is the same in both systems)

Inputs: n     - number of points
        lat   - geodetic latitude (radians), length n
        h     - ellipsoidal height (m), length n
        ell   - reference ellipsoid
        r     - (output) geocentric radius (m), length n
        theta - (output) geocentric colatitude (radians), length n

Return: success or error

Notes:
1) r and theta may be the same arrays as lat and h
*/

int
potsynth_geodetic2geocentric(const size_t n, const double * lat, const double * h,
                             const potsynth_ellipsoid * ell, double * r, double * theta)
{
  const double e2 = ell->f * (2.0 - ell->f);
  size_t i;

  if (geodesy_check(ell))
    {
      POTSYNTH_ERROR ("invalid ellipsoid", POTSYNTH_EINVAL);
    }

  for (i = 0; i < n; ++i)
    {
      const double sl = sin(lat[i]), cl = cos(lat[i]);
      const double N = ell->a / sqrt(1.0 - e2 * sl * sl); /* prime vertical radius */
      const double p = (N + h[i]) * cl;                     /* distance from the axis */
      const double z = (N * (1.0 - e2) + h[i]) * sl;

      r[i] = hypot(p, z);
      theta[i] = atan2(p, z);
    }

  return POTSYNTH_SUCCESS;
}

/*
potsynth_normal_zonals()
  Zonal coefficients of the normal gravitational potential of an ellipsoid,
4-pi normalized, in the ellipsoid's own scaling (GM and a of the ellipsoid):

U = GM/r sum_l (a/r)^l C_l Pbar_l^0(cos theta)

Inputs: ell  - reference ellipsoid
        lmax - maximum degree
        C    - (output) C[l], l = 0..lmax; C[0] = 1, odd degrees and degrees
               above 20 are 0

Return: success or error

Notes:
1) U is the gravitational part of the normal potential (no centrifugal term)
*/

int
potsynth_normal_zonals(const potsynth_ellipsoid * ell, const size_t lmax, double * C)
{
  const double e2 = ell->f * (2.0 - ell->f);
  double J2, ge, gp, e2n = 1.0;
  size_t l, k;

  if (geodesy_check(ell))
    {
      POTSYNTH_ERROR ("invalid ellipsoid", POTSYNTH_EINVAL);
    }

  geodesy_derived(ell, &J2, &ge, &gp);

  for (l = 0; l <= lmax; ++l)
    C[l] = 0.0;

  C[0] = 1.0;

  for (k = 1; 2 * k <= lmax && 2 * k <= POTSYNTH_NORMAL_LMAX; ++k)
    {
      const double dk = (double) k;
      double J2k;

      e2n *= e2;
      J2k = ((k & 1) ? 3.0 : -3.0) * e2n / ((2.0 * dk + 1.0) * (2.0 * dk + 3.0)) *
            (1.0 - dk + 5.0 * dk * J2 / e2);
      C[2 * k] = -J2k / sqrt(4.0 * dk + 1.0);
    }

  return POTSYNTH_SUCCESS;
}

/*
potsynth_grav_subtract_normal()
  Subtract the normal gravitational potential of an ellipsoid from a model,
giving the coefficients of the disturbing potential T = V - U

Inputs: lmin - minimum degree of the coefficient vector
        lmax - maximum degree
        mmax - maximum order
        coef - (input/output) 4-pi normalized coefficients of the model, indexed
               by potsynth_idx(l,m,lmin,mmax), in the scaling of GM and a
        GM   - GM of the model (m^3/s^2)
        a    - reference radius of the model (m)
        ell  - reference ellipsoid

Return: success or error

Notes:
1) The normal zonals are rescaled to the model: C_l (GM_ell/GM) (a_ell/a)^l.
All even degrees up to min(lmax, 20) are subtracted, including degree 0, so
that T contains the zero-degree term (GM - GM_ell)/r. Degrees below lmin are
not present in the vector and are skipped

2) Only the gravitational potential is concerned: the centrifugal potential is
part of neither V nor U here
*/

int
potsynth_grav_subtract_normal(const size_t lmin, const size_t lmax, const size_t mmax, double * coef,
                              const double GM, const double a, const potsynth_ellipsoid * ell)
{
  double C[POTSYNTH_NORMAL_LMAX + 1];
  const size_t L = (lmax < POTSYNTH_NORMAL_LMAX) ? lmax : POTSYNTH_NORMAL_LMAX;
  const double q = ell->a / a;
  double ql = 1.0;
  size_t l;
  int status;

  if (!(GM > 0.0) || !(a > 0.0) || lmin > lmax || mmax > lmax)
    {
      POTSYNTH_ERROR ("invalid model parameters", POTSYNTH_EINVAL);
    }

  status = potsynth_normal_zonals(ell, L, C);
  if (status)
    return status;

  for (l = 0; l <= L; ++l)
    {
      if (l >= lmin && C[l] != 0.0)
        coef[potsynth_idx((long) l, 0L, (long) lmin, (long) mmax)] -= C[l] * (ell->GM / GM) * ql;

      ql *= q;
    }

  return POTSYNTH_SUCCESS;
}

/*
potsynth_normal_gravity()
  Normal gravity on the surface of the ellipsoid (Somigliana's formula)

gamma = (a gamma_e cos^2(lat) + b gamma_p sin^2(lat)) / sqrt(a^2 cos^2(lat) + b^2 sin^2(lat))

Inputs: n     - number of points
        lat   - geodetic latitude (radians), length n
        ell   - reference ellipsoid
        gamma - (output) normal gravity (m/s^2), length n

Return: success or error

Notes:
1) This is the magnitude of gravity (gravitation and centrifugal) of the
normal field at h = 0; it is what Bruns' formula zeta = T/gamma needs for
height anomalies on the ellipsoid
*/

int
potsynth_normal_gravity(const size_t n, const double * lat, const potsynth_ellipsoid * ell,
                        double * gamma)
{
  const double a = ell->a, b = ell->a * (1.0 - ell->f);
  double J2, ge, gp;
  size_t i;

  if (geodesy_check(ell))
    {
      POTSYNTH_ERROR ("invalid ellipsoid", POTSYNTH_EINVAL);
    }

  geodesy_derived(ell, &J2, &ge, &gp);

  for (i = 0; i < n; ++i)
    {
      const double s2 = sin(lat[i]) * sin(lat[i]), c2 = 1.0 - s2;

      gamma[i] = (a * ge * c2 + b * gp * s2) / sqrt(a * a * c2 + b * b * s2);
    }

  return POTSYNTH_SUCCESS;
}

/* 0 if the ellipsoid parameters are valid */
static int
geodesy_check(const potsynth_ellipsoid * ell)
{
  return !(ell != NULL && ell->a > 0.0 && ell->f > 0.0 && ell->f < 0.5 && ell->GM > 0.0 &&
           ell->omega >= 0.0 && isfinite(ell->a) && isfinite(ell->GM) && isfinite(ell->omega));
}

/* derived constants J2, gamma_e, gamma_p (see the header comment) */
static void
geodesy_derived(const potsynth_ellipsoid * ell, double * J2, double * gamma_e, double * gamma_p)
{
  const double a = ell->a, f = ell->f;
  const double b = a * (1.0 - f);
  const double e2 = f * (2.0 - f);
  const double ep2 = e2 / (1.0 - e2);
  const double ep = sqrt(ep2);
  const double m = ell->omega * ell->omega * a * a * b / ell->GM;
  double q0 = 0.0, q0p = 0.0, epn = 1.0;
  size_t n;

  /* series in e'^2 (< 1 for f < 0.5): terms decrease at least geometrically */
  for (n = 1; n <= 200; ++n)
    {
      const double dn = (double) n;
      const double sgn = (n & 1) ? 1.0 : -1.0;
      const double d = (2.0 * dn + 1.0) * (2.0 * dn + 3.0);
      double t0, t1;

      epn *= ep2;
      t0 = sgn * 2.0 * dn * epn * ep / d;
      t1 = sgn * 6.0 * epn / d;
      q0 += t0;
      q0p += t1;

      if (fabs(t0) < 1.0e-18 * fabs(q0) && fabs(t1) < 1.0e-18 * fabs(q0p))
        break;
    }

  *J2 = e2 / 3.0 * (1.0 - 2.0 / 15.0 * m * ep / q0);
  *gamma_e = ell->GM / (a * b) * (1.0 - m - m / 6.0 * ep * q0p / q0);
  *gamma_p = ell->GM / (a * a) * (1.0 + m / 3.0 * ep * q0p / q0);
}
