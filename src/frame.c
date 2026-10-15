/*
 * frame.c
 *
 * Frame conversions of symmetric tensors (gradient tensors), stored as six
 * components per point in the order (11, 12, 13, 22, 23, 33)
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <math.h>

#include "potsynth_internal.h"

/*
potsynth_tensor_rtp2nec()
  Convert a symmetric tensor in place from spherical components
(rr, rt, rp, tt, tp, pp) to North-East-Center components
(NN, NE, NC, EE, EC, CC), with e_N = -e_theta, e_E = e_phi, e_C = -e_r:

NN = tt, NE = -tp, NC = rt, EE = pp, EC = -rp, CC = rr

Inputs: n - number of points
        T - tensor, component-major as output by potsynth_eval(), T[k*n + i],
            length 6*n
*/

void
potsynth_tensor_rtp2nec(const size_t n, double * T)
{
  size_t i;

  for (i = 0; i < n; ++i)
    {
      const double rr = T[i], rt = T[n + i], rp = T[2 * n + i];
      const double tt = T[3 * n + i], tp = T[4 * n + i], pp = T[5 * n + i];

      T[i] = tt;
      T[n + i] = -tp;
      T[2 * n + i] = rt;
      T[3 * n + i] = pp;
      T[4 * n + i] = -rp;
      T[5 * n + i] = rr;
    }
}

/*
potsynth_tensor_rtp2xyz()
  Convert a symmetric tensor in place from spherical components
(rr, rt, rp, tt, tp, pp) to Earth-fixed Cartesian components
(xx, xy, xz, yy, yz, zz):

T_xyz = R^T T_rtp R,  R = [  s cp   s sp    c ]
                          [  c cp   c sp   -s ]
                          [   -sp     cp    0 ]

with s = sin(theta), c = cos(theta), sp = sin(phi), cp = cos(phi)

Inputs: n     - number of points
        theta - colatitude (radians), length n
        phi   - longitude (radians), length n
        T     - tensor, component-major as output by potsynth_eval(), T[k*n + i],
                length 6*n

Notes:
1) Unlike the spherical components, the Cartesian components are unique at the
poles: they do not depend on the longitude given for a polar point
*/

void
potsynth_tensor_rtp2xyz(const size_t n, const double * theta, const double * phi, double * T)
{
  size_t i;

  for (i = 0; i < n; ++i)
    {
      const double s = sin(theta[i]), c = cos(theta[i]);
      const double sp = sin(phi[i]), cp = cos(phi[i]);
      const double R[3][3] = { { s * cp, s * sp, c }, { c * cp, c * sp, -s }, { -sp, cp, 0.0 } };
      const double S[3][3] = { { T[i], T[n + i], T[2 * n + i] },
                               { T[n + i], T[3 * n + i], T[4 * n + i] },
                               { T[2 * n + i], T[4 * n + i], T[5 * n + i] } };
      const size_t row[6] = { 0, 0, 0, 1, 1, 2 };
      const size_t col[6] = { 0, 1, 2, 1, 2, 2 };
      size_t k, a, b;

      for (k = 0; k < 6; ++k)
        {
          double sum = 0.0;

          /* (R^T S R)_{pq} = sum_ab R[a][p] S[a][b] R[b][q] */
          for (a = 0; a < 3; ++a)
            {
              for (b = 0; b < 3; ++b)
                sum += R[a][row[k]] * S[a][b] * R[b][col[k]];
            }

          T[k * n + i] = sum;
        }
    }
}
