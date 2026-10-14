/*
 * potsynth_mag.h -- magnetic front-end (IGRF / CHAOS / lithospheric models)
 *
 *   V(r,theta,phi) = a sum_{n=1}^{L} sum_{m=0}^{n} (a/r)^(n+1)
 *                      [g_n^m cos(m phi) + h_n^m sin(m phi)] P_n^m(cos theta)
 *
 * with Schmidt semi-normalized P_n^m, and B = -grad V. The front-end converts the
 * Gauss coefficients to the core normalization and defines three channels.
 *
 * Copyright (c) 2026, Patrick Alken and Johnathan Burchill. BSD 3-Clause License.
 *
 * Pre-1.0: the interface may still change.
 */

#ifndef POTSYNTH_MAG_H
#define POTSYNTH_MAG_H

#include "potsynth.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build a transform returning (B_r, B_theta, B_phi) as channels 0, 1, 2.
 *
 * gh    Schmidt semi-normalized Gauss coefficients in nT, in the core layout
 *       (green2 convention): gh[POTSYNTH_IDX(n,m,lmin,mmax)] = g_n^m and
 *       gh[POTSYNTH_IDX(n,-m,lmin,mmax)] = h_n^m. Requires lmin >= 1.
 *
 * Evaluate with potsynth_eval(); out[0*npts + i] = B_r, out[1*npts + i] = B_theta,
 * out[2*npts + i] = B_phi, in nT. All three components are finite and correct at
 * the poles (B_theta and B_phi there refer to the local frame defined by phi).
 */
POTSYNTH_API potsynth_workspace *potsynth_mag_alloc(size_t lmin, size_t lmax, size_t mmax,
                                                    const double *gh, double a,
                                                    double rmin, double rmax,
                                                    const potsynth_params *params,
                                                    int *status);

/* convert in place: (B_r, B_theta, B_phi) -> (B_north, B_east, B_center) */
POTSYNTH_API void potsynth_mag_rtp2nec(size_t npts, double *B);

/*
 * Planned (not in this version):
 *   - time-dependent models (piecewise-polynomial / B-spline coefficients, CHAOS core)
 *   - SHC file reader
 */

#ifdef __cplusplus
}
#endif

#endif /* POTSYNTH_MAG_H */
