/*
 * potsynth_mag.h
 *
 * Magnetic front-end (IGRF / CHAOS / lithospheric models):
 *
 * V(r,theta,phi) = a \sum_{l=lmin}^{lmax} \sum_{m=0}^{min(l,mmax)} (a/r)^(l+1)
 *                    [ g_l^m cos(m phi) + h_l^m sin(m phi) ] P_l^m(cos theta)
 *
 * with Schmidt semi-normalized P_l^m, and B = -grad V. The front-end converts
 * the Gauss coefficients to the core normalization and defines three channels
 * (B_r, B_theta, B_phi).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_potsynth_mag_h
#define INCLUDED_potsynth_mag_h

#include "potsynth.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Prototypes
 */

/* mag.c */
POTSYNTH_API potsynth_workspace * potsynth_mag_alloc(const size_t lmin, const size_t lmax, const size_t mmax,
                                                     const double * gh, const double a,
                                                     const double rmin, const double rmax,
                                                     const potsynth_parameters * params);
POTSYNTH_API void potsynth_mag_rtp2nec(const size_t n, double * B);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_potsynth_mag_h */
