/*
 * potsynth_mag.h
 *
 * Magnetic front-end (IGRF / CHAOS / lithospheric models):
 *
 * V(r,theta,phi) = a \sum_{l=lmin}^{lmax} \sum_{m=0}^{min(l,mmax)} (a/r)^(l+1)
 *                    [ g_l^m cos(m phi) + h_l^m sin(m phi) ] P_l^m(cos theta)
 *
 * with Schmidt semi-normalized P_l^m, and B = -grad V. The front-end converts
 * the Gauss coefficients to the core normalization and defines the channels
 * of the requested quantities: the field (B_r, B_theta, B_phi) and/or its
 * gradient tensor grad B = -grad grad V.
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_potsynth_mag_h
#define INCLUDED_potsynth_mag_h

#include "potsynth.h"

#ifdef __cplusplus
extern "C" {
#endif

/* quantities computed (bit flags); outputs are ordered B_r, B_theta, B_phi, then
 * the gradient tensor (rr, rt, rp, tt, tp, pp) */
#define POTSYNTH_MAG_B               0x1 /* field (B_r, B_theta, B_phi), nT */
#define POTSYNTH_MAG_GRADB           0x2 /* gradient tensor grad B = -grad grad V, nT/km:
                                            dB_rr, dB_rt, dB_rp, dB_tt, dB_tp, dB_pp */

/* static model read from a Swarm SHC file */
typedef struct
{
  size_t lmin;          /* minimum SH degree */
  size_t lmax;          /* maximum SH degree */
  size_t mmax;          /* maximum SH order (= lmax) */
  double epoch;         /* model epoch (decimal year) */
  double * gh;          /* Schmidt Gauss coefficients (nT), indexed by potsynth_idx(l,m,lmin,mmax) */
} potsynth_shc;

/*
 * Prototypes
 */

/* mag.c */
POTSYNTH_API potsynth_workspace * potsynth_mag_alloc(const size_t lmin, const size_t lmax, const size_t mmax,
                                                     const double * gh, const double a,
                                                     const double rmin, const double rmax,
                                                     const int quantities, const potsynth_parameters * params);
POTSYNTH_API size_t potsynth_mag_nout(const int quantities);
POTSYNTH_API void potsynth_mag_rtp2nec(const size_t n, double * B);

/* shc.c */
POTSYNTH_API potsynth_shc * potsynth_shc_read(const char * filename);
POTSYNTH_API void potsynth_shc_free(potsynth_shc * shc);
POTSYNTH_API potsynth_workspace * potsynth_mag_alloc_shc(const char * filename, const double a,
                                                         const double rmin, const double rmax,
                                                         const int quantities,
                                                         const potsynth_parameters * params);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_potsynth_mag_h */
