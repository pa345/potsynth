/*
 * potsynth_grav.h
 *
 * Gravity front-end (EGM2008, EIGEN, XGM, GOCO, ... models in ICGEM format):
 *
 * V(r,theta,phi) = GM/r \sum_{l=lmin}^{lmax} \sum_{m=0}^{min(l,mmax)} (a/r)^l
 *                    [ C_lm cos(m phi) + S_lm sin(m phi) ] Pbar_l^m(cos theta)
 *
 * with 4-pi fully normalized Pbar_l^m, gravitational acceleration g = grad V
 * and gravity gradient tensor T = grad grad V. This is the gravitational field
 * of the model coefficients only: no centrifugal term and no normal field are
 * included (subtract the normal field zonals from the coefficients for
 * disturbing quantities).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#ifndef INCLUDED_potsynth_grav_h
#define INCLUDED_potsynth_grav_h

#include "potsynth.h"

#ifdef __cplusplus
extern "C" {
#endif

/* quantities computed (bit flags); outputs are ordered V, g_r, g_theta, g_phi, then the
 * gradient tensor (rr, rt, rp, tt, tp, pp) */
#define POTSYNTH_GRAV_V              0x1 /* potential V (m^2/s^2) */
#define POTSYNTH_GRAV_GR             0x2 /* radial acceleration g_r = dV/dr (m/s^2) */
#define POTSYNTH_GRAV_G              0x4 /* acceleration (g_r, g_theta, g_phi) = grad V (m/s^2); includes GR */
#define POTSYNTH_GRAV_T              0x8 /* gravity gradient tensor T = grad grad V (1/s^2; 1 E = 1e-9/s^2):
                                            T_rr, T_rt, T_rp, T_tt, T_tp, T_pp */

/* static model read from an ICGEM .gfc file */
typedef struct
{
  size_t lmin;          /* minimum SH degree (0) */
  size_t lmax;          /* maximum SH degree */
  size_t mmax;          /* maximum SH order (= lmax) */
  double GM;            /* earth_gravity_constant (m^3/s^2) */
  double a;             /* reference radius (m) */
  char modelname[128];  /* model name from the header */
  char tide_system[64]; /* tide system from the header, or "unknown" */
  double * coef;        /* 4-pi fully normalized C_lm (m >= 0), S_lm (-m), indexed by
                           potsynth_idx(l,m,lmin,mmax); absent coefficients are 0 */
} potsynth_gfc;

/*
 * Prototypes
 */

/* grav.c */
POTSYNTH_API potsynth_workspace * potsynth_grav_alloc(const size_t lmin, const size_t lmax, const size_t mmax,
                                                      const double * coef, const double GM, const double a,
                                                      const double rmin, const double rmax,
                                                      const int quantities, const potsynth_parameters * params);
POTSYNTH_API size_t potsynth_grav_nout(const int quantities);

/* gfc.c */
POTSYNTH_API potsynth_gfc * potsynth_gfc_read(const char * filename, const size_t lmax);
POTSYNTH_API void potsynth_gfc_free(potsynth_gfc * gfc);
POTSYNTH_API potsynth_workspace * potsynth_grav_alloc_gfc(const char * filename, const size_t lmax,
                                                          const double rmin, const double rmax,
                                                          const int quantities,
                                                          const potsynth_parameters * params);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDED_potsynth_grav_h */
