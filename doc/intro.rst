************
Introduction
************

The problem
===========

The Earth's internal magnetic field is usually described by a scalar potential,

.. math::
   :label: mag_potential

   V(r,\theta,\phi) = a \sum_{l=1}^{L} \sum_{m=0}^{l} \left( \frac{a}{r} \right)^{l+1}
   \left[ g_l^m \cos m\phi + h_l^m \sin m\phi \right] P_l^m(\cos\theta),

with :math:`\mathbf{B} = -\nabla V`. Here :math:`(r,\theta,\phi)` are geocentric radius,
colatitude and longitude, and :math:`a = 6371.2` km is the reference radius. The
:math:`g_l^m, h_l^m` are the Gauss coefficients, and the :math:`P_l^m` are the Schmidt
semi-normalized associated Legendre functions. Gravity models have the same structure,
with a different normalization and prefactor:

.. math::
   :label: grav_potential

   V(r,\theta,\phi) = \frac{GM}{r} \sum_{l=0}^{L} \sum_{m=0}^{l} \left( \frac{a}{r} \right)^{l}
   \left[ \bar{C}_{lm} \cos m\phi + \bar{S}_{lm} \sin m\phi \right] \bar{P}_l^m(\cos\theta),

with gravitational acceleration :math:`\mathbf{g} = \nabla V` and 4-:math:`\pi` fully
normalized :math:`\bar{P}_l^m`.

To compare such a model with satellite data (Swarm, CHAMP, Ørsted, ...), it has to be
evaluated at every measurement position. That can mean millions or billions of points.
Each point sums about :math:`L^2` terms, so the total cost is :math:`O(ML^2)` for
:math:`M` points. This is cheap for the IGRF (:math:`L = 13`), but expensive for
lithospheric models (:math:`L = 185` and above) and for gravity models (:math:`L` up to
2190).

What potsynth does
==================

potsynth offers two methods. It chooses between them automatically, or the user can
select one:

**Direct method.**
  A point-by-point sum, optimized as far as possible:

  - The Legendre recursion and the sums over :math:`l` are done in a single pass,
    vectorized across points, with the accumulators held in vector registers.
  - No table of Legendre functions is stored.
  - It is fastest for low-degree models, such as the IGRF or the CHAOS core field.

**Fast method.**
  The model is rewritten once as a set of values on a regular, oversampled grid on the
  sphere. After that, each point costs only a small :math:`w \times w` weighted sum over
  nearby grid values, plus a few radial basis functions (see :ref:`algorithm`).

  - The cost per point does not depend on :math:`L`.
  - The only approximation is the width :math:`w` of that sum, which sets the accuracy
    from :math:`10^{-4}` to about :math:`5 \times 10^{-15}`.

Both methods are:

- **Exact at the poles.** They evaluate correctly at every point on the sphere,
  including both poles. :math:`P_l^m/\sin\theta` and :math:`dP_l^m/d\theta` are never
  computed by dividing by :math:`\sin\theta`.
- **Double precision throughout,** using the Holmes & Featherstone (2002) recursions
  with exact power-of-two scaling. They are valid up to degree :math:`L = 2800`.
- **Thread-safe:** a model built once can be shared by any number of threads, and
  evaluation never allocates memory.
- **Free of required dependencies:** written in C11, with optional OpenMP.

potsynth is aimed at **static, high-degree models**: lithospheric field models such as
LCS-1 (degree 185), and gravity models such as EGM2008 (degree 2190). Static models in
the Swarm SHC format and the ICGEM ``.gfc`` format can be read directly. Time-dependent core field models are of low degree (13 for the IGRF, about 20
for CHAOS), where a parallel direct sum is already fast, so time-dependent coefficients
are not handled.

The core of the library is generic. It evaluates any set of *channels*, meaning linear
functionals of a potential such as :math:`V`, its radial derivative, or its horizontal
gradient. Two front-ends supply the usual quantities:

- **Magnetic:** :math:`B_r, B_\theta, B_\phi` from Gauss coefficients.
- **Gravity:** the potential :math:`V` and the acceleration
  :math:`g_r, g_\theta, g_\phi` from fully normalized coefficients.

potsynth computes the spherical harmonic synthesis only. Geodetic quantities built on top
of it, such as the normal field, gravity disturbances and anomalies, and geoid heights,
are left to the user. For example, subtracting the normal field's even zonal
coefficients before set-up gives the disturbing potential and the gravity disturbance.

Performance
===========

Timings are wall-clock times for :math:`M = 10^6` random points in the 250--800 km
shell, with :math:`\varepsilon = 10^{-10}`, on 16 threads of an Intel Core Ultra 7
265H:

=====  ========  ========  ==========  ============  ========
L      grid      set-up    fast eval   direct eval   speedup
=====  ========  ========  ==========  ============  ========
13     0.4 MB    0.004 s   0.044 s     0.023 s       0.5
20     1.0 MB    0.006 s   0.046 s     0.041 s       0.9
30     2.1 MB    0.005 s   0.051 s     0.080 s       1.6
60     6.8 MB    0.008 s   0.053 s     0.21 s        3.8
133    29 MB     0.043 s   0.058 s     0.75 s        13
185    64 MB     0.073 s   0.072 s     1.6 s         23
400    288 MB    0.25 s    0.076 s     10.8 s        141
720    922 MB    1.1 s     0.19 s      39 s          205
=====  ========  ========  ==========  ============  ========

At :math:`L = 185` on a single thread:

====================  ======  ========  ==========  ============  ========
accuracy              w       set-up    fast eval   direct eval   speedup
====================  ======  ========  ==========  ============  ========
:math:`10^{-6}`       7       0.15 s    0.23 s      12.0 s        52
:math:`10^{-10}`      11      0.15 s    0.49 s      11.1 s        23
:math:`10^{-14}`      15      0.19 s    1.00 s      11.9 s        12
====================  ======  ========  ==========  ============  ========

The direct method is itself 4.5--9.5 times faster than a conventional implementation
that first computes the Legendre functions (``gsl_sf_alf_vsh_array``) and then sums the
series.
