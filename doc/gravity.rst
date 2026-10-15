.. _gravity-recipes:

***************
Gravity recipes
***************

.. py:currentmodule:: potsynth

potsynth computes the gravitational potential :math:`V`, the acceleration
:math:`\mathbf{g} = \nabla V` and the gradient tensor :math:`\nabla\nabla V` of a set of
coefficients. The usual geodetic quantities follow from the *disturbing potential*
:math:`T = V - U`, where :math:`U` is the normal potential of a reference ellipsoid,
plus a few formulas. This chapter shows how, with the helper functions of the gravity
front-end.

The recipes use the spherical approximation that is conventional for global models:
derivatives are taken along the geocentric radius, and the difference between geodetic
and geocentric latitude in the formulas is neglected. Ellipsoidal corrections,
topographic effects and tide-system conversions are not included.

The examples are in Python; the C functions have the same names with the ``potsynth_``
prefix (see :ref:`c-geodesy`).

Coordinates
===========

Gravity models are evaluated in geocentric spherical coordinates
:math:`(r, \theta, \phi)`. Points given by geodetic latitude :math:`\varphi`,
longitude :math:`\lambda` and ellipsoidal height :math:`h` are converted with

.. code-block:: python

   r, theta = potsynth.geodetic_to_geocentric(lat, h)     # WGS84 by default
   phi = lon                                             # longitude is unchanged

``potsynth.WGS84`` and ``potsynth.GRS80`` are predefined; any ellipsoid can be given as
``potsynth.Ellipsoid(a, f, GM, omega)``.

Disturbing potential
====================

The normal gravitational potential of an ellipsoid has only even zonal terms,

.. math::

   U = \frac{GM_e}{r} \sum_{n=0}^{10} \left( \frac{a_e}{r} \right)^{2n}
       \bar{C}^e_{2n,0}\, \bar{P}_{2n}^0(\cos\theta),

with :math:`\bar{C}^e_{2n,0} = -J_{2n}/\sqrt{4n+1}` computed from the ellipsoid's
defining parameters (Moritz, 2000). To get :math:`T`, these terms are rescaled to the
model's :math:`GM` and :math:`a` and subtracted from its coefficients:

.. code-block:: python

   gfc = potsynth.read_gfc("EGM2008.gfc")
   coefT = potsynth.subtract_normal_field(gfc.coef, gfc.GM, gfc.a, ellipsoid=potsynth.WGS84)
   modelT = potsynth.GravModel(coefT, gfc.GM, gfc.a, quantities="V,g", rmin=rmin, rmax=rmax)
   T, gr, gt, gp = modelT(r, theta, phi)                 # T and grad T

The subtraction includes degree 0, so :math:`T` contains the zero-degree term
:math:`(GM - GM_e)/r`. Some conventions omit it and add a constant instead (see the
geoid example below).

The coefficients are used as given: the tide system of the model (reported by
:py:func:`read_gfc`) and that of the reference ellipsoid are not converted.

Gravity disturbance and anomaly
===============================

.. math::

   \delta g = -\frac{\partial T}{\partial r}, \qquad
   \Delta g = -\frac{\partial T}{\partial r} - \frac{2}{r}\, T,

both from the outputs of the model of :math:`T` (1 mGal = :math:`10^{-5}` m/s\ :sup:`2`):

.. code-block:: python

   dg = -gr * 1e5                       # gravity disturbance (mGal)
   Dg = (-gr - 2.0 * T / r) * 1e5       # gravity anomaly (mGal)

Height anomaly
==============

Bruns' formula gives the height anomaly from :math:`T` and the normal gravity
:math:`\gamma` on the ellipsoid (Somigliana's formula):

.. math::

   \zeta = \frac{T}{\gamma}

.. code-block:: python

   zeta = T / potsynth.normal_gravity(lat)              # m, points on the ellipsoid

``normal_gravity`` is the value on the ellipsoid (:math:`h = 0`). For points above it,
normal gravity decreases by about 0.3086 mGal per metre of height.

Deflections of the vertical
===========================

The north-south and east-west components, in radians:

.. math::

   \xi = -\frac{1}{r\gamma} \frac{\partial T}{\partial \varphi} = \frac{g_\theta}{\gamma},
   \qquad
   \eta = -\frac{1}{r\gamma\cos\varphi} \frac{\partial T}{\partial \lambda}
        = -\frac{g_\phi}{\gamma},

with :math:`g_\theta = r^{-1} \partial T/\partial\theta` and
:math:`g_\phi = (r \sin\theta)^{-1} \partial T/\partial\phi` from the model of
:math:`T`:

.. code-block:: python

   gamma = potsynth.normal_gravity(lat)
   xi, eta = gt / gamma, -gp / gamma

Geoid height
============

The geoid height :math:`N` differs from the height anomaly :math:`\zeta` by a term that
depends on the topography, up to a few metres in mountains. It is model-specific, and
outside the scope of potsynth. EGM2008 provides it as a separate set of spherical
harmonic coefficients (``Zeta-to-N``), a plain sum without radial factor. That sum is a
model with :math:`GM = a = 1`, evaluated at :math:`r = 1`.

The complete EGM2008 geoid, following the conventions of NGA's program
``hsynth_WGS84.f``:

.. code-block:: python

   import numpy as np
   import potsynth

   lat, lon = np.radians(lat_deg), np.radians(lon_deg)

   # disturbing potential: NGA omits degree 0 of T and adds a constant N0 instead
   egm = potsynth.read_gfc("EGM2008.gfc")
   coef = potsynth.subtract_normal_field(egm.coef, egm.GM, egm.a, ellipsoid=potsynth.WGS84)
   coef[potsynth.idx(0, 0, 0, egm.lmax)] = 0.0
   N0 = -0.41

   # points on the ellipsoid
   r, theta = potsynth.geodetic_to_geocentric(lat, 0.0)

   # height anomaly, then geoid height with EGM2008's correction model
   T = potsynth.GravModel(coef, egm.GM, egm.a, quantities="V")(r, theta, lon).V
   zeta = T / potsynth.normal_gravity(lat)
   corr = potsynth.GravModel.from_gfc("Zeta-to-N.gfc", quantities="V")
   N = zeta + N0 + corr(1.0, theta, lon).V

Here ``Zeta-to-N.gfc`` holds NGA's correction coefficients with a header
``earth_gravity_constant 1``, ``radius 1``, ``max_degree 2160``. At 300 points, including
both poles, this reproduces the geoid heights of NGA's program to within its 1 mm output
rounding (largest difference 0.499 mm). The example gives no radial range, so the direct
method is used; for many points, give ``rmin`` and ``rmax`` (and ``npts_hint``) so that
the fast method can be used.

Gravity gradients
=================

The gravity gradient tensor is :math:`\nabla\nabla V`, usually given in Eötvös units
(1 E = :math:`10^{-9}` s\ :sup:`-2`). Satellite gradiometry works with the full
potential, so no normal field needs to be subtracted:

.. code-block:: python

   m = potsynth.GravModel.from_gfc("model.gfc", lmax=300, quantities="T",
                                   rmin=6.55e6, rmax=6.65e6)
   Trr, Trt, Trp, Ttt, Ttp, Tpp = (c * 1e9 for c in m(r, theta, phi))   # E

The components are in the spherical basis :math:`(\hat{r}, \hat{\theta}, \hat{\phi})`
(see :ref:`gradient-tensor`). Other frames follow by sign changes and permutations. For
example, a local north-west-up frame
(:math:`\hat{x} = -\hat{\theta}`, :math:`\hat{y} = -\hat{\phi}`, :math:`\hat{z} = \hat{r}`)
has

.. math::

   T_{xx} = T_{\theta\theta}, \quad T_{yy} = T_{\phi\phi}, \quad T_{zz} = T_{rr}, \quad
   T_{xy} = T_{\theta\phi}, \quad T_{xz} = -T_{r\theta}, \quad T_{yz} = -T_{r\phi}.

:py:func:`tensor_rtp_to_nec` and :py:func:`tensor_rtp_to_xyz` convert to
North-East-Center and Earth-fixed Cartesian components.

What is not included
====================

These depend on conventions or data beyond the coefficients, and are left to the user:

- tide-system conversions (a change of :math:`\bar{C}_{20}`);
- the centrifugal potential and acceleration (needed for gravity rather than
  gravitation);
- normal gravity above the ellipsoid, and ellipsoidal corrections to the formulas
  above;
- topographic and atmospheric corrections, and the geoid-to-height-anomaly separation
  beyond what a model such as EGM2008 provides.

Reference
=========

H. Moritz (2000). Geodetic Reference System 1980. *Journal of Geodesy* 74, 128--133.
