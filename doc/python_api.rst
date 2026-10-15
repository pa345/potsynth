.. _python_api:

**********
Python API
**********

.. py:module:: potsynth

The ``potsynth`` package wraps the C library (see :ref:`c_api`). Arrays are passed
without copying, and the Python interpreter lock is released while points are
evaluated. To build and load the library, see :ref:`installation <python-install>`.

.. code-block:: python

   import numpy as np
   import potsynth

   model = potsynth.MagModel((g, h), rmin=6371.2 + 250, rmax=6371.2 + 800, eps=1e-10)
   Br, Btheta, Bphi = model(r, theta, phi)          # nT; theta = colatitude [rad]
   BN, BE, BC = potsynth.rtp_to_nec(Br, Btheta, Bphi)

   grav = potsynth.GravModel.from_gfc("EGM2008.gfc", lmax=720, quantities="V,g",
                                      rmin=6.55e6, rmax=6.9e6)
   field = grav(r, theta, phi)                       # r in m
   field.V, field.g_r                                # m^2/s^2, m/s^2

Magnetic models
===============

.. py:class:: MagModel(gh, lmax=None, lmin=1, mmax=None, a=6371.2, rmin=0.0, rmax=0.0, eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0, npts_hint=0, quantities="B")

   An internal magnetic field model :math:`\mathbf{B} = -\nabla V`, where :math:`V` is
   given by Schmidt semi-normalized Gauss coefficients (:eq:`mag_potential`), and
   optionally its gradient tensor :math:`\nabla\mathbf{B}`.

   :param gh: Gauss coefficients in nT, in one of two forms:

      - a tuple ``(g, h)`` of 2-D arrays indexed ``[l, m]``; or
      - a 1-D vector in the layout of :py:func:`idx` (``g`` at :math:`m \ge 0`, ``h``
        at :math:`m < 0`).
   :param int lmax: maximum degree. By default it is taken from ``gh``.
   :param int lmin: minimum degree, :math:`\ge 1`
   :param int mmax: maximum order; default ``lmax``
   :param float a: reference radius in km
   :param float rmin: smallest radius (km) at which the model will be evaluated.
      Required by the fast method; ignored by the direct method.
   :param float rmax: largest radius (km) at which the model will be evaluated, as for
      ``rmin``
   :param float eps: requested accuracy, relative to the largest value of each field component
   :param str method: ``"auto"``, ``"fast"`` or ``"direct"``
   :param int npts_hint: expected number of evaluation points. With ``method="auto"``,
      the method with the smaller total time (set-up plus evaluation) is chosen; 0 means
      unknown (see :ref:`choosing-method`).
   :param int nthreads: OpenMP threads; 0 means all
   :param float sigma: expert override: oversampling factor; 0 means automatic
   :param int w: expert override: window width; 0 means automatic
   :param float rank_tol: expert override: radial SVD tolerance; 0 means automatic
   :param int chunk: points per internal chunk; 0 means the default
   :param quantities: ``"B"`` (the field, nT) and/or ``"gradB"`` (the gradient tensor
      :math:`\nabla\mathbf{B} = -\nabla\nabla V`, nT/km), as a list or a
      comma-separated string

   The set-up is done in the constructor. Choose :math:`[r_{\min}, r_{\max}]` as
   tight as the data allow.

   .. py:method:: __call__(r, theta, phi)

      Evaluates the field.

      :param r: radius in km
      :param theta: colatitude in radians, :math:`[0, \pi]`
      :param phi: longitude in radians
      :returns: a named tuple with the requested quantities, each with the broadcast
         shape of the inputs: ``B_r``, ``B_theta``, ``B_phi`` (nT), then ``dB_rr``,
         ``dB_rt``, ``dB_rp``, ``dB_tt``, ``dB_tp``, ``dB_pp`` (nT/km, spherical basis).
         With ``quantities="B"`` it unpacks as ``Br, Btheta, Bphi = model(r, theta, phi)``.

      Inputs may be scalars or arrays of any broadcastable shapes. Points with a NaN
      coordinate give NaN outputs. :py:exc:`DomainError` is raised if a point lies
      outside the valid domain.

   .. py:attribute:: quantities

      The names of the outputs, for example ``("B_r", "B_theta", "B_phi")``.

   .. py:attribute:: info

      A dictionary of the parameters chosen during set-up: ``method``, ``w``,
      ``sigma``, ``rho``, ``ncheb``, ``nf1``, ``nf2``, ``nrows``, ``grid_bytes``,
      ``lmin``, ``lmax``, ``mmax``, ``nchan`` and ``nout``.

   .. py:classmethod:: from_shc(filename, a=6371.2, rmin=0.0, rmax=0.0, **kwargs)

      Builds a model from a static Swarm SHC file (see :py:func:`read_shc`). The other
      arguments are as for :py:class:`MagModel`, for example
      ``MagModel.from_shc("LCS-1.shc", rmin=a + 250, rmax=a + 800)``.

.. py:function:: read_shc(filename)

   Reads a static model (one epoch, e.g. a lithospheric model such as LCS-1) from a
   Swarm SHC file, using the C reader (see :c:func:`potsynth_shc_read`). Malformed
   files raise :py:exc:`PotsynthError` with the line number. Time-dependent files are
   not supported.

   :returns: :py:class:`ShcModel`

.. py:class:: ShcModel

   The result of :py:func:`read_shc`, with attributes ``lmin``, ``lmax``, ``mmax``,
   ``epoch`` (decimal year), and ``gh``: the Schmidt Gauss coefficients in nT, indexed by
   :py:func:`idx`.

.. py:function:: rtp_to_nec(Br, Btheta, Bphi)

   Returns :math:`(B_N, B_E, B_C) = (-B_\theta, B_\phi, -B_r)`.

Gravity models
==============

.. py:class:: GravModel(coef, GM, a, lmax=None, lmin=0, mmax=None, quantities="V,g", rmin=0.0, rmax=0.0, eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0, npts_hint=0)

   A gravity field model: the gravitational potential :eq:`grav_potential` and the
   acceleration :math:`\mathbf{g} = \nabla V`. Only the synthesis is computed: no
   centrifugal term or normal field, and no tide-system conversion (see
   :c:func:`potsynth_grav_alloc`).

   :param coef: 4-:math:`\pi` fully normalized coefficients, in one of two forms:

      - a tuple ``(C, S)`` of 2-D arrays indexed ``[l, m]``; or
      - a 1-D vector in the layout of :py:func:`idx` (``C`` at :math:`m \ge 0`, ``S``
        at :math:`m < 0`).
   :param float GM: gravitational constant times mass in m\ :sup:`3`/s\ :sup:`2`
   :param float a: reference radius in m
   :param int lmax: maximum degree. By default it is taken from ``coef``.
   :param int lmin: minimum degree, :math:`\ge 0`
   :param int mmax: maximum order; default ``lmax``
   :param quantities: any of ``"V"`` (potential, m\ :sup:`2`/s\ :sup:`2`), ``"g_r"``
      (radial acceleration, m/s\ :sup:`2`), ``"g"`` (acceleration
      :math:`g_r, g_\theta, g_\phi`, m/s\ :sup:`2`) and ``"T"`` (gravity gradient tensor,
      s\ :sup:`-2`; :math:`1\ \mathrm{E} = 10^{-9}` s\ :sup:`-2`), as a list or a
      comma-separated string. Each quantity adds to the set-up time and memory of the
      fast method.
   :param float rmin: smallest radius (m) at which the model will be evaluated.
      Required by the fast method; ignored by the direct method.
   :param float rmax: largest radius (m), as for ``rmin``

   The other arguments are as for :py:class:`MagModel`; ``eps`` applies to each
   quantity separately.

   .. py:method:: __call__(r, theta, phi)

      Evaluates the field.

      :param r: radius in m
      :param theta: colatitude in radians, :math:`[0, \pi]`
      :param phi: longitude in radians
      :returns: a named tuple with the requested quantities, in the order ``V``,
         ``g_r``, ``g_theta``, ``g_phi``, ``T_rr``, ``T_rt``, ``T_rp``, ``T_tt``,
         ``T_tp``, ``T_pp``, each with the broadcast shape of the inputs.
         :math:`g_r` is negative (towards the centre) and :math:`g_\theta` is positive
         southwards.

   .. py:attribute:: quantities

      The names of the output quantities, for example ``("V", "g_r", "g_theta",
      "g_phi")``.

   .. py:attribute:: info

      As for :py:attr:`MagModel.info`.

   .. py:classmethod:: from_gfc(filename, lmax=0, quantities="V,g", rmin=0.0, rmax=0.0, **kwargs)

      Builds a model from an ICGEM ``.gfc`` file (see :py:func:`read_gfc`), with the
      file's :math:`GM` and reference radius, truncated at degree ``lmax`` (0 for all
      degrees). The other arguments are as for :py:class:`GravModel`.

.. py:function:: read_gfc(filename, lmax=0)

   Reads a static gravity field model from an ICGEM ``.gfc`` file, using the C reader
   (see :c:func:`potsynth_gfc_read`), truncated at degree ``lmax`` (0 for all degrees).
   Unnormalized files are converted. Malformed files raise :py:exc:`PotsynthError` with
   the line number. Time-variable models are not supported.

   :returns: :py:class:`GfcModel`

.. py:class:: GfcModel

   The result of :py:func:`read_gfc`, with attributes ``lmin`` (0), ``lmax``, ``mmax``,
   ``GM`` (m\ :sup:`3`/s\ :sup:`2`), ``a`` (m), ``modelname``, ``tide_system``, and
   ``coef``: the fully normalized coefficients, indexed by :py:func:`idx`, with absent
   coefficients set to 0.

   To compute disturbing quantities, subtract the normal field's even zonal coefficients
   from ``coef`` before building the model, for example
   ``coef[potsynth.idx(2, 0, 0, lmax)] -= C20_normal``. The normal coefficients must
   first be rescaled to the model's ``GM`` and ``a``.

Geodetic helpers
================

For the recipes of :ref:`gravity-recipes` (see also :ref:`c-geodesy`).

.. py:class:: Ellipsoid(a, f, GM, omega)

   A reference ellipsoid by its defining parameters: semi-major axis (m), flattening,
   geocentric gravitational constant (m\ :sup:`3`/s\ :sup:`2`) and angular velocity
   (rad/s). ``potsynth.WGS84`` and ``potsynth.GRS80`` are predefined.

.. py:function:: geodetic_to_geocentric(lat, h, ellipsoid=WGS84)

   Converts geodetic latitude (radians) and ellipsoidal height (m) to ``(r, theta)``:
   geocentric radius (m) and colatitude (radians).

.. py:function:: normal_zonals(lmax, ellipsoid=WGS84)

   The zonal coefficients ``C[l]`` of the normal gravitational potential, in the
   ellipsoid's own scaling.

.. py:function:: subtract_normal_field(coef, GM, a, lmin=0, lmax=None, mmax=None, ellipsoid=WGS84)

   Returns a copy of the coefficient vector with the normal field subtracted: the
   coefficients of the disturbing potential :math:`T`.

.. py:function:: normal_gravity(lat, ellipsoid=WGS84)

   Normal gravity (m/s\ :sup:`2`) on the ellipsoid at geodetic latitude ``lat``
   (radians).

Gradient tensors
================

``MagModel(..., quantities="gradB")`` and ``GravModel(..., quantities="T")`` return the
gradient tensor as six components in the spherical basis (see :ref:`gradient-tensor`).
These functions convert them; all accept arrays of any broadcastable shapes.

.. code-block:: python

   field = model(r, theta, phi)                       # MagModel, quantities="B,gradB"
   dB = field[3:]                                     # dB_rr, dB_rt, ..., dB_pp
   NN, NE, NC, EE, EC, CC = potsynth.tensor_rtp_to_nec(*dB)
   M = potsynth.tensor_matrix(*dB)                    # shape (..., 3, 3)

.. py:function:: tensor_rtp_to_nec(rr, rt, rp, tt, tp, pp)

   Returns the North-East-Center components (NN, NE, NC, EE, EC, CC), with
   :math:`\hat{e}_N = -\hat{\theta}`, :math:`\hat{e}_E = \hat{\phi}`,
   :math:`\hat{e}_C = -\hat{r}` (see :c:func:`potsynth_tensor_rtp2nec`).

.. py:function:: tensor_rtp_to_xyz(theta, phi, rr, rt, rp, tt, tp, pp)

   Returns the Earth-fixed Cartesian components (xx, xy, xz, yy, yz, zz) at colatitude
   ``theta`` and longitude ``phi`` (radians). Unlike the spherical components, these are
   unique at the poles (see :c:func:`potsynth_tensor_rtp2xyz`).

.. py:function:: tensor_matrix(t11, t12, t13, t22, t23, t33)

   Assembles the six components into symmetric matrices of shape ``(..., 3, 3)``.

Generic models
==============

.. py:class:: Channel(rpow, theta="P", dphi=0, rfac=None)

   One output quantity of a generic model, as in :eq:`channel`.

   :param int rpow: radial power offset :math:`p`; the term carries
      :math:`(a/r)^{l+p}`
   :param str theta: the operator in colatitude: ``"P"``, ``"dP"``
      (:math:`d\bar{P}/d\theta`), ``"Psin"`` (:math:`\bar{P}/\sin\theta`), or
      ``"S2A"``, ``"S2B"`` (the second-derivative operators of the gradient tensor, see
      :c:macro:`POTSYNTH_THETA_S2A`)
   :param int dphi: 1 to apply :math:`\partial/\partial\phi`
   :param rfac: per-degree factors :math:`\rho(l)`, length :math:`L+1`, or ``None``
      for 1

.. py:class:: Model(coef, lmin, lmax, channels, a, rmin=0.0, rmax=0.0, mmax=None, eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0, npts_hint=0)

   A model with user-defined channels. ``coef`` holds the 4-:math:`\pi` fully
   normalized coefficients as a vector (see :py:func:`coef_vector`), and ``lmin``
   may be 0. The other arguments are as for :py:class:`MagModel`.

   Calling the model with ``(r, theta, phi)`` returns an array of shape
   ``(len(channels),) + shape``. :py:attr:`MagModel.info` is also available.

   Example: a potential :math:`V/a` and its radial derivative:

   .. code-block:: python

      l = np.arange(L + 1)
      chans = [potsynth.Channel(rpow=1),                        # sum (a/r)^(l+1) ...
               potsynth.Channel(rpow=2, rfac=l + 1.0)]          # -dV/dr
      model = potsynth.Model(coef, 0, L, chans, a, rmin, rmax)
      V, dV = model(r, theta, phi)

Coefficients
============

.. py:function:: coef_vector(C, S, lmin=None, lmax=None, mmax=None)

   Builds a coefficient vector from 2-D arrays ``C[l, m]`` (cosine) and ``S[l, m]``
   (sine). ``S`` may be ``None``. By default ``lmin`` is 0 if ``C[0, 0]`` is nonzero and
   1 otherwise.

.. py:function:: idx(l, m, lmin, mmax)

   Index of coefficient :math:`(l, m)` in a coefficient vector. :math:`m < 0` gives the
   sine coefficient of order :math:`|m|`. This is the same as :c:macro:`potsynth_idx`.

.. py:function:: nlm(lmin, lmax, mmax=None)

   Length of a coefficient vector.

Exceptions
==========

.. py:exception:: PotsynthError

   An error reported by the C library. The attribute ``status`` holds the status code.

.. py:exception:: DomainError

   Subclass of :py:exc:`PotsynthError` and :py:exc:`ValueError`. Raised when an
   evaluation point lies outside the valid domain: :math:`\theta \notin [0,\pi]`,
   :math:`r \le 0`, or, for the fast method, :math:`r \notin [r_{\min}, r_{\max}]`.

Threads
=======

A model object holds one evaluation workspace, and calls on the same object are
serialized by a lock. Each call already uses all the threads given by ``nthreads``.
To evaluate from several Python threads at once, create one model object per thread.
