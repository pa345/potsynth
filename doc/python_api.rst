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

Magnetic models
===============

.. py:class:: MagModel(gh, lmax=None, lmin=1, mmax=None, a=6371.2, rmin=0.0, rmax=0.0, eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0)

   An internal magnetic field model :math:`\mathbf{B} = -\nabla V`, where :math:`V` is
   given by Schmidt semi-normalized Gauss coefficients (:eq:`mag_potential`).

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
   :param float eps: requested accuracy, relative to the largest field value
   :param str method: ``"auto"``, ``"fast"`` or ``"direct"``
   :param int nthreads: OpenMP threads; 0 means all
   :param float sigma: expert override: oversampling factor; 0 means automatic
   :param int w: expert override: window width; 0 means automatic
   :param float rank_tol: expert override: radial SVD tolerance; 0 means automatic
   :param int chunk: points per internal chunk; 0 means the default

   The set-up is done in the constructor. Choose :math:`[r_{\min}, r_{\max}]` as
   tight as the data allow.

   .. py:method:: __call__(r, theta, phi)

      Evaluates the field.

      :param r: radius in km
      :param theta: colatitude in radians, :math:`[0, \pi]`
      :param phi: longitude in radians
      :returns: ``(Br, Btheta, Bphi)`` in nT, each with the broadcast shape of the
         inputs

      Inputs may be scalars or arrays of any broadcastable shapes. Points with a NaN
      coordinate give NaN outputs. :py:exc:`DomainError` is raised if a point lies
      outside the valid domain.

   .. py:attribute:: info

      A dictionary of the parameters chosen during set-up: ``method``, ``w``,
      ``sigma``, ``rho``, ``ncheb``, ``nf1``, ``nf2``, ``nrows``, ``grid_bytes``,
      ``lmin``, ``lmax``, ``mmax`` and ``nchan``.

.. py:function:: rtp_to_nec(Br, Btheta, Bphi)

   Returns :math:`(B_N, B_E, B_C) = (-B_\theta, B_\phi, -B_r)`.

Generic models
==============

.. py:class:: Channel(rpow, theta="P", dphi=0, rfac=None)

   One output quantity of a generic model, as in :eq:`channel`.

   :param int rpow: radial power offset :math:`p`; the term carries
      :math:`(a/r)^{l+p}`
   :param str theta: the operator in colatitude: ``"P"``, ``"dP"``
      (:math:`d\bar{P}/d\theta`) or ``"Psin"`` (:math:`\bar{P}/\sin\theta`)
   :param int dphi: 1 to apply :math:`\partial/\partial\phi`
   :param rfac: per-degree factors :math:`\rho(l)`, length :math:`L+1`, or ``None``
      for 1

.. py:class:: Model(coef, lmin, lmax, channels, a, rmin=0.0, rmax=0.0, mmax=None, eps=1e-10, method="auto", nthreads=0, sigma=0.0, w=0, rank_tol=0.0, chunk=0)

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
