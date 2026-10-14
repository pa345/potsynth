.. _c_api:

*****
C API
*****

There are two headers:

- ``potsynth.h``: the generic core, with channels of an internal potential.
- ``potsynth_mag.h``: the magnetic front-end. It includes ``potsynth.h``.

Typical use
===========

.. code-block:: c

   #include <potsynth_mag.h>

   potsynth_params p;
   potsynth_params_init(&p);                /* defaults */
   p.eps = 1e-10;                           /* requested accuracy */

   int status;
   potsynth_workspace *w =
     potsynth_mag_alloc(1, L, L, gh, 6371.2, rmin, rmax, &p, &status);
   if (!w) { /* potsynth_strerror(status) */ }

   potsynth_eval_workspace *ew = potsynth_eval_alloc(w, 0);
   status = potsynth_eval(w, ew, n, r, theta, phi, B);   /* B: 3*n doubles */
   /* B[0..n) = B_r, B[n..2n) = B_theta, B[2n..3n) = B_phi */

   potsynth_eval_free(ew);
   potsynth_free(w);

Objects and thread safety
=========================

The library uses two opaque objects:

.. c:type:: potsynth_workspace

   Holds everything that depends on the model: coefficients, radial basis, grids. It
   is built by :c:func:`potsynth_alloc` or :c:func:`potsynth_mag_alloc` and is
   read-only afterwards, so any number of threads can share it.

.. c:type:: potsynth_eval_workspace

   Scratch space for evaluation: point sorting and per-thread buffers. Each thread
   that calls :c:func:`potsynth_eval` at the same time as others needs its own.

Evaluation never allocates memory. All allocation happens in the ``*_alloc`` functions.
Within one call, :c:func:`potsynth_eval` uses the number of OpenMP threads set in the
parameters.

Coefficient layout
==================

Coefficients are passed as a single vector, using the same convention as mainlib's
``green2``. For degrees :math:`n_0 \le n \le L` and orders
:math:`|m| \le \min(n, M)` the order is

.. code-block:: text

   [ (n0,-M0) ... (n0,M0)  (n0+1,-M1) ... (n0+1,M1)  ...  (L,-ML) ... (L,ML) ]

Entry :math:`(n, m)` with :math:`m \ge 0` holds the cosine coefficient
(:math:`C_{nm}` or :math:`g_n^m`). Entry :math:`(n, -m)` holds the sine coefficient
(:math:`S_{nm}` or :math:`h_n^m`). The layout is read only once, during set-up.
Internally the library reorders everything to match its recursions.

.. c:macro:: POTSYNTH_IDX(n, m, lmin, mmax)

   Index of coefficient :math:`(n, m)` in the vector, with :math:`-\min(n,M) \le m \le
   \min(n,M)`. The arguments should be signed integers.

.. c:function:: size_t potsynth_ncoef(size_t lmin, size_t lmax, size_t mmax)

   Length of the coefficient vector.

.. c:macro:: POTSYNTH_LMAX

   Largest supported degree (2800).

Status codes
============

All functions that can fail return, or store in ``*status``, one of:

.. c:enumerator:: POTSYNTH_SUCCESS

   0: success.

.. c:enumerator:: POTSYNTH_EINVAL

   An argument is invalid.

.. c:enumerator:: POTSYNTH_ENOMEM

   A memory allocation failed.

.. c:enumerator:: POTSYNTH_EDOM

   An evaluation point is outside the valid domain.

.. c:enumerator:: POTSYNTH_EUNSUP

   The option is not supported, for example :math:`L >` :c:macro:`POTSYNTH_LMAX`.

.. c:function:: const char *potsynth_strerror(int status)

   A short description of a status code.

Parameters
==========

.. c:struct:: potsynth_params

   Always initialize with :c:func:`potsynth_params_init` before setting fields. A
   ``NULL`` pointer in place of the parameters means all defaults.

   .. c:member:: double eps

      Requested accuracy, relative to the largest output value. Default
      :math:`10^{-10}`. Any value from about :math:`10^{-4}` down to
      :math:`5 \times 10^{-15}` (the double-precision floor) is meaningful.

   .. c:member:: potsynth_method method

      :c:enumerator:`POTSYNTH_METHOD_AUTO` (default),
      :c:enumerator:`POTSYNTH_METHOD_FAST` or :c:enumerator:`POTSYNTH_METHOD_DIRECT`.

   .. c:member:: int nthreads

      Number of OpenMP threads; 0 (the default) means all available.

   .. c:member:: double sigma

      Oversampling factor of the grid; 0 selects the default, 2.

   .. c:member:: int w

      Window width; 0 means choose it from :c:member:`eps`.

   .. c:member:: double rank_tol

      Tolerance of the radial SVD; 0 means :math:`0.1\,\varepsilon`.

.. c:function:: void potsynth_params_init(potsynth_params *params)

   Fills ``params`` with the default values.

.. c:enum:: potsynth_method

   .. c:enumerator:: POTSYNTH_METHOD_AUTO

      Use the fast method for :math:`L \ge 24` and the direct method otherwise.

   .. c:enumerator:: POTSYNTH_METHOD_FAST

      Gridded transform (see :ref:`algorithm`). Needs the radial range
      :math:`[r_{\min}, r_{\max}]` of the points.

   .. c:enumerator:: POTSYNTH_METHOD_DIRECT

      Point-by-point sum. Accepts any :math:`r > 0`.

Magnetic front-end
==================

.. c:function:: potsynth_workspace *potsynth_mag_alloc(size_t lmin, size_t lmax, size_t mmax, const double *gh, double a, double rmin, double rmax, const potsynth_params *params, int *status)

   Builds a model of :math:`\mathbf{B} = -\nabla V` for the potential
   :eq:`mag_potential`. Its three channels are :math:`B_r`, :math:`B_\theta` and
   :math:`B_\phi`.

   :param lmin: minimum degree, :math:`\ge 1`
   :param lmax: maximum degree, :math:`\le` :c:macro:`POTSYNTH_LMAX`
   :param mmax: maximum order, :math:`\le` ``lmax``
   :param gh: Schmidt semi-normalized Gauss coefficients in nT, in the layout of
      :c:macro:`POTSYNTH_IDX`. The array is copied.
   :param a: reference radius (km), usually 6371.2
   :param rmin: smallest radius (km) at which the model will be evaluated. Required
      by the fast method; ignored by the direct method.
   :param rmax: largest radius (km) at which the model will be evaluated, as for
      ``rmin``
   :param params: parameters, or ``NULL`` for the defaults
   :param status: if not ``NULL``, receives the status code
   :returns: the workspace, or ``NULL`` on failure

   Choose :math:`[r_{\min}, r_{\max}]` as tight as the data allow. A narrower shell
   needs fewer radial functions, which makes evaluation faster.

.. c:function:: void potsynth_mag_rtp2nec(size_t npts, double *B)

   Converts output in place, from :math:`(B_r, B_\theta, B_\phi)` to
   :math:`(B_N, B_E, B_C) = (-B_\theta, B_\phi, -B_r)`.

Generic core
============

.. c:struct:: potsynth_channel

   One output quantity, as in :eq:`channel`.

   .. c:member:: int rpow

      Radial power offset :math:`p`: the term carries :math:`(a/r)^{n+p}`.
      :math:`n_0 + p \ge 0` is required.

   .. c:member:: const double *rfac

      Per-degree factors :math:`\rho(n)` for :math:`n = 0 \ldots L` (length
      :math:`L+1`), or ``NULL`` for 1. The array is copied.

   .. c:member:: potsynth_theta_op theta

      The operator in colatitude (see :c:enum:`potsynth_theta_op`).

   .. c:member:: int dphi

      0, or 1 to apply :math:`\partial/\partial\phi` to
      :math:`C\cos m\phi + S\sin m\phi`.

.. c:enum:: potsynth_theta_op

   .. c:enumerator:: POTSYNTH_THETA_P

      :math:`\bar{P}_n^m(\cos\theta)`

   .. c:enumerator:: POTSYNTH_THETA_DP

      :math:`d\bar{P}_n^m/d\theta`

   .. c:enumerator:: POTSYNTH_THETA_PSIN

      :math:`\bar{P}_n^m/\sin\theta`, computed without dividing by :math:`\sin\theta`.
      It is defined for :math:`m \ge 1`; the :math:`m = 0` terms are omitted.

.. c:function:: potsynth_workspace *potsynth_alloc(size_t lmin, size_t lmax, size_t mmax, const double *coef, double a, double rmin, double rmax, size_t nchan, const potsynth_channel *chan, const potsynth_params *params, int *status)

   Builds a model with user-defined channels.

   ``coef`` holds the 4-:math:`\pi` fully normalized coefficients in the layout of
   :c:macro:`POTSYNTH_IDX`. ``lmin`` may be 0. The remaining arguments are as for
   :c:func:`potsynth_mag_alloc`. All arrays are copied.

   For example, the magnetic front-end is equivalent to the three channels
   ``{2, rfac=n+1, THETA_P, 0}``, ``{2, rfac=-1, THETA_DP, 0}`` and
   ``{2, rfac=-1, THETA_PSIN, 1}``, with ``coef = gh / sqrt(2n+1)``.

.. c:function:: void potsynth_free(potsynth_workspace *w)

   Frees a workspace. ``NULL`` is allowed.

Evaluation
==========

.. c:function:: potsynth_eval_workspace *potsynth_eval_alloc(const potsynth_workspace *w, size_t chunk)

   Allocates scratch space for evaluating ``w``.

   Points are processed in chunks of at most ``chunk`` points; 0 selects the default,
   262144. :c:func:`potsynth_eval` accepts any number of points regardless. The result
   does not depend on the chunk size.

   Returns ``NULL`` if allocation fails.

.. c:function:: void potsynth_eval_free(potsynth_eval_workspace *ew)

   Frees an evaluation workspace. ``NULL`` is allowed.

.. c:function:: int potsynth_eval(const potsynth_workspace *w, potsynth_eval_workspace *ew, size_t npts, const double *r, const double *theta, const double *phi, double *out)

   Evaluates all channels at ``npts`` points.

   :param r: geocentric radius (the same unit as ``a``)
   :param theta: colatitude in radians, :math:`0 \le \theta \le \pi`. Both poles are
      valid.
   :param phi: longitude in radians; any finite value
   :param out: output array of ``nchan * npts`` doubles, channel-major:
      ``out[c*npts + i]``
   :returns: a status code

   Error handling:

   - **Validation first.** All points are checked before anything is written.
   - **Domain errors.** :c:enumerator:`POTSYNTH_EDOM` is returned, and the output left
     untouched, if any point has :math:`\theta \notin [0,\pi]` or :math:`r \le 0`, or,
     for the fast method, if :math:`r` lies outside :math:`[r_{\min}, r_{\max}]`.
   - **Non-finite inputs.** A NaN or infinite coordinate is not an error; that point's
     outputs are set to NaN.

   The results are bitwise identical for any number of threads and any chunk size.

Introspection
=============

.. c:struct:: potsynth_info

   The parameters chosen during set-up. Its members are ``lmin``, ``lmax``,
   ``mmax``, ``nchan``, ``method`` (the method actually selected), ``w`` and
   ``sigma``, ``rho`` (the radial rank), ``ncheb`` (Chebyshev terms per radial
   function), ``nf1``, ``nf2`` (fine grid size), ``nrows`` (stored rows), and
   ``grid_bytes``. The fast-method fields are zero for the direct method.

.. c:function:: int potsynth_get_info(const potsynth_workspace *w, potsynth_info *info)

   Fills ``info``.

Version
=======

The macros ``POTSYNTH_VERSION_MAJOR``, ``POTSYNTH_VERSION_MINOR`` and
``POTSYNTH_VERSION_PATCH`` give the version. Before 1.0 the interface may still
change.
