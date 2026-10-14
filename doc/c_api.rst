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

   potsynth_parameters params = potsynth_default_parameters();
   potsynth_workspace * w;
   potsynth_eval_workspace * ew;
   int status;

   params.eps = 1.0e-10;                        /* requested accuracy */

   w = potsynth_mag_alloc(1, lmax, lmax, gh, 6371.2, rmin, rmax, &params);
   ew = potsynth_eval_alloc(0, w);

   /* B[0..n) = B_r, B[n..2n) = B_theta, B[2n..3n) = B_phi */
   status = potsynth_eval(n, r, theta, phi, B, ew);

   potsynth_eval_free(ew);
   potsynth_free(w);

Objects and thread safety
=========================

The library uses two opaque objects:

.. c:type:: potsynth_workspace

   Holds everything that depends on the model: coefficients, radial basis, grids.
   It is built by :c:func:`potsynth_alloc` or :c:func:`potsynth_mag_alloc` and is
   read-only afterwards, so any number of threads can share it.

.. c:type:: potsynth_eval_workspace

   Scratch space for evaluation: point sorting and per-thread buffers. Each thread
   that calls :c:func:`potsynth_eval` at the same time as others needs its own.

Evaluation never allocates memory. All allocation happens in the ``*_alloc``
functions. Within one call, :c:func:`potsynth_eval` uses the number of OpenMP
threads given in the parameters.

Errors
======

Errors are reported in the same way as in GSL. Every function that detects an
error calls the current error handler, then returns an error code or ``NULL``.
Unlike GSL, the default handler does not abort. It prints a message such as::

   potsynth: src/eval.c:181: ERROR: theta outside [0,pi], ... (evaluation point outside the valid domain)

to ``stderr`` and lets the function return.

.. c:type:: potsynth_error_handler_t

   ``void potsynth_error_handler_t(const char * reason, const char * file, int line, int potsynth_errno)``

.. c:function:: potsynth_error_handler_t * potsynth_set_error_handler(potsynth_error_handler_t * new_handler)

   Installs a new error handler and returns the previous one.

.. c:function:: potsynth_error_handler_t * potsynth_set_error_handler_off(void)

   Disables error messages. Functions still return error codes or ``NULL``.

.. c:function:: const char * potsynth_strerror(const int potsynth_errno)

   Returns a short description of an error code.

The error codes are:

.. c:macro:: POTSYNTH_SUCCESS

   0: success.

.. c:macro:: POTSYNTH_EINVAL

   An argument is invalid.

.. c:macro:: POTSYNTH_ENOMEM

   A memory allocation failed.

.. c:macro:: POTSYNTH_EDOM

   An evaluation point is outside the valid domain.

.. c:macro:: POTSYNTH_EUNSUP

   The option is not supported, for example :math:`L >` :c:macro:`POTSYNTH_LMAX`.

Coefficient layout
==================

Coefficients are passed as a single vector, using the same convention as mainlib's
``green2``. For degrees :math:`l_{min} \le l \le l_{max}` and orders
:math:`|m| \le \min(l, m_{max})` the order is

.. code-block:: text

   [ (lmin,-M) ... (lmin,M)  (lmin+1,-M') ... (lmin+1,M')  ...  (lmax,M'') ]

Entry :math:`(l, m)` with :math:`m \ge 0` holds the cosine coefficient
(:math:`C_{lm}` or :math:`g_l^m`). Entry :math:`(l, -m)` holds the sine coefficient
(:math:`S_{lm}` or :math:`h_l^m`). The layout is read only once, during set-up.
Internally the library reorders everything to match its recursions.

.. c:macro:: potsynth_idx(l, m, lmin, mmax)

   Index of coefficient :math:`(l, m)` in the vector, with
   :math:`-\min(l,m_{max}) \le m \le \min(l,m_{max})`. Pass signed integers.

.. c:function:: size_t potsynth_nlm(const size_t lmin, const size_t lmax, const size_t mmax)

   Length of the coefficient vector.

.. c:macro:: POTSYNTH_LMAX

   Largest supported degree (2800).

Parameters
==========

.. c:struct:: potsynth_parameters

   Get the defaults from :c:func:`potsynth_default_parameters` and change fields as
   needed. A ``NULL`` pointer in place of the parameters means all defaults.

   .. c:member:: double eps

      Requested accuracy, relative to the largest output value. Default
      :math:`10^{-10}`. Any value from about :math:`10^{-4}` down to
      :math:`5 \times 10^{-15}` (the double-precision floor) is meaningful.

   .. c:member:: int method

      :c:macro:`POTSYNTH_METHOD_AUTO` (default),
      :c:macro:`POTSYNTH_METHOD_FAST` or :c:macro:`POTSYNTH_METHOD_DIRECT`.

   .. c:member:: int nthreads

      Number of OpenMP threads; 0 (the default) means all available.

   .. c:member:: double sigma

      Oversampling factor of the grid; 0 selects the default, 2.

   .. c:member:: int w

      Window width; 0 means choose it from :c:member:`eps`.

   .. c:member:: double rank_tol

      Tolerance of the radial SVD; 0 means :math:`0.1\,\varepsilon`.

.. c:function:: potsynth_parameters potsynth_default_parameters(void)

   Returns the default parameters.

.. c:macro:: POTSYNTH_METHOD_AUTO

   Use the fast method for :math:`l_{max} \ge 24` and the direct method otherwise.

.. c:macro:: POTSYNTH_METHOD_FAST

   Gridded transform (see :ref:`algorithm`). Needs the radial range
   :math:`[r_{min}, r_{max}]` of the points.

.. c:macro:: POTSYNTH_METHOD_DIRECT

   Point-by-point sum. Accepts any :math:`r > 0`.

Magnetic front-end
==================

.. c:function:: potsynth_workspace * potsynth_mag_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * gh, const double a, const double rmin, const double rmax, const potsynth_parameters * params)

   Builds a model of :math:`\mathbf{B} = -\nabla V` for the potential
   :eq:`mag_potential`. Its three channels are :math:`B_r`, :math:`B_\theta` and
   :math:`B_\phi`.

   :param lmin: minimum degree, :math:`\ge 1`
   :param lmax: maximum degree, :math:`\le` :c:macro:`POTSYNTH_LMAX`
   :param mmax: maximum order, :math:`\le` ``lmax``
   :param gh: Schmidt semi-normalized Gauss coefficients in nT, indexed by
      :c:macro:`potsynth_idx`. The array is copied.
   :param a: reference radius (km), usually 6371.2
   :param rmin: smallest radius (km) at which the model will be evaluated. Required
      by the fast method; ignored by the direct method.
   :param rmax: largest radius (km) at which the model will be evaluated, as for
      ``rmin``
   :param params: parameters, or ``NULL`` for the defaults
   :returns: the workspace, or ``NULL`` on error (reported to the error handler)

   Choose :math:`[r_{min}, r_{max}]` as tight as the data allow. A narrower shell
   needs fewer radial functions, which makes evaluation faster.

.. c:function:: void potsynth_mag_rtp2nec(const size_t n, double * B)

   Converts output in place, from :math:`(B_r, B_\theta, B_\phi)` to
   :math:`(B_N, B_E, B_C) = (-B_\theta, B_\phi, -B_r)`.

SHC files
---------

Static models in the Swarm SHC format can be read directly, for example the
lithospheric model LCS-1. The file contains ``#`` comment lines, a header line
``lmin lmax ntimes [spline_order nstep]``, a line with the epochs, and one line
``l m value`` per coefficient (:math:`m \ge 0` for :math:`g_l^m`, :math:`m < 0` for
:math:`h_l^{|m|}`). Only static models (one epoch) are supported. Time-dependent
core field models are of low degree, where a direct sum is already fast.

.. c:struct:: potsynth_shc

   .. c:member:: size_t lmin

      minimum degree

   .. c:member:: size_t lmax

      maximum degree

   .. c:member:: size_t mmax

      maximum order (equal to ``lmax``)

   .. c:member:: double epoch

      model epoch (decimal year)

   .. c:member:: double * gh

      Schmidt semi-normalized Gauss coefficients (nT), indexed by
      :c:macro:`potsynth_idx` ``(l, m, lmin, mmax)``, ready for
      :c:func:`potsynth_mag_alloc`

.. c:function:: potsynth_shc * potsynth_shc_read(const char * filename)

   Reads a static model from an SHC file. All coefficients for
   :math:`l_{min} \le l \le l_{max}`, :math:`|m| \le l` must be present exactly once.
   Malformed files are reported to the error handler with the line number. Problems
   detected include a missing header or epoch, out-of-range or duplicate entries,
   missing coefficients, extra columns, non-finite values and over-long lines.
   Time-dependent files give :c:macro:`POTSYNTH_EUNSUP`.

   :returns: the model, or ``NULL`` on error

.. c:function:: void potsynth_shc_free(potsynth_shc * shc)

   Frees a model read by :c:func:`potsynth_shc_read`.

.. c:function:: potsynth_workspace * potsynth_mag_alloc_shc(const char * filename, const double a, const double rmin, const double rmax, const potsynth_parameters * params)

   Reads an SHC file and builds the magnetic field model, as
   :c:func:`potsynth_mag_alloc`.

Generic core
============

.. c:struct:: potsynth_channel

   One output quantity, as in :eq:`channel`.

   .. c:member:: int rpow

      Radial power offset :math:`p`: the term carries :math:`(a/r)^{l+p}`.
      :math:`l_{min} + p \ge 0` is required.

   .. c:member:: const double * rfac

      Per-degree factors :math:`\rho(l)` for :math:`l = 0 \ldots l_{max}` (length
      :math:`l_{max}+1`), or ``NULL`` for 1. The array is copied.

   .. c:member:: int theta

      The operator in colatitude: one of the ``POTSYNTH_THETA_xxx`` constants below.

   .. c:member:: int dphi

      0, or 1 to apply :math:`\partial/\partial\phi` to
      :math:`C\cos m\phi + S\sin m\phi`.

.. c:macro:: POTSYNTH_THETA_P

   :math:`\bar{P}_l^m(\cos\theta)`

.. c:macro:: POTSYNTH_THETA_DP

   :math:`d\bar{P}_l^m/d\theta`

.. c:macro:: POTSYNTH_THETA_PSIN

   :math:`\bar{P}_l^m/\sin\theta`, computed without dividing by :math:`\sin\theta`.
   It is defined for :math:`m \ge 1`; the :math:`m = 0` terms are omitted.

.. c:function:: potsynth_workspace * potsynth_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * coef, const double a, const double rmin, const double rmax, const size_t nchan, const potsynth_channel * chan, const potsynth_parameters * params)

   Builds a model with user-defined channels.

   ``coef`` holds the 4-:math:`\pi` fully normalized coefficients, indexed by
   :c:macro:`potsynth_idx`. ``lmin`` may be 0. The remaining arguments are as for
   :c:func:`potsynth_mag_alloc`. All arrays are copied.

   For example, the magnetic front-end is equivalent to the three channels
   ``{2, rfac=l+1, THETA_P, 0}``, ``{2, rfac=-1, THETA_DP, 0}`` and
   ``{2, rfac=-1, THETA_PSIN, 1}``, with ``coef = gh / sqrt(2l+1)``.

.. c:function:: void potsynth_free(potsynth_workspace * w)

   Frees a workspace.

Evaluation
==========

.. c:function:: potsynth_eval_workspace * potsynth_eval_alloc(const size_t chunk, const potsynth_workspace * w)

   Allocates scratch space for evaluating ``w``.

   Points are processed in chunks of at most ``chunk`` points; 0 selects the default,
   262144. :c:func:`potsynth_eval` accepts any number of points regardless. The result
   does not depend on the chunk size.

.. c:function:: void potsynth_eval_free(potsynth_eval_workspace * ew)

   Frees an evaluation workspace.

.. c:function:: int potsynth_eval(const size_t n, const double * r, const double * theta, const double * phi, double * out, potsynth_eval_workspace * ew)

   Evaluates all channels of the model at ``n`` points.

   :param n: number of points
   :param r: geocentric radius (the same unit as ``a``), length ``n``
   :param theta: colatitude in radians, :math:`0 \le \theta \le \pi`. Both poles are
      valid.
   :param phi: longitude in radians; any finite value
   :param out: output array of ``nchan * n`` doubles, channel-major:
      ``out[c*n + i]``
   :param ew: evaluation workspace
   :returns: a status code

   Error handling:

   - **Validation first.** All points are checked before anything is written.
   - **Domain errors.** :c:macro:`POTSYNTH_EDOM` is returned, and the output left
     untouched, if any point has :math:`\theta \notin [0,\pi]` or :math:`r \le 0`,
     or, for the fast method, if :math:`r` lies outside :math:`[r_{min}, r_{max}]`.
   - **Non-finite inputs.** A NaN or infinite coordinate is not an error; that
     point's outputs are set to NaN.

   The results are bitwise identical for any number of threads and any chunk size.

Introspection
=============

.. c:struct:: potsynth_info

   The parameters chosen during set-up. Its members are:

   - ``lmin``, ``lmax``, ``mmax``, ``nchan``;
   - ``method``, the method actually selected;
   - ``w`` and ``sigma``;
   - ``rho``, the radial rank;
   - ``ncheb``, the number of Chebyshev terms per radial function;
   - ``nf1``, ``nf2``, the fine grid size;
   - ``nrows``, the number of stored rows;
   - ``grid_bytes``.

   The fast-method fields are zero for the direct method.

.. c:function:: int potsynth_get_info(potsynth_info * info, const potsynth_workspace * w)

   Fills ``info``.

Version
=======

The macros ``POTSYNTH_VERSION_MAJOR``, ``POTSYNTH_VERSION_MINOR`` and
``POTSYNTH_VERSION_PATCH`` give the version. Before 1.0 the interface may still
change.
