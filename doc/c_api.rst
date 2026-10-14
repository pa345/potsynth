.. _c_api:

*****
C API
*****

There are three headers:

- ``potsynth.h``: the generic core, with channels of an internal potential.
- ``potsynth_mag.h``: the magnetic front-end. It includes ``potsynth.h``.
- ``potsynth_grav.h``: the gravity front-end. It includes ``potsynth.h``.

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

Coefficients are passed as a single vector. For degrees :math:`l_{min} \le l \le l_{max}` and orders
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

      Requested accuracy, relative to the largest value of each output channel. Default
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

   .. c:member:: size_t npts_hint

      Expected number of points that will be evaluated with this model; 0 (the
      default) means unknown. :c:macro:`POTSYNTH_METHOD_AUTO` uses it to choose the
      method with the smaller total time, set-up plus evaluation. The fast method's
      set-up only pays off for enough points; see :ref:`choosing-method`.

.. c:function:: potsynth_parameters potsynth_default_parameters(void)

   Returns the default parameters.

.. c:macro:: POTSYNTH_METHOD_AUTO

   With :c:member:`potsynth_parameters.npts_hint` set: use the method with the smaller estimated total
   time for that many points (see :ref:`choosing-method`). Without it: use the fast
   method for :math:`l_{max} \ge 24` and the direct method otherwise.

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

Gravity front-end
=================

The gravity front-end evaluates the gravitational potential :eq:`grav_potential` and the
acceleration :math:`\mathbf{g} = \nabla V` of a model given by 4-:math:`\pi` fully
normalized coefficients :math:`\bar{C}_{lm}, \bar{S}_{lm}`. It computes the synthesis
only:

- no centrifugal term and no normal field are included;
- coordinates are geocentric spherical, with :math:`r` in metres;
- coefficients are used as given (no tide-system conversion).

For disturbing quantities, subtract the normal field's even zonal coefficients before
set-up.

.. c:macro:: POTSYNTH_GRAV_V

   The potential :math:`V` (m\ :sup:`2`/s\ :sup:`2`).

.. c:macro:: POTSYNTH_GRAV_GR

   The radial acceleration :math:`g_r = \partial V/\partial r` (m/s\ :sup:`2`), which is
   negative (towards the centre).

.. c:macro:: POTSYNTH_GRAV_G

   The acceleration :math:`(g_r, g_\theta, g_\phi) = \nabla V` (m/s\ :sup:`2`), with
   :math:`g_\theta` positive southwards. It includes :c:macro:`POTSYNTH_GRAV_GR`.

The flags are combined with ``|``. The output channels are in the order :math:`V, g_r,
g_\theta, g_\phi`, restricted to those requested. Each channel adds to the set-up time
and grid memory of the fast method, so request only what is needed: for example
:c:macro:`POTSYNTH_GRAV_GR` alone for radial gravity.

.. c:function:: potsynth_workspace * potsynth_grav_alloc(const size_t lmin, const size_t lmax, const size_t mmax, const double * coef, const double GM, const double a, const double rmin, const double rmax, const int quantities, const potsynth_parameters * params)

   Builds a gravity field model.

   :param lmin: minimum degree, :math:`\ge 0`
   :param lmax: maximum degree, :math:`\le` :c:macro:`POTSYNTH_LMAX`
   :param mmax: maximum order, :math:`\le` ``lmax``
   :param coef: fully normalized coefficients, indexed by :c:macro:`potsynth_idx`
      (:math:`\bar{C}_{lm}` at :math:`m \ge 0`, :math:`\bar{S}_{lm}` at :math:`-m`). The
      array is copied.
   :param GM: gravitational constant times mass (m\ :sup:`3`/s\ :sup:`2`)
   :param a: reference radius (m)
   :param rmin: smallest radius (m) at which the model will be evaluated. Required by
      the fast method; ignored by the direct method.
   :param rmax: largest radius (m) at which the model will be evaluated, as for ``rmin``
   :param quantities: ``POTSYNTH_GRAV_xxx`` flags
   :param params: parameters, or ``NULL`` for the defaults
   :returns: the workspace, or ``NULL`` on error (reported to the error handler)

   The accuracy :c:member:`potsynth_parameters.eps` applies to each output channel
   separately. For a full model, each channel is dominated by its central
   (:math:`l = 0`) term; for :math:`10^{-10}` the potential is accurate to about
   :math:`10^{-3}` m\ :sup:`2`/s\ :sup:`2` (0.1 mm of geoid height) and
   :math:`g_r` to about :math:`2 \times 10^{-5}` mGal, measured with EGM2008 at satellite
   altitudes.

.. c:function:: size_t potsynth_grav_nchan(const int quantities)

   The number of output channels for a set of flags, or 0 if the flags are invalid.

ICGEM files
-----------

Static models in the ICGEM format (``.gfc`` files, as distributed by the International
Centre for Global Earth Models, http://icgem.gfz-potsdam.de) can be read directly. The
file has a header ending with a line ``end_of_head``, and one line
``gfc l m C S [sigma_C sigma_S]`` per coefficient. The reader uses these header keywords:

- ``earth_gravity_constant``, ``radius`` and ``max_degree``, which are required;
- ``norm``: ``fully_normalized`` (the default) or ``unnormalized``, which is converted;
- ``modelname``, ``tide_system`` and ``product_type`` (which must be ``gravity_field``
  if present).

Other header lines are ignored. Numbers may use Fortran exponents (``1.0D-09``).
Coefficients absent from the file are zero: many models omit degrees 0 and 1. Files with
time-variable terms (``gfct``, ``trnd``, ``dot``, ``acos``, ``asin``) give
:c:macro:`POTSYNTH_EUNSUP`.

.. c:struct:: potsynth_gfc

   .. c:member:: size_t lmin

      minimum degree (0)

   .. c:member:: size_t lmax

      maximum degree (``max_degree``, or the truncation degree)

   .. c:member:: size_t mmax

      maximum order (equal to ``lmax``)

   .. c:member:: double GM

      ``earth_gravity_constant`` (m\ :sup:`3`/s\ :sup:`2`)

   .. c:member:: double a

      ``radius`` (m)

   .. c:member:: char modelname[128]

      ``modelname`` from the header

   .. c:member:: char tide_system[64]

      ``tide_system`` from the header, or ``unknown``

   .. c:member:: double * coef

      fully normalized coefficients, indexed by :c:macro:`potsynth_idx`
      ``(l, m, 0, lmax)``, ready for :c:func:`potsynth_grav_alloc`

.. c:function:: potsynth_gfc * potsynth_gfc_read(const char * filename, const size_t lmax)

   Reads a static model from a ``.gfc`` file, truncated at degree ``lmax`` (0 for all
   degrees of the file). Malformed files are reported to the error handler with the line
   number. Problems detected include a missing ``end_of_head`` or required keyword,
   unknown keywords, invalid numbers, :math:`m > l`, :math:`l >` ``max_degree``,
   duplicate entries, a nonzero :math:`S_{l0}` and over-long lines.

   :returns: the model, or ``NULL`` on error

.. c:function:: void potsynth_gfc_free(potsynth_gfc * gfc)

   Frees a model read by :c:func:`potsynth_gfc_read`.

.. c:function:: potsynth_workspace * potsynth_grav_alloc_gfc(const char * filename, const size_t lmax, const double rmin, const double rmax, const int quantities, const potsynth_parameters * params)

   Reads a ``.gfc`` file, truncated at degree ``lmax`` (0 for all degrees), and builds
   the gravity field model with the file's :math:`GM` and reference radius, as
   :c:func:`potsynth_grav_alloc`. For example, radial gravity from EGM2008 to degree 720
   at satellite altitudes:

   .. code-block:: c

      #include <potsynth_grav.h>

      w = potsynth_grav_alloc_gfc("EGM2008.gfc", 720, 6.55e6, 6.9e6, POTSYNTH_GRAV_GR, NULL);
      ew = potsynth_eval_alloc(0, w);
      status = potsynth_eval(n, r, theta, phi, g_r, ew);   /* r in m, g_r in m/s^2 */

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
