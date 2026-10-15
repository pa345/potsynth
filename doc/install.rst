************
Installation
************

Python
======

From a source checkout:

.. code-block:: sh

   pip install .

This builds the C library with CMake and installs it inside the Python package; it
needs a C compiler. Once potsynth is published on PyPI, ``pip install potsynth`` will
install a prebuilt wheel with the library included, and fall back to building from the
source distribution on platforms without one.

On x86-64 the wheels contain two builds of the library: one for CPUs with AVX2 and FMA
(every Intel and AMD processor from about 2013--2015 on), and a portable one. The
package selects the AVX2/FMA build at import time when the processor supports it. The
portable build is 1.6 times slower in the fast method and about 4 times slower in the
direct method; setting the environment variable ``POTSYNTH_GENERIC=1`` forces it.
``potsynth.library_path`` shows which library was loaded.

Requirements
============

- A C11 compiler (GCC, Clang, MSVC).
- CMake 3.15 or later.
- Optional: OpenMP, for multi-threading.
- Optional: GSL, used only by one extra test.
- For the Python interface: Python 3 and NumPy.

The FFT used during set-up (pocketfft, BSD license) is included in ``extern/``.

Building the C library
======================

.. code-block:: sh

   cmake -S . -B build
   cmake --build build
   cd build && ctest

To build a shared library, which the Python interface needs, add
``-DBUILD_SHARED_LIBS=ON``. To install, run ``cmake --install build --prefix ...``.
This installs ``potsynth.h``, ``potsynth_mag.h``, ``potsynth_grav.h`` and the library.

CMake options:

========================  =======  ==========================================================
option                    default  meaning
========================  =======  ==========================================================
``POTSYNTH_OPENMP``       ON       multi-threaded set-up and evaluation
``POTSYNTH_NATIVE``       ON       optimize for the build machine (``-march=native``)
``POTSYNTH_NO_VECEXT``    OFF      portable scalar kernels instead of GCC/Clang vector
                                   extensions (the scalar kernels are always used with
                                   other compilers)
``POTSYNTH_TESTS``        ON       build the tests
``POTSYNTH_EXAMPLES``     ON       build ``example_mag`` and ``example_grav``
``POTSYNTH_WHEEL``        OFF      Python wheel layout (set by ``pyproject.toml``)
========================  =======  ==========================================================

For binaries that will run on other machines, set ``POTSYNTH_NATIVE=OFF``. On x86, the
vector kernels then need AVX: build with ``-DCMAKE_C_FLAGS="-mavx2 -mfma"`` if the
target machines have AVX2 and FMA. Without AVX the portable scalar kernels are used,
which are slower but still several times faster than the vector kernels would be with
SSE2 only. :c:func:`potsynth_cpu_avx2_fma` reports at run time whether a CPU supports
AVX2 and FMA.

.. note::

   The library needs fused multiply-add contraction. CMake enables it with
   ``-ffp-contract=fast`` for GCC/Clang and ``/fp:contract`` for MSVC. Strict ISO modes
   such as ``-std=c11`` turn it off by default, which makes the inner loops about twice
   as slow.

Linking
=======

.. code-block:: sh

   cc -O2 myprog.c -I/prefix/include -L/prefix/lib -lpotsynth -lm

When linking a static library built with OpenMP, also add the compiler's OpenMP flag
(e.g. ``-fopenmp``).

.. _python-install:

Python interface
================

The ``potsynth`` Python package is in ``python/potsynth``. It calls the shared library
through ``ctypes``. NumPy arrays are passed without copying, and the Python interpreter
lock is released while points are evaluated. The package looks for the library in this
order:

1. the path in the environment variable ``POTSYNTH_LIBRARY``;
2. the package directory;
3. the system library path.

.. code-block:: sh

   cmake -S . -B build -DBUILD_SHARED_LIBS=ON
   cmake --build build
   export POTSYNTH_LIBRARY=$PWD/build/libpotsynth.so
   export PYTHONPATH=$PWD/python
   python -c "import potsynth; print(potsynth.__version__)"

For a development setup, the package can also be used directly from the source tree
as above. ``pip install -e .`` is not needed.

Building wheels
---------------

The wheels are built with scikit-build-core (``pyproject.toml``):

.. code-block:: sh

   python -m build                 # wheel and source distribution, for this machine
   cibuildwheel                    # release wheels: manylinux, musllinux, macOS, Windows

A wheel does not depend on the Python version (the package uses ``ctypes``), so one
build per platform covers all Python 3 versions. cibuildwheel runs the Python tests
(``tests/test_python.py`` with ``POTSYNTH_TEST_INSTALLED=1``) on every wheel it builds.

Tests
=====

``ctest`` runs two tests.

``potsynth``
   The test program ``tests/test.c``. It is built from the library sources, so it
   can test the internal components directly as well as the public interface. In
   the same style as GSL's tests, it prints only failures; set
   ``POTSYNTH_TEST_VERBOSE=1`` to see every check. It includes:

   ``test_util.c``, ``test_quad.c``, ``test_fft.c``, ``test_svd.c``, ``test_radial.c``, ``test_alf.c``
      Unit tests of the internal components, each checked against an independent
      reference (long double, brute force, or analytic identities):

      - scaled powers, :math:`\cos m\phi` and :math:`\sin m\phi`, and FFT lengths;
      - Gauss-Legendre quadrature and the window transform;
      - the FFT conventions;
      - the Jacobi SVD and the radial basis;
      - the Legendre columns at and near the poles, their scaling, and the
        accumulation kernel.

   ``test_alf_gsl.c``
      The Legendre functions against GSL. Run only if GSL is found.

   ``test_ref.c``
      The magnetic field against extended-precision reference values
      (``tests/data``, generated by ``tests/gen_reference.py``), for the direct
      method and for the fast method at several accuracies. The points include both
      poles, points within :math:`10^{-300}` of a pole, and the shell boundaries.

   ``test_eval.c``
      The fast method against the direct method, for many models and channel sets.

   ``test_api.c``
      The API contract: NaN inputs, domain errors, error codes, invalid arguments,
      and bitwise reproducibility across chunk sizes and thread counts.

   ``test_threads.c``
      One model evaluated concurrently from 8 threads.

``python``
   The Python interface. Run only for shared builds.

``tests/check.sh`` builds and tests five configurations: release, shared, generic
scalar kernels, AddressSanitizer+UndefinedBehaviorSanitizer, and ThreadSanitizer.
