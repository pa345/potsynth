# potsynth

Fast synthesis of internal potential-field models at large numbers of scattered points.
Geomagnetic models (IGRF, CHAOS, lithospheric models) and gravity field models (EGM2008,
EIGEN, XGM and others in the ICGEM `.gfc` format) are supported.

After a one-off set-up, the cost per point does not depend on the maximum degree L.
With 10^6 points, eps = 1e-10 and 16 threads on an Intel Core Ultra 7 265H, compared
with the library's own optimized direct sum:

| L    | grid    | set-up  | evaluation | direct sum | speedup |
|------|---------|---------|------------|------------|---------|
| 13   | 0.4 MB  | 0.004 s | 0.044 s    | 0.023 s    | 0.5x    |
| 60   | 7 MB    | 0.008 s | 0.053 s    | 0.21 s     | 3.8x    |
| 185  | 64 MB   | 0.073 s | 0.072 s    | 1.6 s      | 23x     |
| 400  | 288 MB  | 0.25 s  | 0.076 s    | 10.8 s     | 141x    |
| 720  | 0.9 GB  | 1.1 s   | 0.19 s     | 39 s       | 205x    |

The fast method has a one-time set-up (0.07 s at L = 185, 7.6 s for EGM2008 at L = 2190),
so it pays off from about 20,000-50,000 points; setting `npts_hint` (the expected number of
points) lets `POTSYNTH_METHOD_AUTO` choose the faster method including set-up.

The direct sum itself is 4.5-9.5x faster than computing the Legendre functions first
(`gsl_sf_alf_vsh_array`) and then summing. The library uses the direct sum for low degrees.

The requested accuracy can be set anywhere from about 1e-4 down to the double-precision
rounding floor (about 5e-15 relative to max |B|). The method is described in the
documentation (`doc/algorithm.rst`) and in
`prototype/Fast_Evaluation_Guide.pdf`.

Status: **pre-1.0**. Forward evaluation of static models is implemented and tested.

Scope: static, high-degree models (lithospheric field, gravity field). Time-dependent core
field models are of low degree (13 for IGRF, about 20 for CHAOS), where a parallel direct
sum is already fast; potsynth does not handle time-dependent coefficients.

Documentation: `doc/` (Sphinx). Build it with `make -C doc html`, then open
`doc/_build/html/index.html`.

## Design

- C library with a Python interface (`python/potsynth`, via ctypes and NumPy). There is a single
  implementation.
- No required dependencies. OpenMP is optional, and an FFT (pocketfft, BSD) is vendored in
  `extern/`.
- Double precision throughout. Legendre functions use the Holmes & Featherstone
  recursions (derived from GSL's `gsl_sf_alf_vsh_array`), with exact power-of-two column
  scaling. This supports degrees up to `POTSYNTH_LMAX` = 2800.
- The results are correct at every point on the sphere, including both poles.
  P/sin(theta) and dP/dtheta are never formed by dividing by sin(theta).
- A generic core evaluates linear functionals ("channels") of an internal potential.
  Front-ends map physical quantities onto channels:
  - `potsynth_mag`: B = -grad V from Schmidt-normalized Gauss coefficients
  - `potsynth_grav`: potential V and acceleration g = grad V from fully normalized
    coefficients, with an ICGEM `.gfc` reader. Only the synthesis is computed; normal
    field, gravity anomalies and geoid heights are left to the user.
- The model workspace is read-only after set-up and can be shared between threads.
  Evaluation does not allocate memory.
- Coefficients are passed degree by degree, with orders -M..M within each degree (see
  `potsynth_idx`). Internally all Legendre data are stored M-major, in recursion order.

## Example

```c
#include <potsynth_mag.h>

potsynth_parameters params = potsynth_default_parameters();
potsynth_workspace * w;
potsynth_eval_workspace * ew;
int status;

params.eps = 1.0e-10;

w = potsynth_mag_alloc(1, lmax, lmax, gh, 6371.2, rmin, rmax, &params);
ew = potsynth_eval_alloc(0, w);

/* B[0..n) = B_r, B[n..2n) = B_theta, B[2n..3n) = B_phi */
status = potsynth_eval(n, r, theta, phi, B, ew);

potsynth_eval_free(ew);
potsynth_free(w);
```

Errors are reported through a GSL-style error handler (`potsynth_set_error_handler()`);
the default handler prints a message and does not abort.

Python:

```python
import potsynth
model = potsynth.MagModel((g, h), rmin=rmin, rmax=rmax, eps=1e-10)
Br, Btheta, Bphi = model(r, theta, phi)
```

Models in the Swarm SHC format (e.g. LCS-1) can be read directly:
`potsynth_mag_alloc_shc("LCS-1.shc", 6371.2, rmin, rmax, &params)` in C, or
`potsynth.MagModel.from_shc("LCS-1.shc", rmin=rmin, rmax=rmax)` in Python.

Gravity models in the ICGEM format (e.g. EGM2008, here truncated at degree 720) give the
potential and acceleration, with r in metres:

```c
#include <potsynth_grav.h>

/* out[0..n) = V, out[n..2n) = g_r, then g_theta, g_phi */
w = potsynth_grav_alloc_gfc("EGM2008.gfc", 720, rmin, rmax,
                            POTSYNTH_GRAV_V | POTSYNTH_GRAV_G, &params);
```

```python
grav = potsynth.GravModel.from_gfc("EGM2008.gfc", lmax=720, quantities="V,g",
                                   rmin=rmin, rmax=rmax)
field = grav(r, theta, phi)      # field.V, field.g_r, field.g_theta, field.g_phi
```

Complete examples are in `examples/`.

## Building

    cmake -S . -B build
    cmake --build build
    cd build && ctest

Options:
- `POTSYNTH_OPENMP` (default ON)
- `POTSYNTH_NATIVE` (`-march=native`, default ON)
- `POTSYNTH_TESTS` (default ON)

- `POTSYNTH_NO_VECEXT` (portable scalar kernels; default OFF)

## Testing

`ctest` runs the test program `tests/test.c` and, for shared builds, the Python test.
`test.c` is built from the library sources and includes `test_*.c`: unit tests of the
internal components against independent references (long double, brute force, analytic
identities), the Legendre functions against GSL (if found), the magnetic field against
extended-precision reference data including both poles, fast versus direct for many
models, the API contract, and concurrent evaluation from 8 threads. Like GSL's tests it
prints only failures; set `POTSYNTH_TEST_VERBOSE=1` to see every check.

`tests/check.sh` builds and tests every configuration: release, shared, generic scalar
kernels, ASan+UBSan, and TSan.

## Layout

    include/     public headers (potsynth.h core, potsynth_mag.h, potsynth_grav.h)
    src/         implementation
    extern/      vendored third-party code (pocketfft)
    tests/       tests, long-double reference data (tests/data), benchmark
    python/      Python package (ctypes interface to the shared library)
    examples/    C and Python examples
    doc/         Sphinx documentation
    prototype/   original Python/numba prototype and write-up

## Roadmap

1. ~~Core set-up and fused evaluation; direct method; validation~~ (done)
2. ~~Swarm SHC reader for static models (e.g. LCS-1)~~ (done)
3. Python wheels (pip-installable packages; the ctypes interface works now).
4. ~~Gravity front-end and ICGEM `.gfc` reader~~ (done)
5. Adjoint (transpose) operator for inversions (separate extension).

## License

BSD 3-Clause, see `LICENSE`. pocketfft: BSD 3-Clause, see `extern/pocketfft/LICENSE.md`.
