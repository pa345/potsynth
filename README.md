# potsynth

Fast synthesis of internal potential-field models at large numbers of scattered points.
Geomagnetic models (IGRF, CHAOS, lithospheric models) are supported now; gravity models
(EGM and similar) are planned.

After a one-off set-up, the cost per point does not depend on the maximum degree L.
For example, with 10^6 points, eps = 1e-10 and 16 threads on an Intel Core Ultra 7 265H:

| L    | grid    | set-up | evaluation | direct sum | speedup |
|------|---------|--------|------------|------------|---------|
| 13   | 0.4 MB  | 3 ms   | 0.052 s    | 0.15 s     | 2.9x    |
| 60   | 7 MB    | 12 ms  | 0.062 s    | 1.1 s      | 19x     |
| 185  | 64 MB   | 57 ms  | 0.072 s    | 8.3 s      | 115x    |
| 720  | 0.9 GB  | 0.85 s | 0.13 s     | 174 s      | 1350x   |

The requested accuracy can be set anywhere from about 1e-4 down to the double-precision
rounding floor (about 5e-15 relative to max |B|). The method is described in
`prototype/Fast_Evaluation_Guide.pdf`.

Status: **pre-1.0**. Forward evaluation of static models is implemented and tested.

## Design

- C library with a Python wrapper (the wrapper is planned). There is a single implementation.
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
  - `potsynth_grav`: potential, gravity disturbance and anomaly, gradients (planned)
- The model workspace is read-only after set-up and can be shared between threads.
  Evaluation does not allocate memory.
- Coefficients use mainlib green2 layout. Internally all Legendre data are stored
  M-major, in recursion order.

## Example

```c
#include <potsynth_mag.h>

potsynth_params p;
potsynth_params_init(&p);
p.eps = 1e-10;

int status;
potsynth_workspace *w = potsynth_mag_alloc(1, L, L, gh, 6371.2, rmin, rmax, &p, &status);
potsynth_eval_workspace *ew = potsynth_eval_alloc(w, 0);

/* B[0..n) = B_r, B[n..2n) = B_theta, B[2n..3n) = B_phi */
status = potsynth_eval(w, ew, n, r, theta, phi, B);

potsynth_eval_free(ew);
potsynth_free(w);
```

## Building

    cmake -S . -B build
    cmake --build build
    cd build && ctest

Options:
- `POTSYNTH_OPENMP` (default ON)
- `POTSYNTH_NATIVE` (`-march=native`, default ON)
- `POTSYNTH_TESTS` (default ON)

If GSL is installed, an extra test compares the Legendre functions with GSL.

## Layout

    include/     public headers (potsynth.h core, potsynth_mag.h)
    src/         implementation
    extern/      vendored third-party code (pocketfft)
    tests/       tests, long-double reference data (tests/data), benchmark
    python/      Python package (planned)
    prototype/   original Python/numba prototype and write-up

## Roadmap

1. ~~Core set-up and fused evaluation; direct method; validation~~ (done)
2. Magnetic front-end: time-dependent models, SHC reader.
3. Python wrapper and wheels.
4. Gravity front-end.
5. Adjoint (transpose) operator for inversions (separate extension).

## License

BSD 3-Clause, see `LICENSE`. pocketfft: BSD 3-Clause, see `extern/pocketfft/LICENSE.md`.
