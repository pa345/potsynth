# Changelog

All notable changes to potsynth are listed here. Versions follow
[semantic versioning](https://semver.org); before 1.0 the interface may still change
between minor versions.

## 0.1.0 (unreleased)

First public release.

### Core

- Fast synthesis of internal potential-field models at scattered points: after a one-off
  set-up, the cost per point does not depend on the maximum degree L. Accuracy can be set
  from about 1e-4 down to the double-precision rounding floor.
- Optimized direct sum (fused single-pass kernels) for low degrees and small point sets;
  `POTSYNTH_METHOD_AUTO` chooses between the two, including the set-up cost when
  `npts_hint` gives the expected number of points.
- Generic channels (linear functionals of the potential), including the theta operators
  for gradient tensors. Results are exact at both poles: no division by sin(theta).
- Read-only model workspace shared between threads; evaluation does not allocate memory.
  OpenMP parallelism (optional).

### Front-ends

- Magnetic (`potsynth_mag.h`): B and its gradient tensor grad B from Schmidt
  semi-normalized Gauss coefficients; Swarm SHC reader (e.g. LCS-1).
- Gravity (`potsynth_grav.h`): potential V, acceleration g and gravity gradient tensor
  T from fully normalized coefficients; ICGEM `.gfc` reader (EGM2008, EIGEN, XGM, ...).
- Geodetic helpers: WGS84/GRS80 ellipsoids, geodetic to geocentric coordinates, normal
  field zonal coefficients, normal gravity. The "Gravity recipes" chapter of the
  documentation shows how to compute disturbances, anomalies, height anomalies,
  deflections and geoid heights.

### Python

- `potsynth` package (ctypes and NumPy): `MagModel`, `GravModel`, the generic `Model`,
  readers, geodetic helpers and tensor rotations.
- Wheels for Linux (x86-64, aarch64), macOS (arm64, x86-64) and Windows (AMD64). The
  x86-64 wheels contain an AVX2/FMA build and a portable build of the library and select
  one at import time.

### Platforms

- Tested on Linux (GCC, x86-64 and aarch64), macOS (Apple clang, arm64) and Windows
  (MSVC), as static and shared libraries, and with AddressSanitizer, UBSan and
  ThreadSanitizer.
