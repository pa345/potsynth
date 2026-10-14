# fastpot: fast evaluation of internal field models (IGRF / CHAOS type)

## Main implementation: `fastfield.py`

A 2-D transform with real-valued, half-sphere grids. The set-up builds the grid once: radial SVD, exact Legendre-to-Fourier
map, pre-sharpening and FFT. Evaluation is a window sum over that grid (the interpolation stage of a type-2 NUFFT), done
by our own compiled fused loop: the window weights are computed once per point, and all 3*rho channels are summed in
one pass together with the radial combination.

```python
from fastfield import FastField
F = FastField(g, h, rmin=6371.2 + 250, rmax=6371.2 + 800, eps=1e-10)   # set-up, ~1 s at L = 185
Br, Btheta, Bphi = F(r_km, theta_rad, phi_rad)                        # theta = colatitude
```

**Results** at 10^6 points, L = 185, 2 cores (`bench_backends.py`, `bench_backends.json`):

| Requested accuracy | Default (own fused loop): speedup vs direct, error | `backend="finufft"`: speedup vs direct, error |
|---|---|---|
| 1e-6 | 200x, 6.3e-7 | 134x, 1.1e-6 |
| 1e-10 | 95x, 1.6e-11 | 65x, 8.6e-11 |
| 1e-13 | 61x, 3.3e-14 | 43x, 5.9e-14 |

* **Why not a general NUFFT library?** FINUFFT's interpolation-only mode also works (`backend="finufft"`), with
  pre-sharpening factors measured through the library. But a general library treats each of the ceil(3*rho/2)
  channel pairs as a separate transform. It recomputes the per-point window weights for each one, and returns
  per-channel results that must then be re-read for the radial sum. That makes it about 1.4x slower here.
* `Fast_Evaluation_Guide.pdf` (source in `doc_fastfield/`) explains every step at undergraduate level.
* Requirements: numpy, scipy, numba. finufft is needed only for `backend="finufft"`.

The rest of this README describes the earlier implementations, kept for reference.

## Earlier implementations

Evaluates B = −∇V for

    V(r,θ,φ) = a Σ_{n=1}^{L} Σ_{m=0}^{n} (a/r)^{n+1} [g_n^m cos mφ + h_n^m sin mφ] P_n^m(cos θ)

at M scattered points (r_i, θ_i, φ_i). The cost is O(K L³) to precompute plus O(K L² log L + M·w³) to evaluate,
where w ≈ log10(1/ε)+1. Direct point-by-point synthesis costs O(M L²).

## The idea

Put u = a/r. Each field component then has the form

    B_c = Re Σ_{n,m} u^{n+2} · w_c(n,m) · Θ_c^{nm}(θ) · e^{imφ}

with Θ_r = (n+1)P_n^m, Θ_θ = −dP_n^m/dθ, Θ_φ = −i m P_n^m / sin θ, and w = g_n^m − i h_n^m.

1. **Radial direction.** u^{n+2} is a *polynomial* in u of degree at most L+2. Substitute
   u = u_c + u_d cos ψ, which maps the radial range [r_min, r_max] of the data to ψ ∈ [0, π].
   The radial factor becomes a cosine (Chebyshev) series of degree at most L+2 in ψ.
   This is **exact** for K = L+2; for a thin shell (e.g. satellite altitudes) K ≈ 4–12 is enough, chosen automatically from ε.
2. **Colatitude.** On the doubled sphere θ ∈ [0, 2π), P_n^m(cos θ), dP_n^m/dθ and P_n^m/sin θ (m ≥ 1) are
   trigonometric polynomials of degree at most n in θ. They are pole-safe, because P_n^m carries a factor sin^m θ.
3. **Longitude.** e^{imφ} is already a Fourier mode.

So every component is **exactly** a band-limited 3-D Fourier series in (ψ, θ, φ):

    B_c(r,θ,φ) = Re Σ_{|k|≤K} Σ_{|p|≤L} Σ_{m=0}^{L} F_c[k,p,m] e^{i(kψ + pθ + mφ)}

Evaluating this at all M points is **one type-2 3-D NUFFT** (FINUFFT), batched over the 3 components.
The coefficient tensor F is built once from (g, h). Legendre values on a 2L+2-point θ grid (shifted off the poles)
are FFT'd in θ and contracted with the Chebyshev coefficients of u^{n+2}, which costs O(K L³) using BLAS.

There are no quadrature grids, no interpolation in r and no approximation in θ or φ. The only error is the NUFFT
tolerance ε, plus optional truncation of the ψ-degree K, which is controlled by the same ε.

## Usage

```python
from fastpot import FastInternalField, vector_to_gh
T = FastInternalField(g, h, a=6371.2, rmin=6371.2+250, rmax=6371.2+800, eps=1e-10)  # g,h: (L+1,L+1) [n,m]
Br, Btheta, Bphi = T(r_km, theta_rad, phi_rad)          # B_north=-Btheta, B_east=Bphi, B_down=-Br
```

* `potential=True` also returns V.
* `precision="single"` is about 1.3–2× faster. It is suitable for ε ≈ 1e-6, but rounding grows with L
  (about 0.1 nT at L=13 and 0.7 nT at L=185).
* **Time-dependent models:** pass coefficient stacks `g[j], h[j]` and per-point weights W (M×J).
  Then B_i = Σ_j W_ij B_j(x_i), and all J×3 transforms share one point set.
  * IGRF: J=2 (main field + secular variation, or the two bracketing epochs), with weights `[1, t_i − t0]`.
  * CHAOS: split the data into knot intervals, and use the local polynomial (or the active B-spline) coefficient sets as the stack.
  * Recommended for CHAOS: evaluate the time-dependent core (n ≤ 20) directly or with a small J. Evaluate the static
    lithospheric part (n up to 185) with this transform, because that is where nearly all of the cost is.

## 2-D variant (radial direction outside the NUFFT)

The radial factor has low numerical rank on a shell. Write

    B_c(r,θ,φ) = Re Σ_{i<ρ} b_i(a/r) · G_{c,i}(θ,φ)

using an SVD-optimal radial basis b_i. On satellite shells ρ is 4–10, smaller than the Chebyshev K+1.
This needs only 2-D transforms on the doubled sphere, and costs about 3·ρ·w² operations per point
instead of 3·w³. The oversampled 2-D grids depend only on the model, so they are FFT'd once and cached.
No FFT is done per call.

* `FastInternalField2D` (in `fastpot.py`): the same split, using batched FINUFFT 2-D transforms. It is correct,
  but slower than 3-D at large M, because FINUFFT redoes kernel set-up and output for each of the 3ρ channels.
* `fused2d.FusedField2D`: one fused interpolation pass (numba) over all 3ρ channels, with cached fine grids.
  It matches or beats the FINUFFT 3-D path in most cases (`bench_fused.py`, `bench_fused.json`):
  1.1–2× at M = 2·10⁶, and up to 8× at M = 10⁵ for L = 185, because there is no per-call FFT.
  It uses a plain exponential-of-semicircle kernel (β = 2.3w), which reaches about 5ε at ε = 1e-10.

### Machine-precision operation

`FusedField2D(..., w=13, sigma=3.0, eps_radial=1e-15)` matches a 19-digit long-double reference to about 3e-14
relative at L = 185. That is the same as the direct double-precision sum (1.6e-14), and the fused transform is
still 42x faster than it (`refld.py`, `bench_exact.py`, `bench_exact.json`).
Other parameters: `w` is the kernel width, `sigma` the oversampling factor, `beta_per_w` the kernel shape, and
`eps_radial` the radial SVD tolerance.

## Files
* `fastfield.py`: **main implementation** (real-valued, half-sphere fused 2-D transform)
* `bench_backends.py`, `bench_final.py`, `bench_fastfield.py`, `fastfield_calibration.json` (+ `.json` results): its benchmarks and calibration
* `fastpot.py`: the transform
* `direct.py`: optimised direct synthesis baseline (numba, multithreaded, pole-safe recursions)
* `test_validate.py`: direct vs ppigrf (IGRF-14), fast vs direct, including the poles and a wide 1–6.6 a radial range
* `test_time.py`: time-dependent IGRF against ppigrf at random dates; single-precision check
* `bench.py`, `bench_results.json`, `bench.log`, `experiments.py`, `breakdown.json`: benchmarks
* `fused2d.py`, `bench2d.py`, `bench_fused.py` (+ `.json`): 2-D variants and their benchmarks
* `refld.py`, `bench_exact.py` (+ `.json`): extended-precision reference and machine-precision benchmark
* `doc/`: LaTeX source of the algorithm description (figures and tables regenerate with `gen.py`)

Requirements: `numpy`, `finufft`, `numba`; `ppigrf` is needed only for the tests.
