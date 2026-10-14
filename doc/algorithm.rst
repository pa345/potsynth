.. _algorithm:

*********
Algorithm
*********

This chapter describes the mathematics behind the library. It is meant for users who
want to know where the speed and the accuracy come from, and for developers.

Model and channels
==================

The library works with an internal potential expanded in 4-:math:`\pi` fully
normalized associated Legendre functions :math:`\bar{P}_l^m`, with coefficients
:math:`C_{lm}, S_{lm}`. Every quantity it computes has the form

.. math::
   :label: channel

   f_c(r,\theta,\phi) = \sum_{l=l_{min}}^{L} \sum_{m=0}^{\min(l,M)}
   \rho_c(l)\, u^{l+p_c}\; T_c\!\left[\bar{P}_l^m(\cos\theta)\right]\;
   D_c\!\left[ C_{lm} \cos m\phi + S_{lm} \sin m\phi \right],
   \qquad u = \frac{a}{r},

where :math:`l_{min}` and :math:`L` are the minimum and maximum degree, :math:`M` is the
maximum order, and :math:`c` labels a *channel*. Each channel is defined by:

- a per-degree factor :math:`\rho_c(l)`;
- a radial power offset :math:`p_c`;
- an operator :math:`T_c` in colatitude: :math:`\bar{P}`, :math:`d\bar{P}/d\theta`,
  :math:`\bar{P}/\sin\theta`, or one of the second-derivative operators
  :math:`\mathcal{A}`, :math:`\mathcal{B}` of the gradient tensor (see
  :ref:`gradient-tensor`);
- an operator :math:`D_c` in longitude, either the identity or :math:`d/d\phi`.

For example, the magnetic field :math:`\mathbf{B} = -\nabla V` of
:eq:`mag_potential` is the set of three channels

.. math::

   B_r &= \sum (l+1)\, u^{l+2}\, \bar{P}_l^m\, [\ldots], \\
   B_\theta &= -\sum u^{l+2}\, \frac{d\bar{P}_l^m}{d\theta}\, [\ldots], \\
   B_\phi &= -\sum u^{l+2}\, \frac{\bar{P}_l^m}{\sin\theta}\, \frac{\partial}{\partial\phi}[\ldots],

with :math:`[\ldots] = C_{lm}\cos m\phi + S_{lm}\sin m\phi` and
:math:`C_{lm} = g_l^m/\sqrt{2l+1}`, :math:`S_{lm} = h_l^m/\sqrt{2l+1}` (conversion from
Schmidt to 4-:math:`\pi` normalization). The gravity front-end uses the same mechanism
for :eq:`grav_potential`: :math:`V` is the channel :math:`(GM/a)\, u^{l+1} \bar{P}_l^m`,
and :math:`\mathbf{g} = \nabla V` has the three channels above with factors
:math:`-(GM/a^2)(l+1)` and :math:`GM/a^2`, without the minus signs.

Overview of the fast method
===========================

The fast method has two phases.

**Set-up.** This is done once per model and radial range.

1. Separate the radial dependence into :math:`\rho` optimal radial functions
   :math:`b_i(u)`, using an SVD.
2. Write each angular function that multiplies a :math:`b_i` as an *exact* 2-D Fourier
   series on the doubled sphere.
3. Pre-sharpen each series and transform it by FFT to an oversampled grid.
4. Keep only the grid rows that evaluation points can reach.

**Evaluation.** This is done once per point.

5. Compute the :math:`\rho` radial values :math:`b_i(a/r)`, form a :math:`w \times w`
   weighted sum of nearby grid values for all channels at once, and combine the two.

Step 5 costs :math:`O(\rho\, w^2)` operations per point, which does not depend on
:math:`L`. Steps 3 and 5 together are a type-2 *non-uniform FFT* (NUFFT). Unlike a
general NUFFT library, the FFT is done once, in the set-up, and the window sum is
written for this specific structure.

Step 1: radial separation
=========================

The radius enters :eq:`channel` only through the powers :math:`u^{l+p}`. All points
lie in a shell :math:`r_{\min} \le r \le r_{\max}`, so :math:`u` lies in an interval
:math:`[u_0, u_1] = [a/r_{\max}, a/r_{\min}]`. On a narrow interval the functions
:math:`u^{l+p}` for different :math:`l` look very much alike, so they are numerically
low-rank:

.. math::

   u^{l+p_c} \approx \sum_{i=1}^{\rho} \beta_{c,i,l}\, b_i(u),
   \qquad u \in [u_0, u_1].

Substituting this into :eq:`channel` separates the radius from the angles:

.. math::
   :label: separated

   f_c(r,\theta,\phi) \approx \sum_{i=1}^{\rho} b_i(u)\, G_{c,i}(\theta,\phi),
   \qquad
   G_{c,i} = \sum_{l,m} \beta_{c,i,l}\, \rho_c(l)\, T_c[\bar{P}_l^m]\, D_c[\ldots].

The optimal basis comes from an SVD. The functions are sampled at the :math:`K+1`
Chebyshev--Lobatto points :math:`u_j = u_c + u_d \cos(\pi j/K)`, where
:math:`u_c = (u_0+u_1)/2` and :math:`u_d = (u_1-u_0)/2`. These samples, weighted by
each function's contribution to the field, form the matrix

.. math::

   X_{j,(p,l)} = \sqrt{c_j}\; w_{p,l}\; u_j^{l+p}, \qquad
   w_{p,l} = \max_{c:\,p_c = p} |\rho_c(l)|\, \tau_c(l)\, \|C_l\|.

Here:

- :math:`c_j` are the Lobatto weights: 1/2 at the end points and 1 otherwise.
- :math:`\|C_l\|^2 = \sum_m (C_{lm}^2 + S_{lm}^2)`.
- :math:`\tau_c(l)` is the r.m.s. size of the operator :math:`T_c D_c`: 1 for
  :math:`\bar{P}` and :math:`\sqrt{l(l+1)/2}` for the horizontal derivatives.

:math:`K` is increased until the Chebyshev series of all functions are resolved, then
reduced to the smallest degree whose truncation tail stays below tolerance.

A one-sided Jacobi SVD, :math:`X = U S W^T`, gives singular values accurate to the
level of rounding, which the machine-precision mode needs. The rank :math:`\rho` is the
number of singular values above :math:`\tau_r s_1`, where :math:`\tau_r = 0.1\varepsilon`
by default. Then:

.. math::

   b_i(u_j) = U_{ji}/\sqrt{c_j}, \qquad
   \beta_{c,i,l} = \sum_j U_{ji}\sqrt{c_j}\, u_j^{l+p_c},

The second formula is the projection onto the basis in the discrete Chebyshev inner
product. Each :math:`b_i` is stored as a Chebyshev series in
:math:`t = (u-u_c)/u_d \in [-1,1]`, so evaluating all :math:`\rho` radial values at a
point costs about :math:`\rho K` operations.

For a satellite shell (250--800 km), :math:`\rho = 5` gives :math:`10^{-6}` accuracy
and :math:`\rho \approx 12` gives :math:`10^{-14}`. Thicker shells need more radial
functions: :math:`r \in [a, 2a]` needs :math:`\rho \approx 23` at :math:`10^{-14}`. When
:math:`r_{\min} = r_{\max}` a single constant function is used.

Step 2: exact Fourier series on the doubled sphere
==================================================

Colatitude only runs from 0 to :math:`\pi`. Extend it to a full circle,
:math:`\theta \in [0, 2\pi)`, keeping :math:`\cos\theta` and :math:`\sin\theta` as usual
(so :math:`\sin\theta < 0` for :math:`\theta > \pi`). This is the *doubled sphere*. On
it, :math:`\bar{P}_l^m(\cos\theta)` and :math:`d\bar{P}_l^m/d\theta` are trigonometric
polynomials of degree at most :math:`l`, and for :math:`m \ge 1` so is
:math:`\bar{P}_l^m/\sin\theta`, of degree at most :math:`l-1`. The reason is that
:math:`\bar{P}_l^m = \sin^m\theta\, q(\cos\theta)`, where :math:`q` is a polynomial. In
particular, :math:`\bar{P}/\sin\theta` has no singularity at the poles. Every angular
function in :eq:`separated` is therefore an **exact** 2-D trigonometric polynomial of
degree at most :math:`L`:

.. math::

   G_{c,i}(\theta,\phi) = \mathrm{Re} \sum_{m=0}^{M} H_m(\theta)\, e^{im\phi}, \qquad
   H_m(\theta) = \sum_{|p| \le L} \hat{H}_m[p]\, e^{ip\theta}.

For each order :math:`m`, the functions :math:`H_m = S_A - i S_B` are evaluated at
:math:`N_\theta \ge 2L+2` equally spaced colatitudes :math:`\theta_j = (j+\tfrac12)\,
2\pi/N_\theta`. Here :math:`S_A` and :math:`S_B` are the sums over :math:`l` that
multiply :math:`\cos m\phi` and :math:`\sin m\phi`. The half-step shift keeps the nodes
off the poles. One FFT per channel then gives the coefficients
:math:`\hat{H}_m[p]` *exactly*. This step makes no approximation.

**Symmetries.** :math:`N_\theta` is a multiple of 4, and the Legendre recursion runs
only on the nodes in :math:`(0, \pi/2)`. The other three quadrants follow from
symmetry. The sums over :math:`l` are split by the parity of :math:`l-m` into
:math:`E` (even) and :math:`O` (odd). Then:

.. list-table::
   :header-rows: 1

   * - node
     - :math:`\bar{P}`, :math:`\bar{P}/\sin\theta`
     - :math:`d\bar{P}/d\theta`
   * - :math:`\theta_j`
     - :math:`E + O`
     - :math:`E + O`
   * - :math:`\pi - \theta_j`
     - :math:`E - O`
     - :math:`O - E`
   * - :math:`2\pi - \theta_j`
     - :math:`\sigma (E + O)`
     - :math:`\sigma (E + O)`
   * - :math:`\pi + \theta_j`
     - :math:`\sigma (E - O)`
     - :math:`\sigma (O - E)`

with :math:`\sigma = (-1)^m` for :math:`\bar{P}`, and :math:`\sigma = (-1)^{m-1}` for
:math:`\bar{P}/\sin\theta` and :math:`d\bar{P}/d\theta`. This cuts the recursion work
by a factor of four.

**Fused accumulation.** The Legendre values are never stored. In blocks of 8 nodes,
each column recursion (fixed :math:`m`, increasing :math:`l`) is consumed immediately
by a register-blocked kernel. The kernel accumulates the even and odd sums of every
(channel, radial function) pair with precomputed weights
:math:`\beta_{c,i,l}\rho_c(l)C_{lm}`. Coefficients and recursion factors are stored in
*M-major* order (fixed :math:`m` contiguous in :math:`l`), which is the order the
recursion reads them in. The orders :math:`m` are distributed over threads.

Step 3: pre-sharpening and the fine grid
========================================

Consider a 1-D trigonometric polynomial :math:`f(x) = \sum_{|k|\le N} c_k e^{ikx}` that
has to be evaluated at arbitrary points. Choose a short window :math:`\phi(x)` that
covers :math:`w` grid steps of a fine grid :math:`x_\nu = \nu h`, :math:`h = 2\pi/n_f`.
Instead of the samples of :math:`f`, store the *pre-sharpened* values

.. math::

   g_\nu = \sum_{|k|\le N} c_k\, \frac{h}{\hat{\phi}(k)}\, e^{ikx_\nu},
   \qquad \hat{\phi}(k) = \int \phi(x)\, e^{-ikx}\, dx.

The window sum :math:`\tilde{f}(x) = \sum_\nu g_\nu\, \phi(x - x_\nu)` involves only the
:math:`w` grid points near :math:`x`. By the Poisson summation formula,

.. math::

   \tilde{f}(x) = f(x) + \sum_k c_k \sum_{j \ne 0}
   \frac{\hat{\phi}(k - j n_f)}{\hat{\phi}(k)}\, e^{i(k - j n_f)x}.

The :math:`j=0` term reproduces :math:`f` exactly, because the factor
:math:`h/\hat\phi(k)` cancels the smoothing of the window. The remaining terms are
aliases. They are small when :math:`\hat\phi` is large on the band :math:`|k| \le N`
and small at the alias frequencies :math:`|k - j n_f| \ge n_f - N`. This is achieved
with an *oversampled* grid, :math:`n_f \approx \sigma(2N+1)` with :math:`\sigma = 2`,
and the "exponential of semicircle" window (Barnett et al., 2019),

.. math::

   \phi(x) = \exp\!\left(\beta\left(\sqrt{1-z^2} - 1\right)\right), \quad
   z = \frac{x}{wh/2}, \quad
   \beta = \pi\left(1 - \frac{1}{2\sigma}\right) w.

The alias ratio is then about :math:`e^{-\beta}`, so the error falls by about one
decimal digit for each extra window point. :math:`\hat\phi(k)` is computed by quadrature
with the substitution :math:`z = \sin t`. This removes the square-root singularity of
:math:`\phi` at the window edges, so Gauss--Legendre quadrature converges exponentially.

In two dimensions the window is the product :math:`\phi(\theta)\phi(\phi)`. For each
order :math:`m`, the coefficients :math:`\hat{H}_m[p]` are multiplied by
:math:`h_1 h_2/(\hat\phi_1(p)\hat\phi_2(m))`, zero-padded, and transformed by an inverse
FFT of length :math:`n_{f,1}` to the fine colatitude grid. The real function
:math:`G_{c,i}` is stored directly as a real grid. Along longitude each row is a
half-complex spectrum (:math:`X_0 = \mathrm{Re}\, T_0`, :math:`X_m = T_m/2`), which one
inverse real FFT of length :math:`n_{f,2}` per row and channel turns into grid values.
These spectra are written into the grid itself, so peak memory during set-up is about
the size of the final grid.

Step 4: storing only what is needed
===================================

Points have :math:`\theta \in [0,\pi]`, and a window reaches :math:`w/2` grid steps
either side. So only the rows :math:`-w/2 - 1 \lesssim \nu \lesssim \pi/h_1 + w/2 + 1`
are stored, about half the doubled-sphere grid plus a margin.

- **Rows below 0 and above** :math:`\pi`: these hold values from the mirror half of the
  doubled sphere. They serve points near a pole, whose windows straddle it. Because the
  angular functions are smooth across the poles, the window sum is as accurate there as
  anywhere else, so the poles need no special treatment.
- **Padding columns:** the grid has :math:`w/2 + 1` periodically repeated columns on
  each side, so a window never wraps around in longitude.
- **Channel layout:** all :math:`n_{ch} = \rho \times` (number of channels) values of a
  grid node are stored next to each other, padded to a multiple of 4. One row of a
  point's window is then a single contiguous block of :math:`w\, n_{ch}` values.

Step 5: evaluation
==================

For a point :math:`(r,\theta,\phi)`:

1. **Window weights.** With :math:`\ell_1 = \lceil \theta/h_1 - w/2 \rceil` and
   :math:`\ell_2 = \lceil \phi/h_2 - w/2 \rceil`, compute the :math:`2w` weights
   :math:`\phi(\theta - (\ell_1+a)h_1)` and :math:`\phi(\phi - (\ell_2+b)h_2)`. The
   indices are clamped to the stored grid, so no input can read out of bounds.
2. **Window sum, all channels at once:**

   .. math::

      s_q = \sum_{a,b=0}^{w-1} \phi_1[a]\, \phi_2[b]\; g_q[\ell_1 + a, \ell_2 + b],
      \qquad q = 1 \ldots n_{ch}.

   This is the main cost, :math:`n_{ch} w^2` multiply-adds. The kernel processes 8
   channels at a time with 4 independent partial sums each, all in vector registers,
   which hides the latency of the fused multiply-add.
3. **Radial values** :math:`b_i(a/r)`, from their Chebyshev series.
4. **Combination:** :math:`f_c = \sum_i b_i\, s_{c,i}`.

Points are processed in chunks. Within a chunk they are counting-sorted by grid block,
so consecutive points read neighbouring memory, and blocks of points are distributed
over threads. All inputs are validated before any output is written. Points with a
non-finite coordinate give NaN outputs.

The whole fast method is one linear map,
:math:`\mathbf{f} = S\, \mathcal{F}^{-1} D\, \Lambda\, R\, \mathbf{C}`:

- :math:`R` is the radial split;
- :math:`\Lambda` is the exact Legendre-to-Fourier map;
- :math:`D` is the diagonal pre-sharpening;
- :math:`\mathcal{F}^{-1}` is the inverse FFT;
- :math:`S` is the sparse window/radial evaluation.

Only :math:`S` (through :math:`w`) and :math:`R` (through :math:`\rho`) are approximate.

The direct method
=================

The direct method evaluates :eq:`channel` point by point:

- **Blocks of points.** Points are taken 8 at a time. For each order :math:`m`, a single
  pass over :math:`l` runs the Legendre column recursion (vectorized across 4 points),
  multiplies by the running radial factor :math:`u^l`, and adds the term to every
  channel. All accumulators stay in vector registers. Specialized kernels exist for
  1--6 channels, with a generic version otherwise.
- **Quantities carried across orders.** :math:`\sin^{m-1}\theta` (as mantissa and
  exponent), :math:`u^m`, and :math:`\cos m\phi`, :math:`\sin m\phi` are updated from
  one order to the next. The angles use a rotation that is recomputed exactly every 16
  orders, so rounding errors cannot build up.

No table of Legendre functions is ever stored.

Associated Legendre functions
=============================

The recursions are those of ``gsl_sf_alf_vsh_array`` (GSL), which follow Holmes &
Featherstone (2002), with the 4-:math:`\pi` recurrence factors

.. math::

   a_l^m = \sqrt{\frac{(2l+1)(2l-1)}{(l+m)(l-m)}}, \quad
   b_l^m = -\sqrt{\frac{(l+m-1)(l-m-1)(2l+1)}{(l+m)(l-m)(2l-3)}}, \quad
   c_m = \sqrt{2m+3}.

For :math:`m \ge 1` the recursion runs on the *scaled, pole-safe* quantities

.. math::

   Q_l^m = 2^{-s_m}\, \frac{\bar{P}_l^m}{\sin^m\theta}, \qquad
   E_l^m = 2^{-s_m}\, \frac{d\bar{P}_l^m/d\theta}{\sin^{m-1}\theta}.

Both are polynomials in :math:`x = \cos\theta`:

.. math::

   Q_m^m &= 2^{-s_m} \prod_{k=1}^{m} d_k, \qquad E_m^m = m\, x\, Q_m^m, \\
   Q_{m+1}^m &= c_m\, x\, Q_m^m, \qquad E_{m+1}^m = c_m \left(x E_m^m - s^2 Q_m^m\right), \\
   Q_l^m &= a_l^m\, x\, Q_{l-1}^m + b_l^m\, Q_{l-2}^m, \qquad
   E_l^m = a_l^m \left(x E_{l-1}^m - s^2 Q_{l-1}^m\right) + b_l^m E_{l-2}^m,

with :math:`s = \sin\theta`, :math:`d_1 = \sqrt{3}` and :math:`d_k = \sqrt{1 + 1/(2k)}`.
No step divides by :math:`\sin\theta`. Then

.. math::

   \bar{P}_l^m = 2^{s_m} s^m Q_l^m, \qquad
   \frac{\bar{P}_l^m}{\sin\theta} = 2^{s_m} s^{m-1} Q_l^m, \qquad
   \frac{d\bar{P}_l^m}{d\theta} = 2^{s_m} s^{m-1} E_l^m.

The powers of :math:`s` are applied as a mantissa and exponent, so they never
underflow or overflow. Only :math:`s^2` enters the recursion, so it works unchanged on
the doubled sphere (:math:`s < 0`) and at the poles (:math:`s = 0`).

**Scaling.** :math:`Q_l^m` is largest at the poles, where

.. math::

   Q_l^m(\pm 1) = 2^{-s_m} \sqrt{2(2l+1)}\, \frac{\sqrt{(l+m)!/(l-m)!}}{2^m\, m!}.

The exponent :math:`s_m \ge 0` is the smallest power of two that keeps all values
below :math:`2^{960}`. Most columns need no scaling at all: none do below
:math:`L \approx 1370`. Because the scaling is an exact power of two, it costs no
accuracy. It is what lets the method run in double precision up to
:math:`L = 2800`.

GSL's ``gsl_sf_alf_array`` and ``gsl_sf_alf_vsh_array`` compute the diagonal seed
:math:`\bar{P}_m^m` without this rescaling. When :math:`\sin^m\theta` underflows, at
degrees above about 1900, whole columns come out zero or wrong. For example, at
:math:`L = 2190`, :math:`\theta = 0.4`, the value :math:`\bar{P}_{2054}^{793} = 6.69`
(confirmed with 60-digit arithmetic) comes out as 0 or 10.2. The scaled columns above
give the correct value.

**Accuracy near the poles.** Close to :math:`x = \pm 1` the three-term recursion
behaves like :math:`P_l \approx 2P_{l-1} - P_{l-2}`, so rounding errors grow linearly
at each step. The relative error is then about :math:`\varepsilon L^2`, roughly
:math:`10^{-12}` at :math:`L = 185` and :math:`10^{-10}` at :math:`L = 2190`. This
holds for any implementation of the standard recursion, including GSL.

.. _gradient-tensor:

The gradient tensor
===================

The second derivatives of the potential appear in two places:

- the magnetic gradient tensor :math:`\nabla\mathbf{B} = -\nabla\nabla V`;
- the gravity gradient tensor (the quantity measured by GOCE)
  :math:`\Gamma = \nabla\nabla V`, usually given in Eötvös units,
  :math:`1\ \mathrm{E} = 10^{-9}\ \mathrm{s}^{-2}`.

Both are the Hessian :math:`T = \nabla\nabla V`, up to sign. Because :math:`V` is a
potential, :math:`T` is symmetric, and because :math:`\nabla^2 V = 0`, it is traceless.
It therefore has five independent components.

**Components in the spherical basis.** In the basis
:math:`(\hat{r}, \hat{\theta}, \hat{\phi})`, the covariant derivative gives

.. math::

   T_{rr} &= \partial_r^2 V, &
   T_{r\theta} &= \frac{1}{r}\, \partial_r \partial_\theta V - \frac{1}{r^2}\, \partial_\theta V, \\
   T_{\theta\theta} &= \frac{1}{r^2}\, \partial_\theta^2 V + \frac{1}{r}\, \partial_r V, &
   T_{r\phi} &= \frac{1}{r \sin\theta}\, \partial_r \partial_\phi V
                - \frac{1}{r^2 \sin\theta}\, \partial_\phi V, \\
   T_{\phi\phi} &= \frac{1}{r^2 \sin^2\theta}\, \partial_\phi^2 V + \frac{1}{r}\, \partial_r V
                   + \frac{\cot\theta}{r^2}\, \partial_\theta V, &
   T_{\theta\phi} &= \frac{1}{r^2 \sin\theta}\, \partial_\phi
                     \left( \partial_\theta V - \cot\theta\, V \right).

Apply these to one term :math:`V = u^{l+1}\, \bar{P}_l^m(\cos\theta)\, \Phi_m(\phi)`, with
:math:`u = a/r` and :math:`\Phi_m = C_{lm}\cos m\phi + S_{lm}\sin m\phi`. Each component
becomes :math:`u^{l+3}/a^2` times one angular function:

.. math::
   :label: hessian_channels

   T_{rr} &= \frac{u^{l+3}}{a^2}\, (l+1)(l+2)\, \bar{P}_l^m\, \Phi_m, &
   T_{r\theta} &= -\frac{u^{l+3}}{a^2}\, (l+2)\, \frac{d\bar{P}_l^m}{d\theta}\, \Phi_m, \\
   T_{r\phi} &= -\frac{u^{l+3}}{a^2}\, (l+2)\, \frac{\bar{P}_l^m}{\sin\theta}\, \Phi_m', &
   T_{\theta\phi} &= \frac{u^{l+3}}{a^2}\, \mathcal{B}_l^m\, \Phi_m', \\
   T_{\theta\theta} - T_{\phi\phi} &= \frac{u^{l+3}}{a^2}\, \mathcal{A}_l^m\, \Phi_m,

with the two new angular operators

.. math::

   \mathcal{A}_l^m = \frac{d^2\bar{P}_l^m}{d\theta^2} - \cot\theta\, \frac{d\bar{P}_l^m}{d\theta}
                     + \frac{m^2}{\sin^2\theta}\, \bar{P}_l^m, \qquad
   \mathcal{B}_l^m = \frac{1}{\sin\theta}
                     \left( \frac{d\bar{P}_l^m}{d\theta} - \cot\theta\, \bar{P}_l^m \right).

The remaining components follow from the trace:
:math:`T_{\theta\theta} = \tfrac12 (-T_{rr} + D)` and
:math:`T_{\phi\phi} = \tfrac12 (-T_{rr} - D)`, where :math:`D = T_{\theta\theta} - T_{\phi\phi}`.
So the five channels :math:`T_{rr}, T_{r\theta}, T_{r\phi}, D, T_{\theta\phi}` give the
whole tensor. The first three use the existing operators :math:`\bar{P}`,
:math:`d\bar{P}/d\theta` and :math:`\bar{P}/\sin\theta`.

**Removing the divisions by** :math:`\sin\theta`. :math:`\mathcal{A}` and
:math:`\mathcal{B}` contain :math:`1/\sin\theta` and :math:`1/\sin^2\theta`, but the
tensor itself is bounded everywhere. The divisions are only apparent. Every
:math:`\bar{P}_l^m` contains the factor :math:`\sin^m\theta`, and the divisions cancel
against it exactly. Write :math:`s = \sin\theta`, :math:`x = \cos\theta`, and

.. math::

   \bar{P}_l^m = s^m Q_l^m, \qquad Q_l^m = N_l^m\, \frac{d^m P_l}{dx^m}, \qquad
   N_l^m = \sqrt{(2 - \delta_{m0})\,(2l+1)\, \frac{(l-m)!}{(l+m)!}},

as in the previous section, without the power-of-two scaling. Then
:math:`d\bar{P}_l^m/d\theta = s^{m-1} E_l^m` with
:math:`E_l^m = m\,x\,Q_l^m - (1-x^2)\, dQ_l^m/dx`. Because :math:`Q_l^m` is the
:math:`m`-th derivative of :math:`P_l`, its derivative is the next order:

.. math::

   \frac{dQ_l^m}{dx} = k_l^m\, Q_l^{m+1}, \qquad
   k_l^0 = \sqrt{\tfrac12\, l(l+1)}, \qquad k_l^m = \sqrt{(l-m)(l+m+1)} \quad (m \ge 1).

Substituting into :math:`\mathcal{A}` and :math:`\mathcal{B}` gives

.. math::

   \mathcal{A}_l^m &= 2\, s^{m-2} \left( m^2 Q_l^m - x\, E_l^m \right) - l(l+1)\, s^m Q_l^m, \\
   \mathcal{B}_l^m &= s^{m-2} \left( E_l^m - x\, Q_l^m \right).

For :math:`m \ge 2` the power :math:`s^{m-2}` is non-negative, so these are evaluated as
written: no division occurs. For :math:`m = 0` and :math:`m = 1` the bracketed terms
contain the factor :math:`1 - x^2 = s^2`, which cancels algebraically:

.. math::

   \mathcal{A}_l^0 &= 2\, k_l^0\, x\, Q_l^1 - l(l+1)\, Q_l^0, &
   \mathcal{B}_l^0 &\ \text{(not needed: multiplied by } \Phi_0' = 0), \\
   \mathcal{A}_l^1 &= s \left[ \left( 2 - l(l+1) \right) Q_l^1 + 2\, k_l^1\, x\, Q_l^2 \right], &
   \mathcal{B}_l^1 &= -k_l^1\, s\, Q_l^2.

These need the columns :math:`Q^1` and :math:`Q^2`, which the recursion computes anyway.
For :math:`m \le 3` the scale exponents :math:`s_m` are zero at every supported degree,
so the identities hold unchanged for the scaled columns.

Evaluating the divisions numerically would also be inaccurate near the poles, not just
at them. For example, :math:`(Q_l^1 - x E_l^1)/s` subtracts two nearly equal numbers
and then divides by a small one, which loses relative accuracy like
:math:`\varepsilon/\theta^2`. The algebraic forms above have no such loss.

**Values at the poles.** At :math:`s = 0`, only :math:`m = 0` and :math:`m = 2`
contribute to :math:`\mathcal{A}`, and only :math:`m = 2` to :math:`\mathcal{B}`.
Every other term carries a positive power of :math:`s`.

- For :math:`m = 0`, :math:`\mathcal{A}_l^0 = 0` at the poles, because
  :math:`2\, k_l^0\, Q_l^1(1) = l(l+1)\sqrt{2l+1} = l(l+1)\, \bar{P}_l^0(1)`, and likewise
  at :math:`x = -1`. This expresses the fact that a zonal field has an isotropic
  horizontal tensor on the axis.
- As for :math:`B_\theta` and :math:`B_\phi`, the components at a pole refer to the
  local frame defined by the longitude :math:`\phi` of the point. They are the limits
  approached along that meridian.

To get components that are unique at the poles, rotate to Cartesian components with
:math:`T^c = R^T T^s R`, where :math:`R` is the rotation from Cartesian to spherical
components. The rotation is regular everywhere.

**Implementation.** The library uses a form that needs no case distinction at all.
Let :math:`G_l^m = -dQ_l^m/dx`. Differentiating the column recursion for
:math:`Q_l^m` gives a recursion for :math:`G_l^m` with the same coefficients:

.. math::

   G_l^m = a_l^m \left( x\, G_{l-1}^m - Q_{l-1}^m \right) + b_l^m\, G_{l-2}^m,
   \qquad G_m^m = 0.

Since :math:`E_l^m = m\,x\,Q_l^m + s^2 G_l^m`, the operators become, for every
:math:`m`,

.. math::

   \mathcal{A}_l^m &= 2m(m-1)\, s^{m-2} Q_l^m
                      + s^m \left[ \left( 2m - l(l+1) \right) Q_l^m - 2x\, G_l^m \right], \\
   \mathcal{B}_l^m &= (m-1)\, x\, s^{m-2} Q_l^m + s^m G_l^m \qquad (m \ge 1).

The prefactors :math:`m(m-1)` and :math:`m-1` vanish exactly at the orders where
:math:`s^{m-2}` would be singular, so no division by :math:`\sin\theta` occurs at any
order. In the direct method, each channel is a sum of *terms*: the sum over :math:`l` of
one column (:math:`Q`, :math:`E` or :math:`G`) with a weight row, multiplied afterwards
by a power of :math:`s` (and possibly :math:`x`). :math:`\mathcal{A}` has three terms
and :math:`\mathcal{B}` two, all fed by the same pass of the recursion. Terms are
accumulated in vector registers in groups of up to six, and :math:`G` is only computed
when a channel needs it. The full tensor (five channels, eight terms) costs about three
times as much per point as the three components of :math:`\mathbf{B}`.

The fast method does not support these operators yet: a model that uses them is
evaluated with the direct method.

**Bandwidth.** On the doubled sphere, :math:`\mathcal{A}_l^m` and :math:`\mathcal{B}_l^m`
are trigonometric polynomials in :math:`\theta` of degree at most :math:`l`. For
:math:`m \ge 2`, :math:`s^{m-2}` times a polynomial in :math:`x` of degree
:math:`l - m + 1` has degree :math:`l - 1`, and :math:`s^m Q_l^m` has degree :math:`l`;
the cases :math:`m = 0, 1` are similar. The grid of the fast method therefore needs no
extra resolution for the gradient tensor. Only two more angular operators are evaluated
during set-up.

**Verification.** All of these identities were checked with 50-digit arithmetic:

- the formulas for :math:`E_l^m`, :math:`\mathcal{A}_l^m` and :math:`\mathcal{B}_l^m`,
  against the definitions by numerical differentiation, for :math:`l \le 8`, all
  :math:`m`, and points down to :math:`10^{-6}` from both poles;
- the pole values, against the limits approached along a meridian;
- the full component table :eq:`hessian_channels`, against the Cartesian Hessian
  :math:`\partial^2 V/\partial x_i \partial x_j` rotated to the spherical basis, with
  agreement to :math:`10^{-40}`.

Accuracy and parameter choice
=============================

The fast method has three error sources:

- the window, controlled by :math:`w`;
- truncation of the radial SVD, controlled by :math:`\rho` (tolerance
  :math:`0.1\varepsilon`);
- floating-point rounding.

The requested accuracy :math:`\varepsilon` is relative to the largest value
:math:`\max|f|` of each channel. Channels of very different size, such as a potential and
its gradient, are each resolved to :math:`\varepsilon`, because the radial basis weights
each channel by its own size. Coefficients that do not contribute to a channel (order 0
for :math:`\partial/\partial\phi` and :math:`\bar{P}_l^m/\sin\theta`) are left out of
its weights. The window width is the smallest :math:`w` whose *measured* error is
below :math:`\varepsilon`. The measurements below are against 19-digit reference values,
taking the worst case over :math:`L = 13` and :math:`L = 185` and over a satellite shell
and a thick shell:

=====  ==============================  ===========================
w      :math:`\sigma = 2` (default)     :math:`\sigma = 3`
=====  ==============================  ===========================
4      :math:`8 \times 10^{-4}`        :math:`2 \times 10^{-4}`
5      :math:`2 \times 10^{-5}`        :math:`3 \times 10^{-5}`
6      :math:`9 \times 10^{-6}`        :math:`1 \times 10^{-6}`
7      :math:`7 \times 10^{-7}`        :math:`2 \times 10^{-7}`
8      :math:`3 \times 10^{-8}`        :math:`7 \times 10^{-9}`
9      :math:`5 \times 10^{-9}`        :math:`7 \times 10^{-10}`
10     :math:`6 \times 10^{-10}`       :math:`6 \times 10^{-11}`
11     :math:`3 \times 10^{-11}`       :math:`3 \times 10^{-12}`
12     :math:`6 \times 10^{-12}`       :math:`4 \times 10^{-13}`
13     :math:`4 \times 10^{-13}`       :math:`1 \times 10^{-14}`
14     :math:`3 \times 10^{-14}`       :math:`1 \times 10^{-14}`
15     :math:`1 \times 10^{-14}`       :math:`7 \times 10^{-15}`
16     :math:`7 \times 10^{-15}`       :math:`9 \times 10^{-15}`
=====  ==============================  ===========================

The error does not depend on :math:`L` or on the radial range. About
:math:`5 \times 10^{-15}` is the rounding floor of double precision; the direct method
reaches :math:`1\text{--}2 \times 10^{-15}`. :math:`\sigma = 3` gives the same accuracy
with a smaller :math:`w` (a faster evaluation), but uses a grid 2.25 times larger.

Cost and memory
===============

**Set-up.**

- The Legendre recursion and accumulation take about
  :math:`L^2 N_\theta/8 \times (8 + 4 n_{ch})` operations, distributed over the orders
  :math:`m`.
- The FFTs take :math:`O(n_{ch} L^2 \log L)`.

In practice set-up takes about 0.07 s at :math:`L = 185` and about 1 s at :math:`L = 720`
on 16 threads.

**Memory.** The grid holds :math:`n_{rows} \times n_{cols} \times n_{ch}` doubles, with
:math:`n_{rows} \approx n_f/2 + w`, :math:`n_{cols} \approx n_f + 1.5w` and
:math:`n_f \approx \sigma(2L+1)`. That is roughly :math:`8\sigma^2 L^2\, n_{ch}` bytes:
64 MB at :math:`L = 185` and :math:`\varepsilon = 10^{-10}`, 0.9 GB at :math:`L = 720`.

**Evaluation per point.**

- :math:`2w` window weights;
- :math:`n_{ch} w^2` multiply-adds (the dominant term);
- about :math:`\rho K` operations for the radial values.

None of these depend on :math:`L`. At :math:`\varepsilon = 10^{-10}`
(:math:`w = 11`, :math:`n_{ch} = 27`) this is about 3300 multiply-adds, or 0.5 µs per
point on one core.

**Direct method:** :math:`O(L^2)` per point. It is faster than the fast method up to
:math:`L \approx 20\text{--}30`, depending on the accuracy and the number of threads.
``POTSYNTH_METHOD_AUTO`` uses the direct method for :math:`L < 24`.

.. _choosing-method:

Choosing between the methods
============================

The fast method has a one-time set-up cost: building the grid. After that, each point
is much cheaper than with the direct method. So the direct method is faster for few
points and the fast method for many. The table gives measured times on 16 threads with
:math:`\varepsilon = 10^{-10}`: magnetic models with 3 channels, and EGM2008
(:math:`L = 2190`) with 1 channel.

=====  =========  ===============  =============  ==================
L      set-up     direct, per pt   fast, per pt   break-even points
=====  =========  ===============  =============  ==================
13                0.023 µs         0.044 µs       never
30     0.005 s    0.080 µs         0.051 µs       about 170 000
60     0.008 s    0.21 µs          0.053 µs       about 50 000
185    0.073 s    1.6 µs           0.072 µs       about 48 000
400    0.25 s     10.8 µs          0.076 µs       about 23 000
720    1.1 s      39 µs            0.19 µs        about 28 000
2190   7.6 s      222 µs           0.24 µs        about 35 000
=====  =========  ===============  =============  ==================

Above about :math:`L = 50`, the break-even lies between about 20 000 and 50 000 points,
because both the set-up and the direct sum grow with :math:`L`. Beyond it the fast
method wins by a growing factor: about 30 times at :math:`10^6` points and 200 times at
:math:`10^7` points for EGM2008.

If the expected number of points is given (``npts_hint``), ``POTSYNTH_METHOD_AUTO``
compares the estimated totals :math:`t_{setup} + n\,t_{fast}` and :math:`n\,t_{direct}`
using these empirical cost models (in seconds):

.. math::

   t_{setup} &= 6.3 \times 10^{-12}\, n_{ch} L^3 + 7.8 \times 10^{-8}\, n_{ch} L^2, \\
   t_{direct} &= 3.5 \times 10^{-11}\, L^2 \left(1 + (n_{chan} - 1)/2\right), \\
   t_{fast} &= 1.5 \times 10^{-11}\, w^2 n_{ch} + 5 \times 10^{-8},

where :math:`n_{ch} = n_{chan}\,\rho`, and the radial rank is estimated as
:math:`\rho \approx 2 + 0.7 \log_{10}(1/\varepsilon)`. The models reproduce the
measured break-even points to within a factor of two. Near the break-even both methods
take about the same time, so an imperfect choice costs little. Without a hint, the fast
method is used for :math:`L \ge 24`.

References
==========

- A. H. Barnett, J. Magland, L. af Klinteberg, A parallel nonuniform fast Fourier
  transform library based on an "exponential of semicircle" kernel, *SIAM J. Sci.
  Comput.* 41(5), C479--C504, 2019.
- A. Dutt, V. Rokhlin, Fast Fourier transforms for nonequispaced data, *SIAM J. Sci.
  Comput.* 14(6), 1368--1393, 1993.
- S. A. Holmes, W. E. Featherstone, A unified approach to the Clenshaw summation and
  the recursive computation of very high degree and order normalised associated
  Legendre functions, *J. Geodesy* 76, 279--299, 2002.
- S. Kunis, D. Potts, Fast spherical Fourier algorithms, *J. Comput. Appl. Math.* 161,
  75--98, 2003.
- N. Schaeffer, Efficient spherical harmonic transforms aimed at pseudospectral
  numerical simulations, *Geochem. Geophys. Geosyst.* 14(3), 751--758, 2013.
- M. Reinecke, pocketfft, https://gitlab.mpcdf.mpg.de/mtr/pocketfft.
