"""
Verify the identities behind the gradient tensor operators (documentation,
"The gradient tensor") with 50-digit arithmetic (mpmath). Standalone: not part
of the C test suite.

Checks:
  1. E = (dPbar/dtheta)/sin^(m-1) = m x Q - (1-x^2) dQ/dx
  2. the forms of A = d2P - cot dP + m^2 P/sin^2 and B = (dP - cot P)/sin
     without division by sin(theta), per order (m = 0, 1, >= 2) and unified
     with G = -dQ/dx, against the definitions, down to 1e-6 from the poles
  3. the recursion G_l = a (x G_{l-1} - Q_{l-1}) + b G_{l-2} used by the library
  4. pole values: A(m=0) = 0, and A(m=2), B(m=2) equal the limits
  5. the full component table against the Cartesian Hessian rotated to the
     spherical basis

Usage: python verify_hessian.py
"""
import sys

import mpmath as mp

mp.mp.dps = 50


def N(l, m):
    return mp.sqrt((1 if m == 0 else 2) * (2 * l + 1) * mp.factorial(l - m) / mp.factorial(l + m))


def Q(l, m, x):
    """Pbar_l^m / sin^m = N_lm d^m P_l / dx^m (no Condon-Shortley phase)."""
    if m > l:
        return mp.mpf(0)
    return N(l, m) * mp.diff(lambda t: mp.legendre(l, t), x, m)


def G(l, m, x):
    return -mp.diff(lambda t: Q(l, m, t), x)


def Pbar(l, m, th):
    return mp.sin(th) ** m * Q(l, m, mp.cos(th))


def k(l, m):
    return mp.sqrt(l * (l + 1) / mp.mpf(2)) if m == 0 else mp.sqrt((l - m) * (l + m + 1))


def E(l, m, x):
    return m * x * Q(l, m, x) - (1 - x * x) * k(l, m) * Q(l, m + 1, x)


def A_def(l, m, th):
    P = lambda t: Pbar(l, m, t)
    s = mp.sin(th)
    return mp.diff(P, th, 2) - mp.cos(th) / s * mp.diff(P, th) + m * m * P(th) / s ** 2


def B_def(l, m, th):
    P = lambda t: Pbar(l, m, t)
    s = mp.sin(th)
    return (mp.diff(P, th) - mp.cos(th) / s * P(th)) / s


def A_orders(l, m, th):
    """Per-order forms (documentation)."""
    x, s = mp.cos(th), mp.sin(th)
    if m == 0:
        return 2 * k(l, 0) * x * Q(l, 1, x) - l * (l + 1) * Q(l, 0, x)
    if m == 1:
        return s * ((2 - l * (l + 1)) * Q(l, 1, x) + 2 * k(l, 1) * x * Q(l, 2, x))
    return 2 * s ** (m - 2) * (m * m * Q(l, m, x) - x * E(l, m, x)) - l * (l + 1) * s ** m * Q(l, m, x)


def B_orders(l, m, th):
    x, s = mp.cos(th), mp.sin(th)
    if m == 1:
        return -k(l, 1) * s * Q(l, 2, x)
    return s ** (m - 2) * (E(l, m, x) - x * Q(l, m, x))


def A_unified(l, m, th):
    """Form used by the library: 2m(m-1) s^(m-2) Q + s^m [(2m - l(l+1)) Q - 2 x G]."""
    x, s = mp.cos(th), mp.sin(th)
    t = 2 * m * (m - 1) * s ** (m - 2) * Q(l, m, x) if m >= 2 else 0
    return t + s ** m * ((2 * m - l * (l + 1)) * Q(l, m, x) - 2 * x * G(l, m, x))


def B_unified(l, m, th):
    """(m-1) x s^(m-2) Q + s^m G, m >= 1."""
    x, s = mp.cos(th), mp.sin(th)
    t = (m - 1) * x * s ** (m - 2) * Q(l, m, x) if m >= 2 else 0
    return t + s ** m * G(l, m, x)


def check(name, err, tol):
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}: {name} (max rel err {mp.nstr(err, 3)})")
    return ok


def main():
    ok = True
    thetas = (mp.mpf("1e-6"), mp.mpf("0.3"), mp.mpf("1.2"), mp.mpf("2.9"), mp.pi - mp.mpf("1e-5"))

    # 1, 2
    eE = eA = eB = eAu = eBu = mp.mpf(0)
    for l in range(0, 9):
        for m in range(0, l + 1):
            for th in thetas:
                x, s = mp.cos(th), mp.sin(th)
                if m >= 1:
                    e = mp.diff(lambda t: Pbar(l, m, t), th) / s ** (m - 1)
                    eE = max(eE, abs(e - E(l, m, x)) / (1 + abs(e)))
                a = A_def(l, m, th)
                eA = max(eA, abs(A_orders(l, m, th) - a) / (1 + abs(a)))
                eAu = max(eAu, abs(A_unified(l, m, th) - a) / (1 + abs(a)))
                if m >= 1:
                    b = B_def(l, m, th)
                    eB = max(eB, abs(B_orders(l, m, th) - b) / (1 + abs(b)))
                    eBu = max(eBu, abs(B_unified(l, m, th) - b) / (1 + abs(b)))
    ok &= check("E = m x Q - (1-x^2) dQ/dx", eE, 1e-40)
    ok &= check("A, per-order forms", eA, 1e-40)
    ok &= check("B, per-order forms", eB, 1e-40)
    ok &= check("A, unified form with G", eAu, 1e-40)
    ok &= check("B, unified form with G", eBu, 1e-40)

    # 3
    def a_(l, m):
        return mp.sqrt(mp.mpf((2 * l + 1) * (2 * l - 1)) / ((l + m) * (l - m)))

    def b_(l, m):
        return -mp.sqrt(mp.mpf((l + m - 1) * (l - m - 1) * (2 * l + 1)) / ((l + m) * (l - m) * (2 * l - 3)))

    x = mp.mpf("0.3")
    eG = mp.mpf(0)
    for m in range(0, 5):
        q2, q1, g2, g1 = 0, Q(m, m, x), 0, 0
        for l in range(m + 1, 12):
            a = mp.sqrt(2 * m + 3) if l == m + 1 else a_(l, m)
            b = 0 if l == m + 1 else b_(l, m)
            q, g = a * x * q1 + b * q2, a * (x * g1 - q1) + b * g2
            q2, q1, g2, g1 = q1, q, g1, g
            eG = max(eG, abs(g - G(l, m, x)) / (1 + abs(G(l, m, x))))
    ok &= check("G recursion", eG, 1e-40)

    # 4
    ep = mp.mpf(0)
    for l in range(2, 7):
        ep = max(ep, abs(A_unified(l, 0, mp.mpf(0))))
        ep = max(ep, abs(A_unified(l, 2, mp.mpf(0)) - A_def(l, 2, mp.mpf("1e-15"))) / abs(A_unified(l, 2, mp.mpf(0))))
        ep = max(ep, abs(B_unified(l, 2, mp.mpf(0)) - B_def(l, 2, mp.mpf("1e-15"))) / abs(B_unified(l, 2, mp.mpf(0))))
    ok &= check("pole values (A(m=0) = 0, m = 2 limits)", ep, 1e-12)

    # 5
    mp.mp.dps = 40
    eT = mp.mpf(0)
    for (l, m, r, th, ph) in ((3, 0, 1.3, 0.4, 1.0), (4, 1, 1.1, 2.2, 3.0), (5, 2, 1.2, 0.05, 0.3),
                              (6, 3, 1.05, 1.3, 5.0), (6, 1, 1.4, 3.1, 2.0)):
        ph0 = mp.mpf("0.37")
        r, th, ph = mp.mpf(r), mp.mpf(th), mp.mpf(ph)

        def V(X, Y, Z):
            rho = mp.sqrt(X * X + Y * Y)
            rr = mp.sqrt(rho * rho + Z * Z)
            return (1 / rr) ** (l + 1) * Pbar(l, m, mp.atan2(rho, Z)) * mp.cos(m * mp.atan2(Y, X) + ph0)

        p = [r * mp.sin(th) * mp.cos(ph), r * mp.sin(th) * mp.sin(ph), r * mp.cos(th)]
        H = mp.matrix(3, 3)
        for i in range(3):
            for j in range(3):
                o = [0, 0, 0]
                o[i] += 1
                o[j] += 1
                H[i, j] = mp.diff(V, p, tuple(o))
        st, ct, sp, cp = mp.sin(th), mp.cos(th), mp.sin(ph), mp.cos(ph)
        R = mp.matrix([[st * cp, st * sp, ct], [ct * cp, ct * sp, -st], [-sp, cp, 0]])
        T = R * H * R.T

        u, Phi, dPhi = 1 / r, mp.cos(m * ph + ph0), -m * mp.sin(m * ph + ph0)
        P = Pbar(l, m, th)
        dP = mp.diff(lambda t: Pbar(l, m, t), th)
        PoS = st ** (m - 1) * Q(l, m, ct) if m >= 1 else 0
        f = u ** (l + 3)
        Trr = f * (l + 1) * (l + 2) * P * Phi
        D = f * A_unified(l, m, th) * Phi
        ref = {(0, 0): Trr, (0, 1): -f * (l + 2) * dP * Phi, (0, 2): -f * (l + 2) * PoS * dPhi,
               (1, 1): (-Trr + D) / 2, (1, 2): f * (B_unified(l, m, th) if m >= 1 else 0) * dPhi,
               (2, 2): (-Trr - D) / 2}
        scale = max(abs(T[i, j]) for i in range(3) for j in range(3))
        for (i, j), v in ref.items():
            eT = max(eT, abs(T[i, j] - v) / scale, abs(T[j, i] - v) / scale)
    ok &= check("component table vs rotated Cartesian Hessian", eT, 1e-30)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
