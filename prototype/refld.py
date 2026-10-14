"""Extended-precision (x87 long double, ~19 digits) reference synthesis, vectorised over points."""
import numpy as np
LD = np.longdouble

def reference_field(g, h, a, r, theta, phi):
    L = g.shape[0] - 1
    r = np.asarray(r, LD); th = np.asarray(theta, LD); ph = np.asarray(phi, LD)
    x, s = np.cos(th), np.sin(th)
    u = LD(a) / r
    M = r.shape[0]
    Br = np.zeros(M, LD); Bt = np.zeros(M, LD); Bp = np.zeros(M, LD)
    un = [np.ones(M, LD)]
    for k in range(1, L + 3):
        un.append(un[-1] * u)
    Pmm = np.ones(M, LD); dPmm = np.zeros(M, LD); Qmm = np.zeros(M, LD)
    for m in range(L + 1):
        if m == 1:
            Pmm, dPmm, Qmm = s.copy(), x.copy(), np.ones(M, LD)
        elif m >= 2:
            f = np.sqrt(LD(2 * m - 1) / LD(2 * m))
            dPmm = f * (s * dPmm + x * Pmm); Pmm = f * s * Pmm; Qmm = f * s * Qmm
        cm, sm = np.cos(m * ph), np.sin(m * ph)
        P2 = P1 = None
        Pn, dPn, Qn = Pmm, dPmm, Qmm
        Pm1 = dPm1 = Qm1 = None
        for n in range(m, L + 1):
            if n > m:
                aa = LD(2 * n - 1) / np.sqrt(LD(n * n - m * m))
                if n >= m + 2:
                    bb = np.sqrt(LD((n - 1) ** 2 - m * m) / LD(n * n - m * m))
                    newP = aa * x * Pn - bb * Pm1
                    newdP = aa * (x * dPn - s * Pn) - bb * dPm1
                    newQ = aa * x * Qn - bb * Qm1
                else:
                    newP = aa * x * Pn; newdP = aa * (x * dPn - s * Pn); newQ = aa * x * Qn
                Pm1, dPm1, Qm1 = Pn, dPn, Qn
                Pn, dPn, Qn = newP, newdP, newQ
            if n >= 1:
                gc = LD(g[n, m]) * cm + LD(h[n, m]) * sm
                w = un[n + 2]
                Br += (n + 1) * w * gc * Pn
                Bt -= w * gc * dPn
                if m > 0:
                    Bp += m * w * (LD(g[n, m]) * sm - LD(h[n, m]) * cm) * Qn
    return Br, Bt, Bp
