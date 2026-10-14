"""
Tests of the Python interface (python/potsynth) against the reference data.

Usage: POTSYNTH_LIBRARY=/path/to/libpotsynth.so python test_python.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "python"))
import potsynth  # noqa: E402

nfail = 0


def check(cond, msg):
    global nfail
    if not cond:
        print("FAIL:", msg)
        nfail += 1


def read_ref(path):
    with open(path, "rb") as f:
        L, n = np.fromfile(f, "<i8", 2)
        a, rmin, rmax = np.fromfile(f, "<f8", 3)
        gh = np.fromfile(f, "<f8", potsynth.nlm(1, int(L), int(L)))
        r, th, ph = (np.fromfile(f, "<f8", int(n)) for _ in range(3))
        B = np.fromfile(f, "<f8", 3 * int(n)).reshape(3, int(n))
    return int(L), a, rmin, rmax, gh, r, th, ph, B


def rel_err(B, Bref):
    return np.max(np.abs(np.asarray(B) - Bref)) / np.max(np.abs(Bref))


def test_reference():
    for name in ("ref_L13_s1.bin", "ref_L185_s3.bin"):
        L, a, rmin, rmax, gh, r, th, ph, Bref = read_ref(os.path.join(HERE, "data", name))
        for method, eps, tol in (("direct", 1e-10, 1e-13), ("fast", 1e-10, 1e-10),
                                 ("fast", 1e-14, 1e-13)):
            m = potsynth.MagModel(gh, lmax=L, a=a, rmin=rmin, rmax=rmax, eps=eps, method=method)
            err = rel_err(np.array(m(r, th, ph)), Bref)
            check(err <= tol, f"{name} {method} eps={eps}: {err:.2e}")
            print(f"  {name} {method:6s} eps={eps:.0e} err {err:.2e}")


def test_api():
    L, a, rmin, rmax, gh, r, th, ph, Bref = read_ref(os.path.join(HERE, "data", "ref_L60_s2.bin"))
    m = potsynth.MagModel(gh, a=a, rmin=rmin, rmax=rmax, method="fast")
    check(m.info["lmax"] == L and m.info["method"] == "fast", f"info {m.info}")

    # (g, h) arrays give the same model as the vector
    g = np.zeros((L + 1, L + 1))
    h = np.zeros((L + 1, L + 1))
    for n in range(1, L + 1):
        for k in range(n + 1):
            g[n, k] = gh[potsynth.idx(n, k, 1, L)]
            if k:
                h[n, k] = gh[potsynth.idx(n, -k, 1, L)]
    m2 = potsynth.MagModel((g, h), a=a, rmin=rmin, rmax=rmax, method="fast")
    check(np.array_equal(np.array(m(r, th, ph)), np.array(m2(r, th, ph))), "(g, h) input differs")

    # broadcasting and shapes
    rr = np.full((4, 5), 0.5 * (rmin + rmax))
    Br, Bt, Bp = m(rr, np.linspace(0, np.pi, 5), 1.0)
    check(Br.shape == (4, 5) and Bt.shape == (4, 5), "broadcast shape")
    Br0, _, _ = m(rmin, 0.0, 0.0)
    check(np.ndim(Br0) == 0 and np.isfinite(Br0), "scalar input")

    # NaN in, NaN out; domain errors raise
    Br, _, _ = m([rmin, np.nan], [1.0, 1.0], [0.0, 0.0])
    check(np.isfinite(Br[0]) and np.isnan(Br[1]), "NaN handling")
    try:
        m(rmax * 1.01, 1.0, 0.0)
        check(False, "r > rmax accepted")
    except potsynth.DomainError:
        pass
    try:
        potsynth.MagModel(gh[:-1], lmax=L)
        check(False, "short coefficient vector accepted")
    except ValueError:
        pass

    # NEC conversion
    N, E, C = potsynth.rtp_to_nec(1.0, 2.0, 3.0)
    check((N, E, C) == (-2.0, 3.0, -1.0), "rtp_to_nec")

    # generic core: potential V and dV/dr agree between fast and direct
    nn = np.arange(L + 1, dtype=float)
    chans = [potsynth.Channel(rpow=1, theta="P"), potsynth.Channel(rpow=2, theta="P", rfac=nn + 1)]
    coef = potsynth.coef_vector(g / np.sqrt(2 * nn + 1)[:, None], h / np.sqrt(2 * nn + 1)[:, None])
    vf = potsynth.Model(coef, 1, L, chans, a, rmin, rmax, method="fast", eps=1e-12)(r, th, ph)
    vd = potsynth.Model(coef, 1, L, chans, a, method="direct")(r, th, ph)
    check(rel_err(vf, vd) < 1e-11, f"core fast vs direct {rel_err(vf, vd):.2e}")
    # dV/dr channel equals B_r of the magnetic model
    check(rel_err(vd[1], Bref[0]) < 1e-13, f"core dV/dr vs B_r {rel_err(vd[1], Bref[0]):.2e}")
    print("  API checks done")


if __name__ == "__main__":
    test_reference()
    test_api()
    print("FAILED" if nfail else "all passed")
    sys.exit(1 if nfail else 0)
