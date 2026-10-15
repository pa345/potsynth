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
    for l in range(1, L + 1):
        for k in range(l + 1):
            g[l, k] = gh[potsynth.idx(l, k, 1, L)]
            if k:
                h[l, k] = gh[potsynth.idx(l, -k, 1, L)]
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
    # SHC reader: file -> model equals vector -> model; time-dependent files rejected
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.shc")
        with open(f, "w") as fp:
            fp.write("# test\n 1 2 1 1 1\n 2015.0\n")
            for ll in range(1, 3):
                for mm in range(-ll, ll + 1):
                    fp.write(f"{ll} {mm} {0.5 * ll + 0.1 * mm}\n")
        shc = potsynth.read_shc(f)
        check(shc.lmin == 1 and shc.lmax == 2 and shc.epoch == 2015.0, "read_shc header")
        check(shc.gh[potsynth.idx(2, -1, 1, 2)] == 0.9, "read_shc coefficient h21")
        m1 = potsynth.MagModel.from_shc(f, method="direct")
        m2 = potsynth.MagModel(shc.gh, lmax=2, method="direct")
        check(np.array_equal(np.array(m1(7000.0, 1.0, 2.0)), np.array(m2(7000.0, 1.0, 2.0))),
              "MagModel.from_shc")
        with open(f, "w") as fp:
            fp.write("1 1 2 2 1\n2015.0 2020.0\n1 0 1 1\n1 1 1 1\n1 -1 1 1\n")
        try:
            potsynth.read_shc(f)
            check(False, "time-dependent SHC accepted")
        except potsynth.PotsynthError as e:
            check(e.status == 4 and "time-dependent" in str(e), "time-dependent SHC error")

    # npts_hint: few points -> direct, many -> fast (L = 60)
    gh60 = np.ones(potsynth.nlm(1, 60, 60))
    m_few = potsynth.MagModel(gh60, rmin=rmin, rmax=rmax, npts_hint=1000)
    m_many = potsynth.MagModel(gh60, rmin=rmin, rmax=rmax, npts_hint=10_000_000)
    check(m_few.info["method"] == "direct" and m_many.info["method"] == "fast",
          f"npts_hint selection {m_few.info['method']} {m_many.info['method']}")

    print("  API checks done")


def test_grav():
    # gfc reader and gravity model: point mass + J2 against closed forms, from_gfc,
    # quantity selection, time-variable files rejected
    import tempfile
    GM, a, C20 = 3.986004415e14, 6378136.3, -0.484165143790e-03
    head = ("free text\nproduct_type gravity_field\nmodelname TEST\n"
            f"earth_gravity_constant {GM:.10e}\nradius {a:.10e}\nmax_degree 2\n"
            "tide_system zero_tide\nend_of_head\n")
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.gfc")
        with open(f, "w") as fp:
            fp.write(head + "gfc 0 0 1.0D+00 0.0\n" + f"gfc 2 0 {C20:.15e} 0.0\n")
        g = potsynth.read_gfc(f)
        check(g.lmax == 2 and g.GM == GM and g.a == a and g.modelname == "TEST"
              and g.tide_system == "zero_tide", "read_gfc header")
        check(g.coef[potsynth.idx(2, 0, 0, 2)] == C20 and g.coef[potsynth.idx(1, 0, 0, 2)] == 0.0,
              "read_gfc coefficients")

        r = np.array([6.3e6, 6.5e6, 6.9e6, 6.4e6])
        th = np.array([0.0, 0.7, 2.0, np.pi])
        ph = np.array([0.0, 1.0, 4.0, 2.0])
        x, q = np.cos(th), a / r
        P2, dP2 = np.sqrt(5) * 0.5 * (3 * x * x - 1), -np.sqrt(5) * 3 * x * np.sin(th)
        V = GM / r * (1 + q * q * C20 * P2)
        gr = -GM / r ** 2 * (1 + 3 * q * q * C20 * P2)
        gt = GM / r ** 2 * q * q * C20 * dP2
        for method in ("direct", "fast"):
            m = potsynth.GravModel.from_gfc(f, rmin=6.3e6, rmax=6.9e6, method=method)
            out = m(r, th, ph)
            tol = 1e-13 if method == "direct" else 1e-9
            check(m.quantities == ("V", "g_r", "g_theta", "g_phi"), "GravModel quantities")
            check(rel_err(out.V, V) < tol, f"GravModel V ({method})")
            check(rel_err(np.array([out.g_r, out.g_theta, out.g_phi]),
                          np.array([gr, gt, 0 * gr])) < tol, f"GravModel g ({method})")

        m = potsynth.GravModel(g.coef, GM, a, quantities=["g_r"], method="direct")
        check(m.quantities == ("g_r",) and rel_err(m(r, th, ph).g_r, gr) < 1e-13,
              "GravModel g_r only")
        try:
            potsynth.GravModel(g.coef, GM, a, quantities="W")
            check(False, "unknown quantity accepted")
        except ValueError:
            pass

        with open(f, "w") as fp:
            fp.write(head + "gfc 0 0 1.0 0.0\ngfct 2 0 -4.8e-4 0.0 0.0 0.0 20050101.0\n")
        try:
            potsynth.read_gfc(f)
            check(False, "time-variable gfc accepted")
        except potsynth.PotsynthError as e:
            check(e.status == 4, "time-variable gfc error")

    print("  gravity checks done")


def test_hessian():
    # gradient tensor channels (S2A, S2B) against the 40-digit reference values
    with open(os.path.join(HERE, "data", "hess_L24_s12.bin"), "rb") as f:
        L, n = (int(v) for v in np.fromfile(f, "<i8", 2))
        a = np.fromfile(f, "<f8", 1)[0]
        coef = np.fromfile(f, "<f8", potsynth.nlm(0, L, L))
        r, th, ph = (np.fromfile(f, "<f8", n) for _ in range(3))
        T = np.fromfile(f, "<f8", 6 * n).reshape(6, n)
    l = np.arange(L + 1.0)
    chans = [potsynth.Channel(3, "P", 0, (l + 1) * (l + 2) / a**2),
             potsynth.Channel(3, "dP", 0, -(l + 2) / a**2),
             potsynth.Channel(3, "Psin", 1, -(l + 2) / a**2),
             potsynth.Channel(3, "S2A", 0, np.full(L + 1, 1 / a**2)),
             potsynth.Channel(3, "S2B", 1, np.full(L + 1, 1 / a**2))]
    m = potsynth.Model(coef, 0, L, chans, a)
    check(m.info["method"] == "direct", "no radial range: direct method")
    Trr, Trt, Trp, D, Ttp = m(r, th, ph)
    Tm = np.array([Trr, Trt, Trp, 0.5 * (-Trr + D), Ttp, 0.5 * (-Trr - D)])
    check(rel_err(Tm, T) < 1e-13, f"Hessian channels vs reference ({rel_err(Tm, T):.2e})")
    for method, tol in (("direct", 1e-13), ("fast", 1e-11)):
        m = potsynth.Model(coef, 0, L, chans, a, rmin=1.0, rmax=1.3, method=method, eps=1e-12)
        Trr, Trt, Trp, D, Ttp = m(r, th, ph)
        Tm = np.array([Trr, Trt, Trp, 0.5 * (-Trr + D), Ttp, 0.5 * (-Trr - D)])
        check(rel_err(Tm, T) < tol, f"Hessian channels vs reference, {method} ({rel_err(Tm, T):.2e})")

    # front-ends: GravModel "T" is the reference tensor for GM = a = 1
    for method in ("direct", "fast"):
        g = potsynth.GravModel(coef, 1.0, 1.0, lmax=L, quantities="T", rmin=1.0, rmax=1.3,
                               method=method, eps=1e-12)
        out = g(r, th, ph)
        check(g.quantities == ("T_rr", "T_rt", "T_rp", "T_tt", "T_tp", "T_pp"), "GravModel T names")
        check(rel_err(np.array(out), T) < (1e-13 if method == "direct" else 1e-11),
              f"GravModel T vs reference, {method} ({rel_err(np.array(out), T):.2e})")

    # MagModel gradB = -T of the same potential (gh = C sqrt(2l+1), GM = a^2)
    am = 1.3
    cm = coef[potsynth.nlm(0, 0, 0):]
    gh = cm * np.concatenate([np.full(2 * l + 1, np.sqrt(2 * l + 1.0)) for l in range(1, L + 1)])
    mm = potsynth.MagModel(gh, lmax=L, a=am, quantities="B,gradB", method="direct")
    gm = potsynth.GravModel(cm, am * am, am, lmax=L, lmin=1, quantities="T", method="direct")
    fm, fg = mm(r * am, th, ph), gm(r * am, th, ph)
    check(mm.quantities[:3] == ("B_r", "B_theta", "B_phi") and len(mm.quantities) == 9,
          "MagModel B,gradB names")
    check(rel_err(np.array(fm[3:]), -np.array(fg)) < 1e-13, "MagModel gradB = -GravModel T")
    check(mm.info["nout"] == 9 and mm.info["nchan"] == 8, "MagModel B,gradB info nout/nchan")
    Br, Bt, Bp = potsynth.MagModel(gh, lmax=L, a=am, method="direct")(r * am, th, ph)
    check(np.array_equal(Br, fm.B_r) and np.array_equal(Bp, fm.B_phi), "MagModel B unchanged with gradB")

    # frame helpers: NEC mapping; Cartesian components at the poles independent of phi
    nec = potsynth.tensor_rtp_to_nec(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    check([float(v) for v in nec] == [4.0, -5.0, 2.0, 6.0, -3.0, 1.0], "tensor_rtp_to_nec mapping")
    phis = np.array([0.0, 1.0, 2.5, -2.0])
    for th0 in (0.0, np.pi):
        tp = potsynth.GravModel(coef, 1.0, 1.0, lmax=L, quantities="T", method="direct")(1.2, th0, phis)
        X = np.array(potsynth.tensor_rtp_to_xyz(th0, phis, *tp))
        check(np.abs(X - X[:, :1]).max() < 1e-14 * np.abs(X).max(), f"Cartesian tensor at theta={th0:g}")
    M = potsynth.tensor_matrix(*T[:, :5])
    check(M.shape == (5, 3, 3) and np.allclose(M, np.swapaxes(M, -1, -2)), "tensor_matrix shape, symmetry")
    check(np.abs(np.trace(M, axis1=-2, axis2=-1)).max() < 1e-13 * np.abs(M).max(), "reference tensor traceless")

    print("  gradient tensor checks done")


def test_geodesy():
    # helpers: published WGS84 values, copy semantics, shapes
    w = potsynth.WGS84
    check(w.a == 6378137.0 and abs(1 / w.f - 298.257223563) < 1e-9, "WGS84 constants")
    C = potsynth.normal_zonals(10)
    check(abs(C[2] / -0.484166774985e-3 - 1) < 1e-12 and abs(C[4] / 0.790303733511e-6 - 1) < 1e-11,
          "WGS84 normal zonals")
    g = potsynth.normal_gravity(np.radians([0.0, 90.0]))
    check(abs(g[0] - 9.7803253359) < 1e-10 and abs(g[1] - 9.8321849378) < 1e-10, "WGS84 normal gravity")
    r, th = potsynth.geodetic_to_geocentric(np.zeros((2, 3)), 100.0)
    check(r.shape == (2, 3) and np.allclose(r, w.a + 100.0) and np.allclose(th, np.pi / 2),
          "geodetic_to_geocentric shape and equator")
    c = np.zeros(potsynth.nlm(0, 4, 4))
    c[0] = 1.0
    cT = potsynth.subtract_normal_field(c, w.GM, w.a)
    check(c[0] == 1.0 and abs(cT[0]) < 1e-16 and abs(cT[potsynth.idx(2, 0, 0, 4)] - (-C[2])) < 1e-18,
          "subtract_normal_field returns a modified copy")

    # recipe signs against the definitions (finite differences of T in latitude/longitude)
    rng = np.random.default_rng(5)
    L = 30
    coef = rng.normal(0, 1e-6, potsynth.nlm(0, L, L)) / 10
    m = potsynth.GravModel(coef, w.GM, w.a, quantities="V,g", method="direct")
    lat, lon, rr = 0.6, 1.1, 6.4e6
    th, h = np.pi / 2 - lat, 1e-6
    T, gr, gt, gp = (float(v) for v in m(rr, th, lon))
    dTdlat = (m(rr, th - h, lon).V - m(rr, th + h, lon).V) / (2 * h)
    dTdlon = (m(rr, th, lon + h).V - m(rr, th, lon - h).V) / (2 * h)
    gam = 9.8
    xi_def, eta_def = -dTdlat / (rr * gam), -dTdlon / (rr * gam * np.cos(lat))
    check(abs(gt / gam - xi_def) < 1e-8 * abs(xi_def), "recipe: xi = g_theta / gamma")
    check(abs(-gp / gam - eta_def) < 1e-8 * abs(eta_def), "recipe: eta = -g_phi / gamma")
    dTdr = (m(rr + 1.0, th, lon).V - m(rr - 1.0, th, lon).V) / 2.0
    check(abs(-gr - (-dTdr)) < 1e-7 * abs(dTdr), "recipe: delta g = -g_r")

    print("  geodesy checks done")


if __name__ == "__main__":
    test_reference()
    test_api()
    test_grav()
    test_hessian()
    test_geodesy()
    print("FAILED" if nfail else "all passed")
    sys.exit(1 if nfail else 0)
