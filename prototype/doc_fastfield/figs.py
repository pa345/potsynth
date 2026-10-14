import sys, json, numpy as np, matplotlib
sys.path.insert(0, '/home/claude/fastpot')
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fastfield import _sectoral, _legendre_column, window_ft, _cheb_coeffs
from models import model, a

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
    "font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})
def save(fig, name):
    fig.tight_layout(); fig.savefig(f"fig/{name}.pdf"); plt.close(fig)
P = '/home/claude/fastpot/'

# 1. doubled sphere: Legendre functions extended to theta in [0, 2pi]
th = np.linspace(0, 2*np.pi, 1201); x, s = np.cos(th), np.sin(th)
L = 6; m = 2; n = 5
Pmm, dPmm, Qmm = _sectoral(L, x, s)
Pc, dPc, Qc = _legendre_column(m, L, x, s, Pmm, dPmm, Qmm)
fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.axvspan(0, np.pi, color=GRID, alpha=0.7, lw=0)
ax.plot(th, Pc[n-m], color=C1, label=r"$P_5^2(\cos\theta)$")
ax.plot(th, dPc[n-m]/5, color=C2, label=r"$\frac{1}{5}\,\mathrm{d}P_5^2/\mathrm{d}\theta$")
ax.plot(th, Qc[n-m], color=C3, label=r"$P_5^2(\cos\theta)/\sin\theta$")
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]); ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_ylim(-1.7, 2.2)
ax.text(np.pi/2, 1.95, "physical colatitudes", ha="center", fontsize=8, color=INK2)
ax.text(3*np.pi/2, 1.95, "mirror half (doubled sphere)", ha="center", fontsize=8, color=INK2)
ax.set_xlabel(r"$\theta$"); ax.legend(loc="lower left", fontsize=8, ncol=3)
save(fig, "doubled")

# 2. singular values of the weighted radial functions, L = 185
g, h = model(185); C = g - 1j*h; L = 185; nn = np.arange(1, L+1)
wn = (nn+1)*np.sqrt(np.sum(np.abs(C[1:])**2, axis=1))
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for (rmin, rmax, lab, col) in ((a+250, a+800, "250–800 km (satellite shell)", C1), (a, a+3000, "0–3000 km", C2), (a, 2*a, r"$a$ to $2a$", C3)):
    u0, u1 = a/rmax, a/rmin; uc, ud = (u0+u1)/2, (u1-u0)/2
    K = 186; t = np.cos(np.pi*np.arange(K+1)/K)
    A = (uc+ud*t)[:, None]**(nn[None, :]+2)*wn[None, :]
    cw = np.ones(K+1); cw[0] = cw[-1] = 0.5
    S = np.linalg.svd(np.sqrt(cw)[:, None]*A, compute_uv=False); S /= S[0]
    k = np.arange(1, 41)
    ax.semilogy(k, S[:40], marker="o", ms=3, color=col, label=lab)
for e in (1e-6, 1e-10, 1e-13):
    ax.axhline(e, color=INK2, lw=0.7, ls=":")
    ax.text(37, e*3, f"$10^{{{int(np.log10(e))}}}$", fontsize=7.5, va="bottom", color=INK2)
ax.set_ylim(1e-16, 2); ax.set_xlabel(r"index $i$"); ax.set_ylabel(r"$s_i/s_1$"); ax.legend(fontsize=7.5)
save(fig, "singular")

# 3. window and its Fourier transform (w = 7, sigma = 2)
w, sigma = 7, 2.0; beta = np.pi*(1-1/(2*sigma))*w
nf = 200; hgrid = 2*np.pi/nf; N = nf/(2*sigma)
z = np.linspace(-1, 1, 400)
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6))
axs[0].plot(z*w/2, np.exp(beta*(np.sqrt(1-z*z)-1)), color=C1)
axs[0].plot(np.arange(-3, 4), np.exp(beta*(np.sqrt(1-(np.arange(-3, 4)/3.5)**2)-1)), "o", color=C1, ms=4)
axs[0].set_xlabel(r"distance from point, in grid steps $h$"); axs[0].set_title(r"window $\phi$ ($w=7$)", fontsize=9)
k = np.linspace(0, nf*1.3, 1200)
ft = np.abs(window_ft(k, w, beta, hgrid)); ft /= ft[0]
axs[1].semilogy(k/nf, np.maximum(ft, 1e-18), color=C1)
axs[1].axvspan(0, N/nf, color=C3, alpha=0.18, lw=0)
axs[1].axvspan((nf-N)/nf, 1.3, color=C2, alpha=0.15, lw=0)
axs[1].text(N/nf/2, 1e-15, "band\n$|k|\\leq N$", ha="center", fontsize=7.5, color=INK)
axs[1].text((nf-N)/nf+0.12, 1e-2, "nearest\naliases", ha="center", fontsize=7.5, color=INK)
axs[1].set_ylim(1e-18, 3); axs[1].set_xlabel(r"frequency $k / n_f$"); axs[1].set_title(r"$|\hat\phi(k)|/\hat\phi(0)$", fontsize=9)
save(fig, "window")

# 4. 1-D pre-sharpening example
rng = np.random.default_rng(3); N = 10
kk = np.arange(-N, N+1); ck = (rng.normal(size=2*N+1) + 1j*rng.normal(size=2*N+1))/np.sqrt(2*N+1)
ck = 0.5*(ck + np.conj(ck[::-1]))                     # real-valued function
f = lambda x: np.real(np.exp(1j*np.outer(x, kk)) @ ck)
nf = 2*(2*N+1); hh = 2*np.pi/nf; xl = np.arange(nf)*hh
xs = np.linspace(0, 2*np.pi, 1500)
def grid_vals(w):
    bt = np.pi*(1-1/4)*w
    cp = ck*hh/window_ft(kk, w, bt, hh)
    return np.real(np.exp(1j*np.outer(xl, kk)) @ cp), bt
def recon(x, w):
    b, bt = grid_vals(w); out = np.zeros_like(x)
    for j, xx in enumerate(x):
        l0 = int(np.ceil(xx/hh - w/2))
        for q in range(w):
            l = l0+q; zz = (xx - l*hh)/(w*hh/2)
            out[j] += b[l % nf]*np.exp(bt*(np.sqrt(max(0, 1-zz*zz))-1))
    return out
fig, axs = plt.subplots(2, 1, figsize=(6.2, 4.2), sharex=True)
b7, _ = grid_vals(7)
axs[0].plot(xs, f(xs), color=INK, lw=1.2, label=r"$f(x)$")
axs[0].plot(xl, f(xl), "o", ms=4, mfc="white", color=C1, label=r"plain samples $f(x_l)$")
axs[0].plot(xl, b7, "s", ms=3.5, color=C2, label=r"pre-sharpened values $b_l$ ($w=7$)")
axs[0].legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22)); axs[0].set_ylabel("value")
for w, col in ((4, C1), (7, C2), (10, C3)):
    axs[1].semilogy(xs, np.maximum(np.abs(recon(xs, w) - f(xs)), 1e-17), color=col, lw=1, label=f"$w={w}$")
axs[1].set_ylim(1e-14, 1e-1); axs[1].set_xlabel(r"$x$"); axs[1].set_ylabel("reconstruction error"); axs[1].legend(fontsize=7.5, ncol=3)
axs[1].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]); axs[1].set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
save(fig, "presharp")

# 5. stored band on the doubled-sphere grid
fig, ax = plt.subplots(figsize=(4.4, 3.6))
nfg = 24; hg = 2*np.pi/nfg; wv = 5
ax.set_xlim(0, 2*np.pi); ax.set_ylim(-np.pi/2.5, 2*np.pi)
for l in range(nfg+1):
    ax.axvline(l*hg, color=GRID, lw=0.5); ax.axhline(l*hg, color=GRID, lw=0.5)
ax.add_patch(Rectangle((0, -wv/2*hg), 2*np.pi, np.pi+wv*hg, color=C1, alpha=0.12, lw=0))
ax.axhline(0, color=INK2, lw=0.8); ax.axhline(np.pi, color=INK2, lw=0.8)
ax.add_patch(Rectangle((0, np.pi + wv/2*hg), 2*np.pi, np.pi - wv/2*hg, hatch="///", fill=False, color=GRID, lw=0))
for (tp, pp, col) in ((0.12, 2.0, C2), (2.6, 4.6, C3)):
    l1 = int(np.ceil(tp/hg - wv/2)); l2 = int(np.ceil(pp/hg - wv/2))
    ax.add_patch(Rectangle((l2*hg, l1*hg), (wv-1)*hg, (wv-1)*hg, fill=False, ec=col, lw=1.5))
    ax.plot(pp, tp, "o", color=col, ms=5)
ax.annotate("margin rows\n(below the pole)", xy=(4.0, -0.3), xytext=(3.4, -1.05), fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7))
ax.text(0.15, np.pi/2, "stored rows:\n" + r"$\theta\in[0,\pi]$ plus margin", fontsize=8, color=INK, va="center")
ax.text(0.15, 1.5*np.pi+0.3, "not stored", fontsize=8, color=INK2)
ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"$\theta$ (doubled sphere)")
ax.set_xticks([0, np.pi, 2*np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.set_yticks([0, np.pi, 2*np.pi]); ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])
ax.grid(False)
save(fig, "band")

# 6. calibration: error vs window width
cal = json.load(open(P + 'fastfield_calibration.json'))
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for key, col, mk, lab in (("L185_s2", C1, "o", r"$L=185$, $\sigma=2$"), ("L13_s2", C1, "x", r"$L=13$, $\sigma=2$"),
                          ("L185_s3", C2, "o", r"$L=185$, $\sigma=3$"), ("L13_s3", C2, "x", r"$L=13$, $\sigma=3$")):
    ws = sorted(int(k) for k in cal[key]); es = [cal[key][str(k)]["err"] for k in ws]
    ax.semilogy(ws, es, marker=mk, color=col, ls="-" if mk == "o" else ":", label=lab)
ax.axhline(1.6e-14, color=INK2, lw=0.8, ls="--"); ax.text(4.2, 2.4e-14, "direct sum, $L=185$", fontsize=7.5, color=INK2)
ax.set_xlabel(r"window width $w$ (grid points per direction)"); ax.set_ylabel("max relative error"); ax.legend(fontsize=7.5)
save(fig, "calib")

# 7/8. speed
R = json.load(open(P + 'bench_final.json'))
Ls = sorted({x['L'] for x in R})
gv = lambda L, e, M: next(x for x in R if x['L'] == L and x['eps'] == e and x['M'] == M)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
for e, col, mk, lab in ((1e-6, C1, "o", r"$\varepsilon=10^{-6}$"), (1e-10, C2, "s", r"$\varepsilon=10^{-10}$"), (1e-13, C3, "^", r"$\varepsilon=10^{-13}$ (machine prec.)")):
    axs[0].loglog(Ls, [gv(L, e, 10**6)['t_direct']/gv(L, e, 10**6)['t_fast'] for L in Ls], marker=mk, color=col, label=lab)
    axs[1].loglog(Ls, [gv(L, e, 10**6)['t_fast'] for L in Ls], marker=mk, color=col, label="fast, " + lab)
axs[1].loglog(Ls, [gv(L, 1e-6, 10**6)['t_direct'] for L in Ls], marker="D", color=INK, label="direct sum")
axs[0].axhline(1, color=INK2, lw=0.8)
for ax in axs:
    ax.set_xticks(Ls); ax.set_xticklabels(Ls, fontsize=7); ax.minorticks_off(); ax.set_xlabel(r"maximum degree $L$")
axs[0].set_ylabel(r"speedup over direct sum"); axs[1].set_ylabel(r"time for $10^6$ points (s)")
axs[0].legend(fontsize=7); axs[1].legend(fontsize=6.5)
save(fig, "speed")
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
Ms = [10**4, 10**5, 10**6]
for ax, L in zip(axs, (13, 185)):
    ax.loglog(Ms, [gv(L, 1e-6, M)['t_direct'] for M in Ms], marker="D", color=INK, label="direct sum")
    for e, col, mk in ((1e-6, C1, "o"), (1e-10, C2, "s"), (1e-13, C3, "^")):
        ax.loglog(Ms, [gv(L, e, M)['t_fast'] + gv(L, e, M)['t_pre'] for M in Ms], marker=mk, color=col,
                  label=f"fast incl. setup, $\\varepsilon=10^{{{int(np.log10(e))}}}$")
    ax.set_title(f"$L={L}$", fontsize=9); ax.set_xlabel(r"number of points $M$"); ax.set_ylabel("wall time (s)")
axs[0].legend(fontsize=6.5)
save(fig, "timeM")
print("ok")
