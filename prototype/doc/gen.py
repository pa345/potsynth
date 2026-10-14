import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = json.load(open('/home/claude/fastpot/bench_results.json'))
E = json.load(open('/home/claude/fastpot/experiments.json'))
BD = json.load(open('/home/claude/fastpot/breakdown.json'))
C1,C2,C3,C4 = "#2a78d6","#eb6834","#1baf7a","#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,"xtick.color":INK2,"ytick.color":INK2,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
    "lines.linewidth":1.6,"lines.markersize":4.5,"legend.frameon":False,"font.family":"serif","mathtext.fontset":"cm","pdf.fonttype":42})
def save(fig,name): fig.tight_layout(); fig.savefig(f"fig/{name}.pdf"); plt.close(fig)
cfgs = [("double 1e-10",C1,"o","double, $\\varepsilon=10^{-10}$"),("double 1e-6",C2,"s","double, $\\varepsilon=10^{-6}$"),("single 1e-6",C3,"^","single, $\\varepsilon=10^{-6}$")]
Ls = sorted({b['L'] for b in B})
get = lambda L,cfg,M: next(b for b in B if b['L']==L and b['cfg']==cfg and b['M']==M)

# accuracy
fig,ax=plt.subplots(figsize=(3.5,3.0))
for L,c,mk in (("13",C1,"o"),("185",C2,"s")):
    rows=E['accuracy'][L]; ax.loglog([r['eps'] for r in rows],[r['rel'] for r in rows],marker=mk,color=c,label=f"$L={L}$")
e=np.array([1e-13,1e-3]); ax.loglog(e,e,"--",color=INK2,lw=1,label="error $=\\varepsilon$")
ax.set_xlabel("requested tolerance $\\varepsilon$"); ax.set_ylabel("max relative error"); ax.legend(); save(fig,"accuracy")

# K vs thickness
K=E['K_vs_thickness']; alt=np.array(K['alt_km'])
fig,ax=plt.subplots(figsize=(3.5,3.0))
for L,c in (("13",C1),("45",C2),("185",C3)):
    ax.semilogx(alt,K['K'][f"{L}_1e-06"],marker="o",color=c,label=f"$L={L}$, $10^{{-6}}$")
    ax.semilogx(alt,K['K'][f"{L}_1e-10"],marker="o",ls="--",mfc="white",color=c,label=f"$L={L}$, $10^{{-10}}$")
ax.axvspan(250,800,color=GRID,alpha=0.6,lw=0)
ax.set_xlabel("shell thickness $r_{\\max}-a$ (km)"); ax.set_ylabel("radial degree $K$"); ax.legend(fontsize=7.5,ncol=1,handlelength=2.2); save(fig,"K_thickness")

# Chebyshev decay
fig,ax=plt.subplots(figsize=(4.6,2.6))
for (q,c) in zip(("15","50","100","187"),(C1,C2,C3,C4)):
    v=np.array(E['cheb_decay'][q])[:40]; v=np.where(v>0,v,np.nan)
    ax.semilogy(np.arange(len(v)),v,marker=".",color=c,label=f"$q={q}$")
ax.set_ylim(1e-17,2); ax.set_xlabel("Chebyshev index $k$"); ax.set_ylabel("$|\\alpha_k^{(q)}|/u_1^q$"); ax.legend(); save(fig,"cheb_decay")

# speedup vs L
fig,ax=plt.subplots(figsize=(3.5,3.0))
for cfg,c,mk,lab in cfgs:
    ax.loglog(Ls,[get(L,cfg,2000000)['speedup'] for L in Ls],marker=mk,color=c,label=lab)
ax.axhline(1,color=INK2,lw=0.8); ax.set_xlabel("maximum degree $L$"); ax.set_ylabel("speedup over direct"); ax.legend(fontsize=7.5)
ax.set_xticks(Ls); ax.set_xticklabels(Ls,fontsize=7); ax.minorticks_off(); save(fig,"speedup_L")

# per point vs L
fig,ax=plt.subplots(figsize=(3.5,3.0))
ax.loglog(Ls,[get(L,"double 1e-6",2000000)['t_direct']/2e6*1e6 for L in Ls],marker="D",color=INK,label="direct")
for cfg,c,mk,lab in cfgs:
    ax.loglog(Ls,[get(L,cfg,2000000)['t_fast']/2e6*1e6 for L in Ls],marker=mk,color=c,label="fast, "+lab.split(", ")[0]+", "+lab.split(", ")[1])
ax.set_xlabel("maximum degree $L$"); ax.set_ylabel("time per point ($\\mu$s, 2 cores)"); ax.legend(fontsize=7.5)
ax.set_xticks(Ls); ax.set_xticklabels(Ls,fontsize=7); ax.minorticks_off(); save(fig,"perpoint_L")

# time vs M
for L in (13,185):
    fig,ax=plt.subplots(figsize=(3.5,3.0)); Ms=[10**4,10**5,10**6,2*10**6]
    ax.loglog(Ms,[get(L,"double 1e-6",M)['t_direct'] for M in Ms],marker="D",color=INK,label="direct")
    for cfg,c,mk,lab in cfgs:
        ax.loglog(Ms,[get(L,cfg,M)['t_fast']+get(L,cfg,M)['t_pre'] for M in Ms],marker=mk,color=c,label="fast (incl. pre), "+lab.split(", ")[1].replace("$\\varepsilon=",'$\\varepsilon\\!=\\!') + (" sp" if cfg.startswith("single") else ""))
    ax.set_xlabel("number of points $M$"); ax.set_ylabel("wall time (s)"); ax.set_title(f"$L={L}$",fontsize=9); ax.legend(fontsize=7.5); save(fig,f"time_M_{L}")

# tables
def f(x,d=1): return f"{x:.{d}f}"
rows=[]
for L in Ls:
    d=get(L,"double 1e-6",1000000)['t_direct']
    r=[str(L), str(get(L,"double 1e-6",10**6)['K']), str(get(L,"double 1e-10",10**6)['K']), f"{get(L,'double 1e-10',10**6)['t_pre']:.3f}", f(d,2)]
    r+= [f(get(L,c,10**6)['t_fast'],2) for c,_,_,_ in cfgs]
    r+= [f(get(L,c,2*10**6)['speedup'],1) for c,_,_,_ in cfgs]
    r+= [f"{get(L,c,2*10**6)['maxerr_nT']:.0e}".replace("e-0","e-") for c,_,_,_ in cfgs]
    rows.append(" & ".join(r)+r" \\")
tab=r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.6pt}
\caption{Benchmark summary (250--800\,km shell, 2 cores). $K$: radial degree chosen automatically. Times for $M=10^6$
points in seconds (fast: evaluation only; precompute listed separately). Speedup is the ratio of evaluation times at
$M=2\times10^6$. The error is the maximum $|\Delta B|$ over the three components versus direct synthesis, in nT
($|\mathbf B|\lesssim 6\times10^4$\,nT). Columns: d10 = double, $\varepsilon=10^{-10}$; d6 = double, $10^{-6}$; s6 = single, $10^{-6}$.}
\label{tab:speedup}
\begin{tabular}{rrrrr@{\hspace{8pt}}rrr@{\hspace{8pt}}rrr@{\hspace{8pt}}rrr}\toprule
&\multicolumn{2}{c}{$K$}&pre&direct&\multicolumn{3}{c}{fast (s)}&\multicolumn{3}{c}{speedup}&\multicolumn{3}{c}{error (nT)}\\
\cmidrule(lr){2-3}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}
$L$&$10^{-6}$&$10^{-10}$&(s)&(s)&d10&d6&s6&d10&d6&s6&d10&d6&s6\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
"""
open("tab_speedup.tex","w").write(tab)
s=next(b for b in B if b['L']==185 and b['cfg']=="double 1e-6" and b['M']==2000000)
s10=next(b for b in B if b['L']==185 and b['cfg']=="double 1e-10" and b['M']==2000000)
ss=next(b for b in B if b['L']==185 and b['cfg']=="single 1e-6" and b['M']==2000000)
rows=[]
for k,v in BD.items():
    rows.append(f"{v['L']} & $10^{{{int(round(np.log10(v['eps'])))}}}$ & {v['w']} & {v['K']} & ${v['modes'][0]}\\times{v['modes'][1]}\\times{v['modes'][2]}$ & {v['setpts']:.3f} & {v['fft_part']:.3f} & {v['interp']:.3f} & {v['interp']/1e6*1e6:.2f}\\\\")
open("tab_breakdown.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Breakdown of one evaluation for $M=10^6$ points (3 components, $\sigma=2$, 2 cores). FFT = deconvolution plus FFT
(measured with $10$ points); interpolation = the remainder of the NUFFT time. The interpolation cost per point depends on
$w$ only, not on $L$.}\label{tab:breakdown}
\begin{tabular}{rrrrcrrrr}\toprule
$L$&$\varepsilon$&$w$&$K$&modes $(\psi,\theta,\varphi)$&sort (s)&FFT (s)&interp.\ (s)&$\mu$s/point\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
up=[(45,"1e-6",0.769,0.744,1.98),(45,"1e-10",1.751,1.65,5.118),(100,"1e-6",2.027,0.949,2.018),(100,"1e-10",1.889,1.941,5.073),(185,"1e-6",2.135,1.435,2.099),(185,"1e-10",2.866,2.754,5.115)]
rows=[f"{L} & $10^{{{e[2:]}}}$ & {a:.2f} & {b:.2f} & {c:.2f}\\\\" for L,e,a,b,c in up]
open("tab_upsamp.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Effect of the NUFFT oversampling factor on evaluation time ($M=10^6$, seconds). ``auto'' is FINUFFT's heuristic.
$\sigma=1.25$ at $\varepsilon=10^{-10}$ needs a kernel wider than FINUFFT's maximum of 16 points.}\label{tab:upsamp}
\begin{tabular}{rrrrr}\toprule $L$&$\varepsilon$&auto&$\sigma=2$&$\sigma=1.25$\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
open("macros.tex","w").write("\\newcommand\\SPEEDUPDOUBLE{$%d\\times$}\n\\newcommand\\SPEEDUPDOUBLETEN{$%d\\times$}\n\\newcommand\\SPEEDUPSINGLE{$%d\\times$}\n"%(round(s['speedup']),round(s10['speedup']),round(ss['speedup'])))
print(open("macros.tex").read())

# ---------------- 2-D variant -----------------
fs = lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"
F2 = json.load(open('/home/claude/fastpot/bench_fused.json'))
B2 = json.load(open('/home/claude/fastpot/bench2d.json'))
g2 = lambda L,eps,M: next(b for b in F2 if b['L']==L and b['eps']==eps and b['M']==M)
gb = lambda L,eps,M: next(b for b in B2 if b['L']==L and b['eps']==eps and b['M']==M and b['prec']=="double")
L2 = sorted({b['L'] for b in F2})
fig,ax=plt.subplots(figsize=(5.2,3.6))
for eps,c in ((1e-6,C1),(1e-10,C2)):
    e = "10^{-6}" if eps==1e-6 else "10^{-10}"
    ax.loglog(L2,[g2(L,eps,2*10**6)['t3']/g2(L,eps,2*10**6)['t2'] for L in L2],marker="o",color=c,label=f"$\\varepsilon={e}$, $M=2\\times10^6$")
    ax.loglog(L2,[g2(L,eps,10**5)['t3']/g2(L,eps,10**5)['t2'] for L in L2],marker="o",ls="--",mfc="white",color=c,label=f"$\\varepsilon={e}$, $M=10^5$")
    ax.loglog(L2,[gb(L,eps,2*10**6)['t3']/gb(L,eps,2*10**6)['t2'] for L in L2],marker="x",ls=":",color=c,label=f"$\\varepsilon={e}$, unfused FINUFFT 2-D, $M=2\\times10^6$")
ax.axhline(1,color=INK2,lw=0.8)
ax.set_xticks(L2); ax.set_xticklabels(L2); ax.minorticks_off()
ax.set_yticks([0.5,1,2,4,8]); ax.set_yticklabels(["0.5","1","2","4","8"])
ax.set_xlabel("maximum degree $L$"); ax.set_ylabel("time(3-D) / time(2-D)"); ax.legend(fontsize=7.5,loc="upper left")
save(fig,"twod_ratio")
rows=[]
for L in L2:
    for eps in (1e-6,1e-10):
        a,b_,c=g2(L,eps,10**5),g2(L,eps,10**6),g2(L,eps,2*10**6); bb=gb(L,eps,2*10**6)
        e = "$10^{-6}$" if eps==1e-6 else "$10^{-10}$"
        rows.append(f"{L} & {e} & {c['K']+1} & {c['rho']} & {a['t3']:.3f} & {a['t2']:.3f} & {c['t3']:.2f} & {c['t2']:.2f} & {bb['t2']:.2f} & {c['t3']/c['t2']:.2f} & {fs(c['direct']/c['t3'])} & {fs(c['direct']/c['t2'])}\\\\")
open("tab_2d.tex","w").write(r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Fused 2-D variant versus the 3-D NUFFT method (FINUFFT), 250--800\,km shell, 2 cores, double precision.
$K+1$: number of Chebyshev radial terms; $\rho$: SVD radial rank (cutoff $0.3\varepsilon$). Times in seconds, evaluation only.
``FINUFFT 2-D'' is the same 2-D split executed as $3\rho$ batched FINUFFT transforms, without fusion.
The last two columns give the speedups over direct synthesis at $M=2\times10^6$.}\label{tab:2d}
\begin{tabular}{rrrr rr rrr r rr}\toprule
&&&&\multicolumn{2}{c}{$M=10^5$}&\multicolumn{3}{c}{$M=2\times10^6$}&3-D/2-D&\multicolumn{2}{c}{vs direct}\\
\cmidrule(lr){5-6}\cmidrule(lr){7-9}\cmidrule(lr){11-12}
$L$&$\varepsilon$&$K{+}1$&$\rho$&3-D&2-D&3-D&2-D&FINUFFT 2-D&($2{\times}10^6$)&3-D&2-D\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")

# ---------------- machine precision -----------------
EX = json.load(open('/home/claude/fastpot/bench_exact.json'))
SC = json.load(open('/home/claude/fastpot/exact_scan.json'))
rows=[]
for L in sorted({e['L'] for e in EX}):
    r=[e for e in EX if e['L']==L]
    get_=lambda key: next(e for e in r if e['cfg'].startswith(key))
    m2,m3,t10=get_("machine precision, sigma=2"),get_("machine precision, sigma=3"),get_("1e-10")
    rows.append(f"{L} & {m3['rho']} & {m3['direct']:.2f} & {m2['t']:.2f} & {fs(m2['direct']/m2['t'])} & {m3['t']:.2f} & {fs(m3['direct']/m3['t'])} & {m3['mem']/1e6:.0f} & {t10['t']:.2f} & {fs(t10['direct']/t10['t'])}\\\\")
open("tab_exact.tex","w").write(r"""\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4.5pt}
\caption{Fused 2-D transform at machine precision versus direct synthesis, $M=10^6$ points, 250--800\,km shell, 2 cores.
``Machine precision'' means a relative error of $\approx2$--$3\times10^{-14}$ against a long-double reference, the same as
the direct double-precision sum (\cref{tab:wscan}). $\rho$: radial rank (tolerance $10^{-15}$). Times in seconds, evaluation only.}\label{tab:exact}
\begin{tabular}{rrr rr rrr rr}\toprule
&&&\multicolumn{2}{c}{$\sigma=2$, $w=16$}&\multicolumn{3}{c}{$\sigma=3$, $w=13$}&\multicolumn{2}{c}{$\varepsilon\approx10^{-10}$, $w=12$}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}
$L$&$\rho$&direct&time&speedup&time&speedup&grid (MB)&time&speedup\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
ws=range(8,17)
def cell(s,w):
    v=SC[s].get(str(w)); return f"{v:.0e}".replace("e-0","e-").replace("e-","\\times10^{-")+"}" if v else "--"
lines=[]
for s in ("sigma=2","sigma=3","sigma=4"):
    lines.append(f"$\\sigma={s[-1]}$ ({SC['grid_MB'][s]}\\,MB) & "+" & ".join(("$"+cell(s,w)+"$" if SC[s].get(str(w)) else "--") for w in ws)+"\\\\")
open("tab_wscan.tex","w").write(r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Maximum relative error of the fused 2-D transform against a 19-digit (long double) reference, as a function of
kernel width $w$ and oversampling $\sigma$ ($L=185$, 1500 points including both poles, radial rank $\rho=14$).
For comparison, the direct double-precision sum has error $1.6\times10^{-14}$. Grid sizes are the stored oversampled grids.}\label{tab:wscan}
\begin{tabular}{l"""+"c"*len(ws)+r"""}\toprule
&"""+" & ".join(f"$w={w}$" for w in ws)+r"""\\\midrule
"""+"\n".join(lines)+r"""
\bottomrule\end{tabular}\end{table}
""")
