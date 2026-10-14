import json, numpy as np
P='/home/claude/fastpot/'
R=json.load(open(P+'bench_final.json'))
gv=lambda L,e,M: next(x for x in R if x['L']==L and x['eps']==e and x['M']==M)
fs=lambda x: f"{x:.1f}" if x<10 else f"{x:.0f}"
def sci(v):
    m,e=f"{v:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
Ls=sorted({x['L'] for x in R})
# main results table
rows=[]
for L in Ls:
    for j,e in enumerate((1e-6,1e-10,1e-13)):
        x=gv(L,e,10**6)
        first = f"\\multirow{{3}}{{*}}{{{L}}} & \\multirow{{3}}{{*}}{{{x['t_direct']:.2f}}} & \\multirow{{3}}{{*}}{{{sci(x['err_direct'])}}}" if j==0 else " & & "
        el = "$10^{%d}$" % int(np.log10(e))
        rows.append(first+f" & {el} & {x['w']} & {x['rho']} & {x['t_fast']:.2f} & \\textbf{{{fs(x['t_direct']/x['t_fast'])}}} & {sci(x['err'])}\\\\"+("\\addlinespace[2pt]" if j==2 else ""))
open("tab_results.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Main results (default fused loop): $M=10^6$ points in the 250--800\,km shell, two CPU cores. Direct sum: time (s) and relative error
against the long-double reference. Fast method, for each requested accuracy $\varepsilon$: window width $w$, radial rank $\rho$, evaluation time (s),
speedup over the direct sum, and measured maximum relative error against the long-double reference.}\label{tab:results}
\begin{tabular}{rrr rrrrrr}\toprule
&\multicolumn{2}{c}{direct sum}&\multicolumn{6}{c}{fast method}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-9}
$L$&time&error&$\varepsilon$&$w$&$\rho$&time&speedup&error\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
# set-up cost and memory
rows=[]
for L in Ls:
    c=[str(L)]
    for e in (1e-6,1e-10,1e-13):
        x=gv(L,e,10**6); c+=[f"{x['t_pre']:.2f}", f"{x['mem']/1e6:.1f}" if x['mem']<1e7 else f"{x['mem']/1e6:.0f}"]
    rows.append(" & ".join(c)+r"\\")
open("tab_setup.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Set-up (precomputation) time in seconds and size of the stored grid in MB.}\label{tab:setup}
\begin{tabular}{r rr rr rr}\toprule
&\multicolumn{2}{c}{$\varepsilon=10^{-6}$}&\multicolumn{2}{c}{$\varepsilon=10^{-10}$}&\multicolumn{2}{c}{$\varepsilon=10^{-13}$}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
$L$&set-up&grid&set-up&grid&set-up&grid\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
# calibration table
cal=json.load(open(P+'fastfield_calibration.json'))
ws=list(range(4,16))
def worst(sig,w):
    v=[cal[f"L{L}_s{sig}"].get(str(w),{}).get("err") for L in (13,185)]
    v=[x for x in v if x is not None]; return max(v) if v else None
def fmt(v):
    if v is None: return "--"
    m,e=f"{v:.0e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
rows=[f"{w} & {fmt(worst(2,w))} & {fmt(worst(3,w))}\\\\" for w in ws]
open("tab_calib.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Measured maximum relative error of the complete transform as a function of the window width $w$
(worst case of $L=13$ and $L=185$, 1500 points including both poles, radial rank at full precision).
The code picks the smallest $w$ whose $\sigma=2$ error is below the requested $\varepsilon$.
For comparison, the direct double-precision sum has error $1.6\times10^{-14}$ at $L=185$.}\label{tab:calib}
\begin{tabular}{rcc}\toprule
$w$&$\sigma=2$ (default)&$\sigma=3$\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
# old vs new
O=json.load(open(P+'bench_fastfield.json'))
rows=[]
for x in O:
    rows.append(f"{x['L']} & {x['w']} & {x['rho_new']} & {x['t_old']:.2f} & {x['t_new']:.2f} & {x['t_old']/x['t_new']:.2f} & {x['mem_old']/1e6:.0f} & {x['mem_new']/1e6:.0f}\\\\")
open("tab_oldnew.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{Effect of the two simplifications of \cref{sec:real,sec:half}: the earlier complex-valued, full-grid implementation (``old'')
versus the real-valued, half-grid implementation (``new''), at identical window width $w$, $\sigma=2$ and radial rank $\rho$.
$M=10^6$ points; times in seconds, grid sizes in MB (0 means below 0.5\,MB).}\label{tab:oldnew}
\begin{tabular}{rrr rrr rr}\toprule
$L$&$w$&$\rho$&old&new&speedup&old grid&new grid\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
# 3-D comparison (appendix)
B3=json.load(open(P+'bench_results.json'))
g3=lambda L,cfg,M: next(b for b in B3 if b['L']==L and b['cfg']==cfg and b['M']==M)
rows=[]
for L in (13,20,45,100,133,185):
    a6=g3(L,"double 1e-6",10**6); a10=g3(L,"double 1e-10",10**6)
    n6=gv(L,1e-6,10**6) if L in Ls else None; n10=gv(L,1e-10,10**6)
    rows.append(f"{L} & {a6['K']} & {a6['t_fast']:.2f} & {n6['t_fast']:.2f} & {a10['K']} & {a10['t_fast']:.2f} & {n10['t_fast']:.2f}\\\\")
open("tab_3d.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{The 3-D method (FINUFFT, $\sigma=2$) versus the fused 2-D method of the main text, $M=10^6$ points, evaluation times in seconds.
$K$: radial (Chebyshev) degree of the 3-D method. The two were run in separate sessions on the same machine. At the same requested $\varepsilon$
the 2-D method is also more accurate (e.g.\ $1.6\times10^{-11}$ versus $\approx2\times10^{-10}$ at $L=185$, $\varepsilon=10^{-10}$).}\label{tab:3d}
\begin{tabular}{r rrr rrr}\toprule
&\multicolumn{3}{c}{$\varepsilon=10^{-6}$}&\multicolumn{3}{c}{$\varepsilon=10^{-10}$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$L$&$K$&3-D&2-D&$K$&3-D&2-D\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
print("ok")
# backends
BK=json.load(open(P+'bench_backends.json'))
rows=[]
for x in BK:
    f,n=x['finufft'],x['numba']
    el="$10^{%d}$" % int(np.log10(x['eps']))
    rows.append(f"{x['L']} & {el} & {f['w']} & {f['t']:.2f} & {sci(f['err'])} & \\textbf{{{fs(x['direct']/f['t'])}}} & {n['t']:.2f} & {sci(n['err'])} & {fs(x['direct']/n['t'])}\\\\"+("\\addlinespace[2pt]" if x['eps']==1e-13 else ""))
open("tab_backends.tex","w").write(r"""\begin{table}[!htbp]\centering\small
\caption{The two implementations of Step 5 at the same requested accuracy $\varepsilon$: FINUFFT's interpolation-only mode (the default)
and our own fused loop. $M=10^6$ points, 250--800\,km shell, two cores. Time in seconds; error is the maximum relative error against the long-double
reference; speedup is relative to the direct sum.}\label{tab:backends}
\begin{tabular}{rr r rrr rrr}\toprule
&&&\multicolumn{3}{c}{FINUFFT (optional)}&\multicolumn{3}{c}{own fused loop (default)}\\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
$L$&$\varepsilon$&$w$&time&error&speedup&time&error&speedup\\\midrule
"""+"\n".join(rows)+r"""
\bottomrule\end{tabular}\end{table}
""")
