#!/usr/bin/env python3
"""Build the LaTeX table rows of paper_ffq from the EXISTING data files of branch R2.

No new heavy computation: this script only reads the jsonl/json outputs of the
audit, the gate, the proof run and the referee, and prints rows. The only
arithmetic is ratios/differences of numbers already stored there.
Run:  python3 -I tablas.py  (from any directory; paths are absolute).
"""
import json, math

R = "/home/claude/ariadna"
AUD = R + "/auditorias/R2_numerica_2026-10-25.jsonl"
BF = R + "/R2/arbitro2_fable/arb2_bruteforce.jsonl"
SSE2 = R + "/R2/arbitro2_fable/arb2_ss_e2.json"
E2 = R + "/R2/run5_T1/T1_e2.jsonl"
LB = R + "/R2/arbitro2_fable/arb2_lemaB_runs.jsonl"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith("{")]


def binom2(m):
    return m * (m - 1) // 2


print("% ---- Table 1: the four terms, q large (units q^{2n}) ----")
bf = {(d["q"], d["n"]): ("B", d) for d in jl(BF)}
aud = {(d["q"], d["n"]): ("A", d) for d in jl(AUD)}
rows = dict(aud)
rows.update(bf)  # brute force (referee, independent code) takes precedence
keep = [(11, 2), (31, 2), (101, 2), (11, 3), (17, 3), (23, 3), (31, 3), (41, 3), (101, 3), (151, 3),
        (7, 4), (11, 4), (13, 4), (23, 4), (37, 4),
        (11, 5), (17, 5), (7, 6), (11, 6)]
for key in keep:
    src, d = rows[key]
    q, n = key
    t = binom2(n - 1)
    dv = d["VarG_minus_Ep"]
    print(f"{q} & {n} & {d['A0']:.4f} & {d['SS']:.4f} & {d['Xpp']:.4f} & {d['XG']:.4f} & "
          f"{dv:.4f} & {t} & {q*(dv-t):.2f} & {src} \\\\" if n > 2 else
          f"{q} & {n} & {d['A0']:.4f} & {d['SS']:.4f} & {d['Xpp']:.4f} & {d['XG']:.4f} & "
          f"{dv:.1e} & {t} & {q*(dv-t):.1e} & {src} \\\\")
# consistency between the two independent codes on common rows
common = sorted(set(bf) & set(aud))
mx = 0.0
for k in common:
    for f in ("A0", "SS", "Xpp", "XG", "VarG_minus_Ep"):
        a, b = aud[k][1][f], bf[k][1][f]
        mx = max(mx, abs(a - b) / max(1e-300, abs(a)))
print(f"% common rows audit/brute force: {common}; max rel. diff = {mx:.1e}")
mxid = max(max(d.get("identity_relerr", 0) for d in jl(AUD)), max(d.get("identity_relerr", 0) for d in jl(BF)))
print(f"% max identity rel. error (Var_G = A0+Ep+SS+X) over all rows: {mxid:.1e}")

print("% ---- Table 1b: E_p and Var_G separately (measurement only) ----")
for key in [(31, 3), (101, 3), (151, 3), (11, 4), (17, 4), (23, 4), (29, 4), (37, 4), (11, 5), (17, 5), (11, 6)]:
    d = aud[key][1]
    print(f"{key[0]} & {key[1]} & {d['Ep']:.3f} & {d['VarG']:.3f} \\\\")

print("% ---- Table 1c: fixed q (audit) ----")
for key in [(3, 4), (3, 6), (3, 8), (3, 10), (3, 12), (5, 4), (5, 6), (5, 7), (5, 8)]:
    d = aud[key][1]
    q, n = key
    print(f"{q} & {n} & {d['A0']+d['SS']:.3f} & {d['X']:.3f} & {d['VarG_minus_Ep']:.3f} & {n*n/2:.1f} & {d['Ep']/(n*n):.3f} \\\\")

print("% ---- Table 2: q (SS/q^{2n} + binom(n+2,2)) at q=1e4,1e5,1e6 ----")
s = json.load(open(SSE2))
for n in range(2, 7):
    v = s[f"qSS_n{n}"]
    print(f"{n} & " + " & ".join(f"{x:.6f}" for x in v) + f" & {-(n-2)} \\\\")
print(f"% max rel err SS formula vs audit: {s['ss_max_relerr_vs_audit']:.1e}")

print("% ---- Table 3: beta_q, gamma_q, residuals ----")
e2 = jl(E2)
for d in e2:
    res = dict((int(a), b) for a, b in d["residuals"])
    def f(x):
        return "---" if abs(x) < 1e-12 else f"{x:.1e}"
    print(f"{d['q']} & {d['beta_q']:.6f} & {d['gamma_q']:.6f} & {f(res[10])} & {f(res[20])} & {f(res[30])} \\\\")
# cross-check with the referee's independent mpmath route
for q in (3, 5, 7, 11, 101):
    a = s[f"e2_q{q}"]
    d = [x for x in e2 if x["q"] == q][0]
    print(f"% q={q}: |beta diff|={abs(a['beta']-d['beta_q']):.1e}, |gamma diff|={abs(a['gamma']-d['gamma_q']):.1e}")
# per-step ratio of residuals for q=3 (the referee's 'about 1.95')
r = s["e2_q3"]["res"]
xs = sorted((int(k), abs(v)) for k, v in r.items())
for (n1, a1), (n2, a2) in zip(xs, xs[1:]):
    print(f"% q=3 per-step ratio n={n1}->{n2}: {(a1/a2)**(1/(n2-n1)):.3f}")
(n1, a1), (n2, a2) = xs[0], xs[-1]
print(f"% q=3 per-step ratio n={n1}->{n2} (overall): {(a1/a2)**(1/(n2-n1)):.3f}; sqrt(3)={math.sqrt(3):.3f}")

print("% ---- Table 4: spectral test (referee) ----")
for d in jl(LB):
    q, degs = d["q"], d["degs"]
    VW = d["VW"]
    ns = sorted(VW, key=int)
    vp = [VW[k]["Vp"] for k in ns if int(k) >= 2]
    wq = [VW[k]["Wp_times_qm1"] for k in ns]
    hp, hm = VW["3"]["half_plus"], VW["3"]["half_minus"]
    cs = d["class_stats"]["2"]
    print(f"% q={q} degs={degs} nTR={d['nTR']} classes={d['classes']} minclass={d['min_class']}")
    print(f"{q} & {degs} & {d['nTR']} & {d['classes']} & {d['min_class']} & {min(vp):.2f}--{max(vp):.2f} & "
          f"{min(wq):.2f}--{max(wq):.2f} & {hp:.3f}/{hm:.3f} & {cs['max_sqrtq_tw']:.1f}/{cs['mean_sqrtq_tw']:.2f} & "
          f"{cs['max_sqrtq_un']:.1f}/{cs['mean_sqrtq_un']:.2f} \\\\")
