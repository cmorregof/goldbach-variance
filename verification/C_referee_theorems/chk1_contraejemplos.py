"""Árbitro 3: Prop. 3 (contraejemplos), r = rho*tau, cotas de tau y rho, constantes c0 y Gamma_min."""
import sys, math, json, numpy as np
sys.path.insert(0, '/home/claude/ariadna/R3/run6_E1')
from arb_comun import sieve, factor, sing, r_def, rho_def, rho_mult, Gamma_n, C2
import e1_comun  # código de la sumisión, solo para cruzarlo

out = {}
pr = sieve(1 << 21)   # 2^21 > 2n+2 para n <= 10^6

# --- 0. cruce de mi r (desde la definición) con el de la sumisión
for n in (1000, 2310, 30030):
    hs = np.arange(1, n)
    mine = r_def(n, hs, pr)
    theirs, theirs_rho = e1_comun.r_and_rho(n, n - 1, e1_comun.primes_upto(1 << 24))
    rel = np.abs(mine - theirs) / np.maximum(np.abs(theirs), 1e-300)
    rel[theirs == 0] = np.abs(mine[theirs == 0])
    myrho = rho_def(n, hs, pr); myrho2 = rho_mult(n, hs, pr)
    relr = np.abs(myrho - theirs_rho) / np.maximum(theirs_rho, 1e-300); relr[theirs_rho == 0] = np.abs(myrho[theirs_rho == 0])
    relm = np.abs(myrho - myrho2) / np.maximum(myrho, 1e-300); relm[myrho == 0] = np.abs(myrho2[myrho == 0])
    print(f"cruce n={n}: max rel dif r = {rel.max():.2e} ; rho(def) vs sumisión {relr.max():.2e} ; rho(def) vs rho(mult) {relm.max():.2e}")
    out[f"cruce_{n}"] = dict(r=float(rel.max()), rho=float(relr.max()), rho_mult=float(relm.max()))

# --- 1. Prop 3(a): H = 2
print("\n(a) H=2, 3∤n: LHS = sum_{0<|h|<=2}(r-1)")
rows = []
for n in (70, 770, 10010, 170170):
    r = r_def(n, [1, 2], pr); S = sing(n, pr)
    lhs = 2 * float(np.sum(r - 1)); rows.append(dict(n=n, r1=float(r[0]), r2=float(r[1]), lhs=lhs, rhs0=-math.log(2) ** 2 / (2 * S), S=S))
    print(f"  n={n}: r(1)={r[0]:.3e} r(2)={r[1]:.3e} LHS={lhs:+.6f}  -log^2(2)/(2S)={rows[-1]['rhs0']:+.4f}  S={S:.3f}")
out["a_3notdiv"] = rows
print("(a) primoriales (3|n): LHS -> 4C2-4 =", 4 * C2 - 4)
rows = []
for n in (30, 210, 2310, 30030, 510510):
    r = r_def(n, [1, 2], pr); S = sing(n, pr)
    lhs = 2 * float(np.sum(r - 1)); rows.append(dict(n=n, r2=float(r[1]), lhs=lhs, rhs0=-math.log(2) ** 2 / (2 * S), S=S))
    print(f"  n={n}: r(2)={r[1]:.5f} (2C2={2*C2:.5f}) LHS={lhs:+.5f}  RHS0={rows[-1]['rhs0']:+.4f}")
out["a_primorial"] = rows

# --- (a'): H fijo, primoriales, límite sum (S(h)-1)
def Sh(h):
    if h % 2: return 0.0
    s = 2 * C2
    for p in factor(h, pr):
        if p > 2: s *= (p - 1) / (p - 2)
    return s
print("\n(a') H fijo, primoriales: LHS(n_k,H) frente a Z(H)=sum_{0<|h|<=H}(S(h)-1) y -log H")
rows = []
for H in (10, 30, 100):
    Z = 2 * sum(Sh(h) - 1 for h in range(1, H + 1))
    row = dict(H=H, Z=Z, mlogH=-math.log(H))
    for n in (2310, 30030, 510510):
        r = r_def(n, np.arange(1, H + 1), pr)
        row[f"n{n}"] = 2 * float(np.sum(r - 1))
    rows.append(row); print("  ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
out["a_prime"] = rows

# --- constantes c0 y Gamma_min
P = pr[pr >= 5].astype(float)
c0 = math.exp(float(np.sum(np.log1p(-2.0 / ((P - 2) * (P - 3))))) - 2.0 / (P[-1] * math.log(P[-1])))
Gmin = 0.75 * math.exp(float(np.sum(np.log1p(-2.0 / ((P - 1) * (P - 2))))) - 2.0 / (P[-1] * math.log(P[-1])))
print(f"\nc0 = prod_{{p>=5}}(1-2/((p-2)(p-3))) = {c0:.6f} (sumisión: 0.54999) ; Gamma_min = (3/4)prod_{{p>=5}}(1-2/((p-1)(p-2))) = {Gmin:.6f} (sumisión: 0.5412)")
out["c0"] = c0; out["Gamma_min"] = Gmin
# ¿es Gamma_min realmente el mínimo? comparar factores
print("  factor p|n: 1-1/(p-1)^2 ; factor p∤n: 1-2/((p-1)(p-2)) ; el segundo es menor para todo p>=5:",
      all(1 - 1 / (p - 1) ** 2 > 1 - 2 / ((p - 1) * (p - 2)) for p in range(5, 1000)))

# --- Lema 4: r = rho*tau, cota inferior de tau, rho >= l*Gamma_min en lZ
print("\nLema 4 / Prop 3(b): cotas")
rng = np.random.default_rng(7)
worst_ratio = np.inf; viol = 0; viol_rho = 0; checks = 0
for n in sorted(set(int(2 * x) for x in rng.integers(1000, 400000, 12)) | {30030, 2310 * 7, 510510}):
    H = min(n - 1, 3000)
    hs = np.arange(1, H + 1)
    r = r_def(n, hs, pr); rho = rho_def(n, hs, pr)
    pf = set(factor(n, pr)); l = 2 if 3 in pf else 6; Gam = Gamma_n(n, pr)
    for h in hs[hs % l == 0]:
        h = int(h)
        tau = r[h - 1] / rho[h - 1]
        Q = [p for m in (n - h, n + h) for p in factor(m, pr) if p >= 5 and n % p and h % p]
        lb = c0 * np.prod([1 + 1 / (p - 2) for p in Q]) if Q else c0
        checks += 1
        if tau < lb * (1 - 1e-9): viol += 1
        worst_ratio = min(worst_ratio, tau / lb)
        if rho[h - 1] < l * Gmin * (1 - 1e-9): viol_rho += 1
    # multiplicatividad r = rho * tau con tau desde la definición (Lema 4(i)): tau_def
    tau_def = np.ones(H)
    for p in pr[(pr >= 5)]:
        p = int(p)
        if n % p == 0: continue
        if p > 2 * n + 2: break
        t = np.where(hs % p == 0, 0.0, np.where(((n - hs) % p == 0) | ((n + hs) % p == 0), 1 / (p - 2), -2 / ((p - 2) * (p - 3))))
        tau_def *= 1 + t
    Pm = float(pr[-1]); tau_def *= math.exp(-2.0 / (Pm * math.log(Pm)))  # cola: prod(1-2/p^2) aprox
    m = rho > 0
    print(f"  n={n}: max|r/(rho*tau_def)-1| = {np.max(np.abs(r[m] / (rho[m] * tau_def[m]) - 1)):.1e}")
print(f"  comprobaciones tau >= c0*prod(1+1/(p-2)): {checks}, violaciones {viol}, peor cociente tau/cota = {worst_ratio:.4f}; violaciones rho < l*Gamma_min: {viol_rho}")
out["tau_bound"] = dict(checks=checks, viol=viol, worst_ratio=float(worst_ratio), viol_rho=viol_rho)

# --- (c) h = ±n: factor local para p∤n, p>=5
print("\n(c) h=n: r_p(n) para p∤n, p>=5 vale p(p-3)/(p-2)^2 = 1+(p-4)/(p-2)^2 > 1:")
n = 30
for p in (7, 11, 13):
    s = {0, (-n) % p, n % p, 0}
    nu = len(s); val = (1 - nu / p) / (1 - 2 / p) ** 2
    print(f"  p={p}: nu={nu}, r_p={val:.5f}, fórmula={1 + (p - 4) / (p - 2) ** 2:.5f}")
print("  sum_p (p-4)/(p-2)^2 diverge => r(n,±n)=+inf si 3|n; si 3∤n, r_3(n)=0.")

json.dump(out, open('/home/claude/ariadna/R3/arbitro3_fable/chk1_contraejemplos.json', 'w'), indent=1)
