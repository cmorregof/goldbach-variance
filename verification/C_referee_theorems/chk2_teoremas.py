"""Árbitro 3: Teoremas 7, 7', 8 y Corolarios 5.1/5.2 contra cálculo exacto independiente.
Constante secundaria de Mellin (7') recalculada con mpmath (zeta) + producto de Euler (E_kappa)."""
import math, json, numpy as np, mpmath as mp
from arb_comun import sieve, factor, sing, Gamma_n, rho_def, EULER, C2

pr = sieve(1 << 24)
out = {}


def consts(n):
    pf = set(factor(n, pr)); l = 2 if 3 in pf else 6
    Gam = Gamma_n(n, pr); S = sing(n, pr)
    Sig = sum(math.log(p) / (p - 1) for p in pf if p > 2)
    A = 2 * EULER + 2 * math.log(2) + (0 if 3 in pf else math.log(3)) + Sig
    # log E_kappa: derivadas 1 y 2 en s=0 por producto de Euler (p <= 2^24)
    P = pr.astype(float); lp = np.log(P)
    inF = (P >= 3) & ~((P == 3) & (3 not in pf))
    isn = np.isin(pr, list(pf))
    kap = np.where(isn, 1 / (P - 2), 2 / (P - 3))
    x = np.where(inF, kap, 0.0)             # kappa p^{-s} en s=0
    y = 1 / P                                # p^{-1-s} en s=0
    d1 = -lp * x / (1 + x) + 2 * lp * y / (1 - y)
    d2 = (lp ** 2 * x * (1 + x) - lp ** 2 * x ** 2) / (1 + x) ** 2 - 2 * (lp ** 2 * y * (1 - y) + lp ** 2 * y ** 2) / (1 - y) ** 2
    a1 = float(np.sum(d1)); a2 = float(np.sum(d2))
    E0 = float(np.exp(np.sum(np.log1p(x) + 2 * np.log1p(-y))))
    # g: E_g(0) y A_g
    gp = np.where(isn, 1 / (P - 1), 2 / (P - 2))
    Eg0 = float(np.exp(np.sum(np.log1p(gp) + 2 * np.log1p(-y))))
    Ag = 2 * EULER + float(np.sum(-gp * lp / (1 + gp) + 2 * lp / (P - 1)))
    return dict(pf=sorted(pf), l=l, Gam=Gam, S=S, Sig=Sig, A=A, a1=a1, a2=a2, E0=E0, Eg0=Eg0, Ag=Ag)


def mellin_c2(a1, a2):
    """[s^2] de (s^2 zeta(s+1)^2) * (-2 zeta(s)/(s+1)) * exp(a1 s + a2 s^2/2); también el coeficiente [s^1]."""
    mp.mp.dps = 30
    g0, g1 = mp.stieltjes(0), mp.stieltjes(1)
    # s*zeta(s+1) = 1 + g0 s - g1 s^2 + ...
    z1 = [mp.mpf(1), g0, -g1]
    zt = mp.taylor(mp.zeta, 0, 2)                      # zeta(s) en 0: [-1/2, -log(2pi)/2, ...]
    def mul(a, b): return [sum(a[i] * b[k - i] for i in range(k + 1) if i < len(a) and k - i < len(b)) for k in range(3)]
    inv1 = [mp.mpf(1), mp.mpf(-1), mp.mpf(1)]           # 1/(1+s)
    ex = [mp.mpf(1), mp.mpf(a1), mp.mpf(a2) / 2 + mp.mpf(a1) ** 2 / 2]
    c = mul(mul(mul(mul(z1, z1), [-2 * v for v in zt]), inv1), ex)
    return [float(v) for v in c]


def kappa_arr(n, X, pf):
    k = np.ones(X + 1); k[0] = 0.0
    for p in pr[pr <= X]:
        p = int(p)
        if p == 2 or (p == 3 and 3 not in pf): k[p::p] = 0.0; continue
        k[p::p] *= (1 / (p - 2)) if p in pf else (2 / (p - 3))
        if p * p <= X: k[p * p::p * p] = 0.0
    return k


def g_arr(n, Q, pf):
    g = np.ones(Q + 1); g[0] = 0.0
    for p in pr[pr <= Q]:
        p = int(p)
        g[p::p] *= (1 / (p - 1)) if p in pf else (2 / (p - 2))
        if p * p <= Q: g[p * p::p * p] = 0.0
    return g


def rho_all(n, H, c):
    """rho(h), h=1..H, por la fórmula multiplicativa (Lema 4(ii))."""
    l, Gam = c['l'], c['Gam']
    X = H // l
    k = kappa_arr(n, max(X, 1), set(c['pf']))
    rho = np.zeros(H + 1)
    for d in range(1, X + 1):
        if k[d]: rho[l * d::l * d] += k[d]
    return l * Gam * rho[1:], k


Hs = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
for n in (2 ** 20, 1000002, 30030, 9699690):
    c = consts(n)
    l, Gam, S, A, Sig = c['l'], c['Gam'], c['S'], c['A'], c['Sig']
    tay = mellin_c2(c['a1'], c['a2'])
    cF = 2 * EULER - 1 + math.log(4 * math.pi) + Sig
    print(f"\n=== n={n}: S={S:.6f} l={l} Gamma={Gam:.6f} Sigma_n={Sig:.5f} A_n={A:.5f} ; l*Gamma*E0*S = {l*Gam*c['E0']*S:.8f} (debe ser 1); Eg0*S={c['Eg0']*S:.8f}")
    print(f"    Mellin: a1=E'/E0={c['a1']:.6f} (A_n-2g={A-2*EULER:.6f}) ; Taylor [1,s,s^2] = {tay} ; coef log X esperado A_n-1+log2pi = {A-1+math.log(2*math.pi):.6f}; c2={tay[2]:.6f}; cF(n)={cF:.5f}")
    rows = []
    Hmax = min(max(Hs), n - 1)
    rho, k = rho_all(n, Hmax, c)
    G = np.cumsum(g_arr(n, Hmax, set(c['pf'])))
    for H in Hs:
        if H > n - 1: continue
        X = H / l; L = math.log(X); LH = math.log(H)
        hh = np.arange(1, H + 1)
        # exacto
        fej = 2 * float(np.dot(1 - hh[:H - 1] / H, rho[:H - 1] - 1))
        sharp = 2 * float(np.sum(rho[:H] - 1))
        # predicciones
        pred7p = 1 - (0.5 * L * L + (A - 1 + math.log(2 * math.pi)) * L + tay[2]) / S          # Teorema 7' completo
        pred7p_logH = -(0.5 * LH * LH + cF * LH) / S                                        # en log H, sin constante
        d = np.arange(1, int(X) + 1); yv = X / d; Pw = yv - np.floor(yv) - 0.5
        W = float(np.sum(k[d] * Pw)) if len(d) else 0.0
        pred8 = -(0.5 * LH * LH + (2 * EULER + 2 + math.log(2) + Sig) * LH) / S - 2 * l * Gam * W
        Gpred = (0.5 * LH * LH + c['Ag'] * LH) / S
        Kx = l * Gam * float(np.sum(k[d[d < X]])); Kpred = (0.5 * L * L + A * L) / S
        rows.append(dict(H=H, fej=fej, res7p=fej - pred7p, res7p_logH=fej - pred7p_logH, sharp=sharp, W=W, res8=sharp - pred8,
                         logX53=L ** (5 / 3), Gres=float(G[H]) - Gpred, Kres=Kx - Kpred))
        print(f"  H={H:7d}: T1F={fej:+10.4f} | T1F-Mellin(7',con c2)={fej-pred7p:+.5f} | T1F-(logH,sin cte)={fej-pred7p_logH:+.4f} "
              f"| S_rho={sharp:+9.3f} W={W:+8.3f} res8={sharp-pred8:+.4f} (logX^(5/3)={L**(5/3):.1f}) | G-pred={float(G[H])-Gpred:+.4f} | lGK-pred={Kx-Kpred:+.4f}")
    out[str(n)] = dict(consts={k_: v for k_, v in c.items()}, taylor=tay, cF=cF, rows=rows)

# Identidad: T1F por rho_def (definición) vs rho_all, un caso
n = 30030; c = consts(n); H = 5000
hs = np.arange(1, H)
rd = rho_def(n, hs, sieve(1 << 17))
ra, _ = rho_all(n, H, c)
print(f"\nIdentidad rho(def) vs rho(mult) n={n}, h<{H}: max rel = {np.max(np.abs(rd - ra[:H-1]) / np.maximum(ra[:H-1], 1e-300)):.2e}")

# V(y) = sum_{b<=y} P(y/b)/b  (Walfisz) para ver la escala
print("\nV(y) = sum_{b<=y} P(y/b)/b :")
for y in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 3 * 10 ** 6):
    b = np.arange(1, y + 1); t = y / b; V = float(np.sum((t - np.floor(t) - 0.5) / b))
    print(f"  y={y}: V={V:+.5f}  (log y)^(2/3)={math.log(y)**(2/3):.3f}")
json.dump(out, open('/home/claude/ariadna/R3/arbitro3_fable/chk2_teoremas.json', 'w'), indent=1, default=str)
