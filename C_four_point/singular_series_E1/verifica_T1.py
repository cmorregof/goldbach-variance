"""[M] Verificación de los pasos probados (T1 = parte rho; suma de dimensión 2) contra cálculo exacto.
Para cada n: l = 2 (3|n) o 6 (3∤n);  F = primos impares salvo 3 si 3∤n;
  kappa(p) = 1/(p-2) (p|n), 2/(p-3) (p∤n);  Gamma = prod_{p in F} rho_p(gen);  X = H/l, L = log X.
Predicciones (Teoremas 3-5 de E1_prueba.md):
  l*Gamma*K(X) = (1/S)[L^2/2 + A L] + B + o(1),  A = 2g + 2log2 + [3∤n]log3 + sum_{p|n,p>2} log p/(p-1)
  Fejér T1  = -(1/S)[L^2/2 + (A+1-cf) L] + O_n(1),   cf = 2 - log(2 pi)
  sharp T1  = -(1/S)[L^2/2 + (A+2) L] - 2 l Gamma W(X) + O_n(1),  W(X) = sum_{d<=X} kappa(d) s(X/d)
  G(H) = sum_{q<=H} g(q) = (1/S)[log^2H/2 + Ag log H] + Bg + o(1),
  Ag = 2g + 1.5 log2 + sum_{p|n,p>2} (p+1)log p/(p(p-1)) + sum_{p∤n} 2 log p/(p(p-1))
"""
import json, numpy as np
from e1_comun import r_and_rho, g_array, sing, primes_upto, prime_factors, EULER
pr = primes_upto(1 << 25)
cf = 2 - np.log(2 * np.pi)

def consts(n):
    pf = set(prime_factors(n, pr))
    l = 2 if 3 in pf else 6
    ps = pr[(pr >= 3)].astype(float)
    Gam = 0.0
    for p in ps[:200000]:
        p = int(p)
        if p == 3 and 3 not in pf: continue
        Gam += np.log(p * (p - 2) / (p - 1) ** 2) if p in pf else np.log(p * (p - 3) / ((p - 1) * (p - 2)))
    Gam = np.exp(Gam)
    A = 2 * EULER + 2 * np.log(2) + (0 if 3 in pf else np.log(3)) + sum(np.log(p) / (p - 1) for p in pf if p > 2)
    pp = pr[pr < 10 ** 7].astype(float)
    mask = np.array([int(p) not in pf for p in pp])
    Ag = 2 * EULER + 1.5 * np.log(2) + sum((p + 1) * np.log(p) / (p * (p - 1)) for p in pf if p > 2) \
         + float(np.sum((2 * np.log(pp) / (pp * (pp - 1)))[mask & (pp > 2)]))
    return pf, l, Gam, A, Ag

def kappa_array(n, X, pf, l):
    k = np.ones(X + 1); k[0] = 0
    for p in pr[pr <= X]:
        p = int(p)
        if p == 2 or (p == 3 and 3 not in pf):
            k[p::p] = 0.0; continue
        k[p::p] *= 1.0 / (p - 2) if p in pf else 2.0 / (p - 3)
        if p * p <= X: k[p * p::p * p] = 0.0
    return k

ns = [2 ** 20, 999958, 999998, 1000002, 786432, 510510, 9699690, 2 * 3 * 5 * 7 * 11 * 13 * 17 * 3 * 5]
Hs = [100, 1000, 10000, 100000, 300000]
rows = []
for n in ns:
    pf, l, Gam, A, Ag = consts(n); S = sing(n, pr)
    Hmax = min(max(Hs), n - 1)
    r, rho = r_and_rho(n, Hmax, pr)
    G = np.cumsum(g_array(n, Hmax, pr))
    kap = kappa_array(n, Hmax // l + 1, pf, l)
    print(f"\nn={n}  S={S:.5f}  l={l}  Gamma={Gam:.6f}  A={A:.4f}  Ag={Ag:.4f}  l*Gamma*E(0)*S check -> see K fit")
    for H in Hs:
        if H > n - 1: continue
        X = H / l; L = np.log(X); LH = np.log(H)
        d = np.arange(1, int(X) + 1)
        Kx = l * Gam * np.sum(kap[1:int(np.ceil(X))][d[:len(kap[1:int(np.ceil(X))])] < X]) if X > 1 else 0.0
        Kpred = (0.5 * L * L + A * L) / S
        hh = np.arange(1, H + 1)
        sharp_rho = 2 * float(np.sum(rho[:H] - 1)); sharp = 2 * float(np.sum(r[:H] - 1))
        w = 1 - hh / H
        fej_rho = 2 * float(np.dot(w, rho[:H] - 1)); fej = 2 * float(np.dot(w, r[:H] - 1))
        dd = np.arange(1, int(np.floor(X)) + 1); y = X / dd; s = y - np.floor(y) - 0.5
        W = float(np.sum(kap[dd] * s))
        pred_f = -(0.5 * L * L + (A + 1 - cf) * L) / S
        pred_s = -(0.5 * L * L + (A + 2) * L) / S - 2 * l * Gam * W
        Gpred = (0.5 * LH * LH + Ag * LH) / S
        rec = dict(n=n, H=H, S=S, l=l, Gamma=Gam, A=A, Ag=Ag, K=Kx, Kpred=Kpred, fej_rho=fej_rho, pred_f=pred_f,
                   sharp_rho=sharp_rho, pred_s=pred_s, W=W, G=float(G[H]), Gpred=Gpred, sharp=sharp, fej=fej)
        rows.append(rec)
        print(f" H={H:6d}  lGK-pred={Kx-Kpred:+.4f}  G-pred={G[H]-Gpred:+.4f} | fej_rho-pred={fej_rho-pred_f:+.4f} "
              f"| sharp_rho-pred(incl W)={sharp_rho-pred_s:+.4f}  [W term={-2*l*Gam*W:+.3f}, logX^(5/3)={L**(5/3):.1f}]"
              f" | T2={sharp-sharp_rho:+.3f} T2f={fej-fej_rho:+.4f}", flush=True)
json.dump(rows, open("verifica_T1.json", "w"), indent=1)
