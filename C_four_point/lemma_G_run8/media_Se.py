"""[M] Lema F de G2_prueba.md (Fourier en n):  |sum_n w(n/N) S_e(n)| <<  N' c2(e) e min(1, e/N)^sigma,
con S_e(n) = sum_{0<|h|<H} phi_H(h) rho(n,h) T_e(n,h).  Casos: H = n-1 y H fijo.
Se imprime A_e/(N' c2(e) e), con N' = sum_n w(n/N), junto con el máximo puntual de |S_e(n)|/(c2(e) e log^2 n).
Uso: python3 media_Se.py K"""
import sys, math, json, numpy as np
from g2_comun import r_rho, primes_upto, factor, w_smooth, T_e_arr, gamma_e
K = int(sys.argv[1]); N = 1 << K
pr = primes_upto(4 * N + 10)
P = [int(p) for p in primes_upto(300 * N) if p >= 5]
def nextp(x): return min(p for p in P if p >= x) if x <= P[-1] else None
E = [[5], [7], [13], [5, 7], [7, 11], [5, 7, 11], [5, 7, 11, 13], [5, 7, 11, 13, 17]]
for frac in (1 / 64, 1 / 8, 1 / 2, 1, 2, 4):
    p = nextp(int(frac * N)); E.append([p])
    q = nextp(int(math.sqrt(frac * N))); q2 = nextp(int(frac * N / q) + 1)
    if q2 and q2 != q: E.append(sorted([q, q2]))
    E.append(sorted({5, 7, nextp(int(frac * N / 35) + 1)}))
for big in (16, 64):
    q = nextp(int(math.sqrt(big * N))); q2 = nextp(int(big * N / q) + 1); E.append(sorted([q, q2]))
E = [e for e in E if None not in e and len(set(e)) == len(e)]
Hfix = int(round(N ** 0.6))
acc = np.zeros(len(E)); accf = np.zeros(len(E)); pmax = np.zeros(len(E)); W = 0.0
for n in range(N + 2, 2 * N, 2):
    w = float(w_smooth(np.array([n / N]))[0])
    if w == 0: continue
    r, q = r_rho(n, pr); H = n - 1; h = np.arange(1, H + 1)
    base = 2.0 * (1 - h / H) * q
    basef = 2.0 * np.clip(1 - h / Hfix, 0, None) * q
    W += w
    for i, e in enumerate(E):
        T = T_e_arr(e, n, h)
        s = float(np.dot(base, T)); acc[i] += w * s; accf[i] += w * float(np.dot(basef[:Hfix], T[:Hfix]))
        pmax[i] = max(pmax[i], abs(s) / (gamma_e(e) * math.log(n) ** 2))
rows = []
for i, e in enumerate(E):
    ev = int(np.prod(e)); g = gamma_e(e)
    rows.append(dict(e=ev, primes=e, e_over_N=ev / N, A_e_over_Ngamma=acc[i] / (W * g),
                     A_e_over_Ngamma_min=acc[i] / (W * g * min(1.0, ev / N) ** 0.5),
                     Afix_over_Hgamma=accf[i] / (W * Hfix * g), max_pointwise=pmax[i]))
    print("e=%9d (e/N=%7.3f) A_e/(N'g)=%+.2e  /min(1,e/N)^.5=%+.2e  H fijo: %+.2e  puntual max %.3f" %
          (ev, ev / N, rows[-1]["A_e_over_Ngamma"], rows[-1]["A_e_over_Ngamma_min"], rows[-1]["Afix_over_Hgamma"], pmax[i]))
json.dump(dict(K=K, N=N, Hfix=Hfix, Nprime=W, rows=rows), open("media_Se_K%d.json" % K, "w"), indent=1)
