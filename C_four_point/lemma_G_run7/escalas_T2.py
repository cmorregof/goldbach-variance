"""[M] ¿Dónde vive T2^F? Telescopio por escalas de primos:
  T2^F = sum_j D_j,  D_j = sum_h w(h) rho(h) tau_{<b_j}(h) (tau_{[b_j,b_{j+1})}(h) - 1),
con w = peso de Fejér (h y -h).  También A_j = sum w rho tau_{<b_j} |tau_j - 1| (masa absoluta).
Escalas: H^{1/4}, H^{1/2}, H^{3/4}, H/log^2 H, H, H log^2 H, H^2, 2n+3."""
import json, math, sys, numpy as np
from g_comun import r_and_rho, log_tau_blocks, primes_upto, fejer_w
pr = primes_upto(1 << 24)
rng = np.random.default_rng(7)
ns = sorted(set(int(2 * x) for x in rng.integers(500000, 1000000, 12))) + [510510 * 2, 2 ** 20, 9699690 // 10 * 10 + 2 - 2]
rows = []
for n in ns:
    if n % 2: n += 1
    r, rho = r_and_rho(n, n - 1, pr)
    for H in (1000, 10000, 100000, n - 1):
        L = math.log(H)
        cand = [5, H ** 0.25, H ** 0.5, H ** 0.75, H / L ** 2, H, H * L ** 2, H ** 2, 2 * n + 3]
        b = sorted(set(int(math.ceil(x)) for x in cand if x >= 5)) + [1 << 60]
        lt = log_tau_blocks(n, H, b, pr)
        w = fejer_w(H); base = w * rho[:H]
        cum = np.zeros(H); D = []; A = []
        for j in range(len(lt)):
            t_prev = np.exp(cum); tj = np.exp(lt[j])
            D.append(float(np.dot(base * t_prev, tj - 1))); A.append(float(np.dot(base * t_prev, np.abs(tj - 1))))
            cum += lt[j]
        T2 = float(np.dot(w, r[:H] - rho[:H]))
        rows.append(dict(n=n, H=H, bounds=b[:-1], D=D, A=A, T2=T2, check=float(T2 - sum(D))))
        print(n, H, "T2=%.4f" % T2, "D=", " ".join("%+.3f" % x for x in D), "| A=", " ".join("%.2f" % x for x in A), flush=True)
json.dump(rows, open("escalas_T2.json", "w"), indent=1)
