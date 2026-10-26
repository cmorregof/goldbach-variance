"""[M] Primer momento de T2^F(n, n-1) sobre TODOS los n pares de [N, 2N], con el peso suave w(n/N)
(g2_comun.w_smooth), y con el peso de la aplicación w(n/N) S(n)^2 (n-1).  Teorema 9 de G2_prueba.md:
media = O(log N loglog N).  Se compara con la ley del extremo de run 7:  -1_{3|n} c(n) * 2 log 2.
Uso: python3 media_T2.py K   (N = 2^K)."""
import sys, math, json, numpy as np
from g2_comun import r_rho, primes_upto, factor, w_smooth
K = int(sys.argv[1]); N = 1 << K
pr = primes_upto(4 * N + 10)
P = primes_upto(1 << 22); P5 = P[P >= 5].astype(float)
cbase = float(np.sum(np.log1p(-1 / (P5 - 2) ** 2)))
C2 = 0.66016181584686957
def sing(n, fs):
    s = 2 * C2
    for p in fs:
        if p > 2: s *= (p - 1) / (p - 2)
    return s
rows = []
for n in range(N + 2, 2 * N, 2):
    r, q = r_rho(n, pr)
    H = n - 1; h = np.arange(1, H + 1)
    T2 = 2.0 * float(np.dot(1 - h / H, r - q))
    fs = factor(n, pr)
    c = math.exp(cbase - sum(math.log1p(-1 / (p - 2) ** 2) for p in fs if p >= 5))
    law = -(c * 2 * math.log(2)) if n % 3 == 0 else 0.0
    rows.append((n, T2, law, sing(n, fs)))
a = np.array(rows); n = a[:, 0]; T2 = a[:, 1]; law = a[:, 2]; S = a[:, 3]
w = w_smooth(n / N)
out = dict(K=K, N=N, count=len(a),
           mean_w=float(np.sum(w * T2) / np.sum(w)),
           mean_w_resid=float(np.sum(w * (T2 - law)) / np.sum(w)),
           mean_w_law=float(np.sum(w * law) / np.sum(w)),
           mean_plain=float(T2.mean()), rms=float(np.sqrt(np.mean(T2 ** 2))),
           rms_resid=float(np.sqrt(np.mean((T2 - law) ** 2))), max_abs=float(np.abs(T2).max()),
           max_abs_resid=float(np.abs(T2 - law).max()),
           mean_w_S2=float(np.sum(w * S ** 2 * (n - 1) * T2) / np.sum(w * S * (n - 1) * np.log(n - 1) ** 2)),
           logN_loglogN=math.log(N) * math.log(math.log(N)))
print(json.dumps(out, indent=1))
json.dump(out, open("media_T2_K%d.json" % K, "w"), indent=1)
