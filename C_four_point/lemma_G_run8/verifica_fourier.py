"""[M] Comprobaciones exactas de los lemas de Fourier de G2_prueba.md.
(F1) Transformada 2D de t_p en coordenadas (u,v) = (n-h, n+h) sobre (Z/p)^2:
     eje (a,0),(0,b), ab!=0 : 1/(p(p-2)) ;  diagonal a=b y antidiagonal a=-b : 0 ;  resto: -c2(p)/p.
(F2) Transformadas 1D: |coef. de Fourier de n -> t_p(n,h)| <= c2(p) y de h -> t_p(n,h) <= c2(p).
(F3) Lema P:  |sum_k phi_X(k) T_e(n, l d k)| <= c2(e) * [ (e^2/X) f(X/e) - f(X)/X ],  f(t)={t}(1-{t}),
     para X real (no sólo entero)."""
import math, json, itertools, numpy as np
from g2_comun import t_p_arr, T_e_arr, c2
res = {}
# (F1)
err1 = 0.0
for p in (5, 7, 11, 13, 17, 19, 23):
    inv2 = pow(2, -1, p)
    M = np.zeros((p, p))
    for u in range(p):
        for v in range(p):
            n = (u + v) * inv2 % p; h = (v - u) * inv2 % p
            M[u, v] = 0.0 if n == 0 else t_p_arr(p, n, np.array([h]))[0]
    F = np.fft.fft2(M) / p ** 2           # F[a,b] = (1/p^2) sum M[u,v] e(-(au+bv)/p)
    pred = np.full((p, p), -c2(p) / p, complex)
    for a in range(p):
        for b in range(p):
            if a == b or (a + b) % p == 0: pred[a, b] = 0
            elif a == 0 or b == 0: pred[a, b] = 1.0 / (p * (p - 2))
    err1 = max(err1, float(np.abs(F - pred).max()))
res["F1_max_err"] = err1; print("F1 max err", err1)
# (F2)
worst = 0.0
for p in (5, 7, 11, 13, 29, 31):
    for h in range(1, p):
        col = np.array([0.0 if n % p == 0 else t_p_arr(p, n, np.array([h]))[0] for n in range(p)])
        worst = max(worst, float(np.abs(np.fft.fft(col) / p).max() / c2(p)))
    for n in range(1, p):
        row = t_p_arr(p, n, np.arange(p))
        worst = max(worst, float(np.abs(np.fft.fft(row) / p).max() / c2(p)))
res["F2_max_ratio"] = worst; print("F2 max |coef|/c2(p) =", worst)
# (F3)
rng = np.random.default_rng(26102026)
def fpart(t): t = t - math.floor(t); return t * (1 - t)
ratios = []
for trial in range(400):
    k_ = rng.integers(1, 4); ps = sorted(rng.choice([5, 7, 11, 13, 17, 19, 23, 29, 31], k_, replace=False).tolist())
    e = int(np.prod(ps)); n = int(rng.integers(10, 10 ** 6)) * 2
    ld = int(rng.integers(1, 50)); X = float(rng.uniform(1.0, 3.0 * e))
    K = int(math.floor(X)); k = np.arange(-K, K + 1)
    phi = np.clip(1 - np.abs(k) / X, 0, None)
    S = float(np.dot(phi, T_e_arr(ps, n, ld * k)))
    B = np.prod([c2(p) for p in ps]) * ((e * e / X) * fpart(X / e) - fpart(X) / X)
    ratios.append(abs(S) / B if B > 0 else 0.0)
res["F3_max_ratio"] = float(max(ratios)); res["F3_median_ratio"] = float(np.median(ratios))
print("F3 max |S|/bound =", max(ratios), " mediana", np.median(ratios))
json.dump(res, open("verifica_fourier.json", "w"), indent=1)
