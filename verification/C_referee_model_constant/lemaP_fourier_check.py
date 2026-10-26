"""Árbitro: Lema P(b) con X real, y transformadas de n -> t_p(n,h) (Lema F paso 2), con código propio."""
import math, json, numpy as np
import arb_comun as A
rng = np.random.default_rng(2026)
def f(t): t = t - math.floor(t); return t * (1 - t)
# (1) Lema P(b): |sum_k phi_X(k) T_e(n, m k)| <= c2(e) [ (e^2/X) f(X/e) - f(X)/X ],  (m,e)=1
worst = 0.0; worst_case = None
for _ in range(600):
    ps = sorted(rng.choice([5, 7, 11, 13, 17, 19, 23, 29, 31, 37], int(rng.integers(1, 4)), replace=False).tolist())
    e = int(np.prod(ps)); n = 2 * int(rng.integers(3, 10 ** 6))
    while any(n % p == 0 for p in ps): n += 2    # evitar p | n (caso trivial)
    m = int(rng.integers(1, 60))
    while any(m % p == 0 for p in ps): m += 1
    X = float(rng.uniform(1.0, 2.5 * e))
    k = np.arange(-int(X), int(X) + 1)
    phi = np.clip(1 - np.abs(k) / X, 0, None)
    S = float(np.dot(phi, A.T_e(ps, n, m * k)))
    c2e = np.prod([A.c2(p) for p in ps])
    B = c2e * ((e * e / X) * f(X / e) - f(X) / X)
    assert B >= -1e-12
    ratio = abs(S) / B if B > 1e-15 else (0.0 if abs(S) < 1e-12 else float('inf'))
    if ratio > worst: worst, worst_case = ratio, (ps, n, m, X)
print("Lema P(b): max |S|/cota =", worst, worst_case)
# (2) Fourier en n: para h fijo, la función n -> t_p(n,h) sobre Z/p (t_p=0 si p|n) tiene media 0 y |coef| <= c2(p).
#     También la función m -> T_e(delta m, h) 1[delta m par] 1[delta m ≡ c (3)] sobre Z/6e: coeficiente 0 en J ≡ 0 (6e), |coef| <= c2(e).
w2 = 0.0; mean_max = 0.0
for p in (5, 7, 11, 13, 17, 19, 23):
    for h in range(1, p):
        col = A.t_p(p, 0, np.array([h]))  # dummy
        vals = np.array([0.0 if nn % p == 0 else float(A.t_p(p, nn, np.array([h]))[0]) for nn in range(p)])
        F = np.fft.fft(vals) / p
        mean_max = max(mean_max, abs(F[0])); w2 = max(w2, np.abs(F[1:]).max() / A.c2(p))
print("Fourier en n (mod p): max|media| =", mean_max, " max|coef|/c2(p) =", w2)
w3 = 0.0; zero_max = 0.0
for ps in ([5], [7], [5, 7], [5, 11], [7, 13]):
    e = int(np.prod(ps)); c2e = np.prod([A.c2(p) for p in ps])
    for delta in (1, 17, 19 * 23):
        for h in (1, 2, 3, 6, 10, 35, 77):
            for c in (0, 1):   # c=0: 3|n ; c=1: 3∤n
                m = np.arange(6 * e); nn = delta * m
                ind = ((nn % 2) == 0) & (((nn % 3) == 0) == (c == 0))
                Pv = A.T_e(ps, 0, np.zeros(1))  # dummy
                Pv = np.array([float(A.T_e(ps, int(x), np.array([h]))[0]) if all(x % p for p in ps) else 0.0 for x in nn]) * ind
                F = np.fft.fft(Pv) / (6 * e)
                zero_max = max(zero_max, abs(F[0]))
                # coeficientes con J ≡ 0 mod 6e es sólo J=0; también: J múltiplo de e (e | J) deben ser 0
                w3 = max(w3, np.abs(F).max() / c2e)
                assert np.abs(F[::e]).max() < 1e-13, "coef con e | J no nulo"
print("Fourier en m (mod 6e): max|P̂(0)| =", zero_max, " max|P̂(J)|/c2(e) =", w3, "(≤ 1/6 esperado)")
json.dump(dict(lemaP_max_ratio=worst, fourier_n_mean=mean_max, fourier_n_ratio=w2, fourier_6e_zero=zero_max, fourier_6e_ratio=w3), open("lemaP_fourier_check.json", "w"), indent=1)
