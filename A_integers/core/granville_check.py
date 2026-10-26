"""Granville (2007, §2): conjetura que (g(2N) - I*(2N)) / sqrt(I*(2N)) "se ve Normal" y anota que los experimentos
no lo respaldan. Medimos la varianza y la curtosis de ese cociente, por bloques.
g = pares ordenados de primos, I = S(2N) * J(2N) (J = sum 1/(log a log b), pares ordenados),
I* = I * (1 - 4/sqrt(2N) * prod_{3<=p<=P} (1 - (2N|p)/(p-2)))   [Granville (1.1)], en una submuestra.
Referencia de 'independencia' para pares ordenados: Var = 2 I (cada par no ordenado cuenta dos veces).
"""
import numpy as np, scipy.fft as sf, json
from scipy.stats import kurtosis
LOGX = 24; X = 1 << LOGX; lim = X + 1; L = 2*X
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5)+1):
    if isp[p]: isp[p*p::p] = False
pr = np.nonzero(isp)[0]
def conv(a):
    f = sf.rfft(a, L, workers=2); f *= f; return sf.irfft(f, L, workers=2)[:lim:2].copy()
g = np.rint(conv(isp.astype(float)))
il = np.zeros(lim); il[2:] = 1/np.log(np.arange(2, lim)); J = conv(il); del il
C2 = np.prod(1 - 1.0/(pr[1:].astype(float)-1)**2)
S = np.full(X//2+1, 2*C2)
for p in pr[1:]:
    if p > X//2+1: break
    S[0::p] *= (p-1)/(p-2)
I = S*J
out = {}
print("bloque   Var((g-I)/sqrt(I))   kurtosis(exceso)   [indep. = 2]")
lo = 1 << 14
while lo < X:
    a, b = lo//2, lo
    z = (g[a:b] - I[a:b])/np.sqrt(I[a:b])
    out[int(np.log2(lo))] = dict(var=float(z.var()), mean=float(z.mean()), kurt=float(kurtosis(z)))
    print(f"2^{int(np.log2(lo))}:   {z.var():.4f}   media {z.mean():+.3f}   curtosis {kurtosis(z):+.3f}")
    lo *= 2
# submuestra con la correccion I* de Granville (cuadrados de primos)
rng = np.random.default_rng(1)
idx = rng.choice(np.arange(X//4, X//2), 200000, replace=False)
N2 = 2*idx                       # 2N
P = pr[(pr >= 3) & (pr <= 3000)]
prod = np.ones(len(idx))
for p in P:
    leg = np.array([pow(int(v) % int(p), (int(p)-1)//2, int(p)) for v in N2[:1]])  # placeholder (vectorizado abajo)
    r = N2 % p
    # simbolo de Legendre por exponenciacion modular vectorizada
    e = (p - 1)//2; base = r.astype(np.int64); res = np.ones_like(base); ee = e
    while ee:
        if ee & 1: res = (res*base) % p
        base = (base*base) % p; ee >>= 1
    chi = np.where(res == 0, 0, np.where(res == 1, 1, -1))
    prod *= 1 - chi/(p - 2)
Istar = I[idx]*(1 - 4/np.sqrt(N2)*prod)
z = (g[idx] - Istar)/np.sqrt(Istar); z0 = (g[idx] - I[idx])/np.sqrt(I[idx])
out["submuestra_2e23_2e24"] = dict(var_Istar=float(z.var()), mean_Istar=float(z.mean()), kurt_Istar=float(kurtosis(z)),
                                  var_I=float(z0.var()), mean_I=float(z0.mean()))
print("submuestra [2^23, 2^24], con I*:", out["submuestra_2e23_2e24"])
json.dump(out, open("granville_check_2e24.json", "w"), indent=1)
