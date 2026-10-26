"""Residuo genuino: d(N) = W(N) - S(N) N  (pesos Lambda), ventana N in [X/4, X/2].
 (a) Var(d) / (S N log^k N) por bloque, k = 1, 2  (ley de varianza; sin filtrar)
 (b) C(h) del residuo estandarizado z = d / sqrt(S N log N): compara con S({0,2h}) y kappa.
"""
import numpy as np
from scipy.stats import spearmanr
LOGX = 24; X = 1 << LOGX; lim = X + 1; H = 60
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5)+1):
    if isp[p]: isp[p*p::p] = False
primes = np.nonzero(isp)[0]
Lam = np.zeros(lim)
for p in primes:
    lp = np.log(p); pk = p
    while pk < lim: Lam[pk] = lp; pk *= p
L = 1 << (LOGX+2)
f = np.fft.rfft(Lam, L); W = np.fft.irfft(f*f, L)[:lim]
Nev = np.arange(0, lim, 2)
C2 = np.prod([1-1.0/(p-1)**2 for p in primes[1:]])
S = np.full(len(Nev), 2*C2)
for p in primes[1:]:
    if p > len(Nev): break
    S[0::p] *= (p-1)/(p-2)
N = Nev.astype(float); d = W[Nev] - S*N
print("(a) Var(d)/(S N log^k N) por bloque:")
lo = 1 << 12
while lo < X:
    m = (N >= lo) & (N < 2*lo)
    v = d[m]
    print(f"   2^{int(np.log2(lo))}: mean d/sqrtN={np.mean(v/np.sqrt(N[m])):+.3f}  "
          f"k=1: {np.mean(v**2/(S[m]*N[m]*np.log(N[m]))):.3f}   k=2: {np.mean(v**2/(S[m]*N[m]*np.log(N[m])**2)):.3f}")
    lo *= 2
m = (N >= X/4) & (N <= X/2)
z = d[m]/np.sqrt(S[m]*N[m]*np.log(N[m])); z = z - z.mean()
hs = np.arange(1, H+1)
Cz = np.array([(z[:-h]*z[h:]).mean()/(z*z).mean() for h in hs])
def odd(h):
    o = []; n = h
    while n % 2 == 0: n //= 2
    p = 3
    while p*p <= n:
        if n % p == 0:
            o.append(p)
            while n % p == 0: n //= p
        p += 2
    if n > 1: o.append(n)
    return o
S0 = np.array([np.prod([(p-1)/(p-2) for p in odd(h)]) for h in hs])
gp = lambda p: 1 + 1.0/((p-2)*(p*p-2*p+2))
kap = np.array([np.prod([gp(p) for p in odd(h)]) for h in hs])
def fit(y, x):
    M = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(M, y, rcond=None); yh = M@c
    return np.round(c, 4), round(1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum(), 3)
print("(b) C(h) residuo estandarizado, ventana [X/4,X/2], h=1,2,3,4,5,6,15,30:",
      np.round(Cz[[0,1,2,3,4,5,14,29]], 4))
print("    vs S({0,2h}):", fit(Cz, S0), " spearman", round(spearmanr(Cz, S0).correlation, 3))
print("    vs g(h) [kappa]:", fit(Cz, kap), " spearman", round(spearmanr(Cz, kap).correlation, 3))
# Z (oscilacion de los ceros) es suave en N: ver si domina la correlacion a rezago corto
print("    C(1..5):", np.round(Cz[:5], 4), " C(56..60):", np.round(Cz[-5:], 4))
