"""B': varianza del error genuino d(n) = W(n) - S(n) n, separando el termino diagonal.
 diag(n) = 2 * sum_a Lambda(a)^2 Lambda(n-a)^2  (lo que da la heuristica 'independiente')
 off(n)  = d(n)^2 - diag(n)                      (covarianzas entre terminos distintos)
Por bloques diadicos: medias normalizadas por S n log^2 n y por S n log n.
Uso: python3 bprima_num.py LOGX   (L = 2X basta para n <= X)
"""
import sys, json, numpy as np, scipy.fft as sf
LOGX = int(sys.argv[1]); X = 1 << LOGX; lim = X + 1; L = 2 * X
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5) + 1):
    if isp[p]: isp[p*p::p] = False
primes = np.nonzero(isp)[0]
Lam = np.zeros(lim)
for p in primes:
    lp = np.log(p); pk = int(p)
    while pk < lim: Lam[pk] = lp; pk *= int(p)
del isp
def selfconv_even(a):
    f = sf.rfft(a, L, workers=2); f *= f
    out = sf.irfft(f, L, workers=2)[:lim:2].copy(); del f
    return out
W = selfconv_even(Lam)
D2 = 2 * selfconv_even(Lam**2)
del Lam
Nev = np.arange(0, lim, 2)
C2 = np.prod(1 - 1.0/(primes[1:].astype(float) - 1)**2)
S = np.full(len(Nev), 2*C2)
for p in primes[1:]:
    if p > len(Nev): break
    S[0::p] *= (p-1)/(p-2)
N = Nev.astype(float)
d = W - S*N
rows = []
lo = 1 << 12
while lo < X:
    m = (N >= lo) & (N < 2*lo)
    lg = np.log(N[m]); base2 = (S[m]*N[m]*lg**2).mean(); base1 = (S[m]*N[m]*lg).mean()
    v = (d[m]**2).mean(); dg = D2[m].mean(); off = v - dg
    rows.append(dict(k=int(np.log2(lo)), meanlog=float(lg.mean()),
                     var_over_SNL2=float(v/base2), diag_over_SNL2=float(dg/base2),
                     off_over_SNL2=float(off/base2), off_over_SNL=float(off/base1),
                     mean_d_over_sqrtN=float((d[m]/np.sqrt(N[m])).mean())))
    lo *= 2
json.dump(rows, open(f"bprima_2e{LOGX}.json", "w"), indent=1)
print(" k  <logN>  Var/SNL2  diag/SNL2  off/SNL2  off/SNL")
for r in rows:
    print(f"{r['k']:2d}  {r['meanlog']:6.2f}  {r['var_over_SNL2']:8.3f}  {r['diag_over_SNL2']:9.3f}  "
          f"{r['off_over_SNL2']:8.3f}  {r['off_over_SNL']:8.3f}")
