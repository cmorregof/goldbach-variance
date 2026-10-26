"""Que definicion de e(N) reproduce los valores del cap. 6?
Libro (§6.2): potencias de dos h: C = -0.27, -0.39, -0.33 ; 3|h: +0.67, +0.76, +0.74 ; 15|h: +0.92, +0.90;
ajuste C = 0.81 S({0,2h}) - 1.17, R^2 = 0.83, Spearman 0.83 ; X = 2^22.
"""
import sys, json, numpy as np
from math import gcd
from scipy.stats import spearmanr
LOGX = int(sys.argv[1]) if len(sys.argv) > 1 else 22
X = 1 << LOGX; H = 60; Q = 500
lim = X + 1
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5) + 1):
    if isp[p]: isp[p*p::p] = False
primes = np.nonzero(isp)[0]
Lam = np.zeros(lim); P = isp.astype(float)
for p in primes:
    lp = np.log(p); pk = p
    while pk < lim: Lam[pk] = lp; pk *= p
L = 1 << (LOGX + 2)
def conv(a, b=None):
    fa = np.fft.rfft(a, L); fb = fa if b is None else np.fft.rfft(b, L)
    return np.fft.irfft(fa*fb, L)[:lim]
W = conv(Lam)                     # sum Lambda Lambda
R = np.rint(conv(P))              # pares ordenados de primos
il = np.zeros(lim); il[2:] = 1/np.log(np.arange(2, lim))
J = conv(il)                      # sum 1/(log a log b), a,b>=2
Nev = np.arange(0, lim, 2)
def mu_phi(q):
    n = q; m = 1; ph = 1; p = 2
    while p*p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0: return 0, 0
            m = -m; ph *= p-1
        p += 1
    if n > 1: m = -m; ph *= n-1
    return m, ph
SQ = np.zeros(len(Nev))
for q in range(1, Q+1):
    m, ph = mu_phi(q)
    if m == 0: continue
    tab = np.array([sum(d*mu_phi(q//d)[0] for d in range(1, gcd(r, q)+1) if gcd(r, q) % d == 0)
                    for r in range(q)], float)
    SQ += tab[Nev % q]/ph**2
sel = Nev >= 6
N = Nev[sel].astype(float); n = Nev[sel]
lg = np.log(N)
variants = {
  "W/N - S500 (literal)":          W[n]/N - SQ[sel],
  "r/(N/log^2N) - S500":           R[n]/(N/lg**2) - SQ[sel],
  "r/(2N/log^2N) - S500 (no ord.)":R[n]/(2*N/lg**2) - SQ[sel],
  "r/J - S500":                    R[n]/J[n] - SQ[sel],
  "(W - S500 N)/sqrt(N)":          (W[n] - SQ[sel]*N)/np.sqrt(N),
  "(r - S500 N/log^2N)/sqrt(N)":   (R[n] - SQ[sel]*N/lg**2)/np.sqrt(N),
  "r - S500 N/log^2 N (crudo)":    R[n] - SQ[sel]*N/lg**2,
}
def odd_primes_of(h):
    out = []; m = h
    while m % 2 == 0: m //= 2
    p = 3
    while p*p <= m:
        if m % p == 0:
            out.append(p)
            while m % p == 0: m //= p
        p += 2
    if m > 1: out.append(m)
    return out
hs = np.arange(1, H+1)
S0 = np.array([np.prod([(p-1)/(p-2) for p in odd_primes_of(h)]) for h in hs])
def C(x, center):
    y = x - x.mean() if center else x
    d = (y*y).mean()
    return np.array([(y[:-h]*y[h:]).mean()/d for h in hs])
def fit(y, x):
    M = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(M, y, rcond=None); yh = M@c
    return c[0], c[1], 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
idx = [0, 1, 3, 7, 2, 5, 8, 14, 29]   # h=1,2,4,8 | 3,6,9 | 15,30
out = {}
print("h:            1      2      4      8  |   3      6      9  |  15     30   | a     b     R2   spear")
for k, v in variants.items():
    for center in (False, True):
        c = C(v, center); a, b, r2 = fit(c, S0); sp = spearmanr(c, S0).correlation
        out[f"{k} | center={center}"] = dict(C=c.tolist(), a=a, b=b, R2=r2, spear=sp)
        print(f"{k[:30]:30s} c={int(center)} " + " ".join(f"{c[i]:+.2f}" for i in idx) +
              f" | {a:+.2f} {b:+.2f} {r2:.2f} {sp:.2f}")
json.dump(out, open(f"variantes_2e{LOGX}.json", "w"), indent=1)
