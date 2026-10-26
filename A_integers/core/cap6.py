"""Reproduccion independiente del C(h) del cap. 6 (Riano-Rojas), segun §6.2:
   e(N) = W_Lambda(N)/N - S_500(N),   C(h) = <e(N) e(N+2h)> / <e(N)^2>,  h = 1..60,
   N par <= X.  W_Lambda(N) = sum_{a+b=N} Lambda(a) Lambda(b) (convolucion FFT).
   S_Q(N) = sum_{q<=Q} mu^2(q)/phi(q)^2 c_q(N)  (serie singular truncada, 'paso bajo de Ramanujan').
Ademas: serie singular completa S(N), kappa(h), S({0,2h}) y prueba de 'fuga' de S en e.
"""
import sys, json, numpy as np
from math import gcd

LOGX = int(sys.argv[1]) if len(sys.argv) > 1 else 22
X = 1 << LOGX
Q = 500
H = 60

# --- criba y Lambda ---
lim = X + 1
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5) + 1):
    if isp[p]: isp[p*p::p] = False
primes = np.nonzero(isp)[0]
Lam = np.zeros(lim)
for p in primes:
    lp = np.log(p); pk = p
    while pk < lim:
        Lam[pk] = lp; pk *= p

# --- W_Lambda por FFT ---
L = 1 << (LOGX + 2)
F = np.fft.rfft(Lam, L)
W = np.fft.irfft(F * F, L)[:lim]
# control directo
for N in (1000, 123456, X - 2):
    a = np.arange(1, N); d = (Lam[a] * Lam[N - a]).sum()
    assert abs(d - W[N]) < 1e-6 * max(1, d), (N, d, W[N])

# --- S_Q(N) truncada ---
def mu_phi(q):
    n = q; m = 1; ph = 1; p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0: return 0, 0
            m = -m; ph *= p - 1
        p += 1
    if n > 1: m = -m; ph *= n - 1
    return m, ph

def mobius(n):
    return mu_phi(n)[0]

Nev = np.arange(0, lim, 2)               # todos los pares
SQ = np.zeros(len(Nev))
for q in range(1, Q + 1):
    m, ph = mu_phi(q)
    if m == 0: continue
    # c_q(r) para r = 0..q-1
    tab = np.zeros(q)
    for r in range(q):
        g = gcd(r, q); s = 0
        for dd in range(1, g + 1):
            if g % dd == 0: s += dd * mobius(q // dd)
        tab[r] = s
    SQ += tab[Nev % q] / ph**2

# --- S completa: 2*C2*prod_{p|N,p>2} (p-1)/(p-2) ---
C2 = np.prod([1 - 1.0/(p-1)**2 for p in primes[1:]])
S = np.full(len(Nev), 2*C2)
for p in primes[1:]:
    S[(Nev // 2) % p == 0] *= (p - 1) / (p - 2)   # p | N  <=> p | N/2 (p impar)

# --- e(N) ---
sel = Nev >= 6
N = Nev[sel].astype(float); Wn = W[Nev[sel]]
e = Wn / N - SQ[sel]
Sf = S[sel]; SQs = SQ[sel]

def Cfun(x, H=H, center=False):
    y = x - x.mean() if center else x
    d = (y * y).mean()
    return np.array([(y[:-h] * y[h:]).mean() / d for h in range(1, H + 1)])

C_raw = Cfun(e)
C_cen = Cfun(e, center=True)

# --- predicciones ---
def odd_primes_of(h):
    out = []; n = h
    while n % 2 == 0: n //= 2
    p = 3
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0: n //= p
        p += 2
    if n > 1: out.append(n)
    return out
hs = np.arange(1, H + 1)
S0 = np.array([np.prod([(p-1)/(p-2) for p in odd_primes_of(h)]) for h in hs])  # S({0,2h}) relativo
# kappa(h) = (2C2)^2 prod_p E_p(h) - 4
pr = primes[1:200000] if len(primes) > 200001 else primes[1:]
base = np.prod(1 + 2.0/(pr*(pr-2.0)))
A = (2*C2)**2 * base
def gp(p): return 1 + 1.0/((p-2)*(p*p-2*p+2))
kap = np.array([A*np.prod([gp(p) for p in odd_primes_of(h)]) - 4 for h in hs])
kap0 = 4*np.prod(1 + 1.0/(pr-1.0)**3) - 4

def fit(y, x):
    Mx = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(Mx, y, rcond=None)
    yh = Mx @ c; r2 = 1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum()
    return c, r2
def spear(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b).correlation

res = dict(X=X, mean_e=e.mean(), rms_e=e.std(), var_S=Sf.var(), kappa0=kap0)
for name, C in (("raw", C_raw), ("centered", C_cen)):
    cS, r2S = fit(C, S0); cK, r2K = fit(C, kap/kap0)
    res[name] = dict(C=C.tolist(), fitS=[*cS, r2S], spearS=spear(C, S0),
                     fitKnorm=[*cK, r2K], spearK=spear(C, kap))

# --- fuga: regresion de e sobre S en bloques diadicos ---
blocks = []
lo = 1 << 12
while lo < X:
    m = (N >= lo) & (N < 2*lo)
    if m.sum() > 100:
        b = np.cov(e[m], Sf[m])[0, 1] / Sf[m].var()
        blocks.append((lo, float(b), float(np.corrcoef(e[m], Sf[m])[0, 1]),
                       float(b*np.log(lo))))
    lo *= 2
res["leak_blocks"] = blocks
# residuo tras quitar la fuga bloque a bloque
eperp = e.copy()
lo = 8
while lo < X:
    m = (N >= lo) & (N < 2*lo)
    if m.sum() > 10:
        b = np.cov(e[m], Sf[m])[0, 1] / Sf[m].var()
        eperp[m] = e[m] - b*(Sf[m] - Sf[m].mean()) - (e[m].mean())
    lo *= 2
Cp = Cfun(eperp, center=True)
cS, r2S = fit(Cp, S0)
res["after_leak_removal"] = dict(C=Cp.tolist(), fitS=[*cS, r2S], spearS=spear(Cp, S0),
                                 frac_var_removed=float(1 - eperp.var()/((e-e.mean()).var())))
# tambien: residuo respecto a S completa en lugar de S500
e_full = Wn/N - Sf
res["e_full"] = dict(mean=e_full.mean(), rms=e_full.std(), C=Cfun(e_full, center=True).tolist())
res["SQ_minus_S_rms"] = float((SQs - Sf).std())
res["kappa"] = kap.tolist(); res["S0"] = S0.tolist()
json.dump(res, open(f"cap6_2e{LOGX}.json", "w"), indent=1)

np.set_printoptions(precision=3, suppress=True)
print("X=2^%d  mean e=%.3e  rms e=%.4f  var S=%.4f  rms(S500-S)=%.4f" %
      (LOGX, e.mean(), e.std(), Sf.var(), res["SQ_minus_S_rms"]))
for nm in ("raw", "centered"):
    r = res[nm]
    print(nm, "C(h) h=1,2,3,4,5,6,8,15,16,30:",
          np.array(r["C"])[[0,1,2,3,4,5,7,14,15,29]])
    print("   fit C = a S0 + b:", np.round(r["fitS"], 3), " spearman S0:", round(r["spearS"], 3),
          "| fit vs kappa/kappa0:", np.round(r["fitKnorm"], 3))
print("kappa/kappa0 h=1,2,3,4,5,6,8,15,16,30:", (kap/kap0)[[0,1,2,3,4,5,7,14,15,29]])
print("leak blocks (lo, beta, corr(e,S), beta*log lo):")
for b in blocks: print("   %9d  %+.5f  %+.3f  %+.4f" % b)
r = res["after_leak_removal"]
print("after leak removal: frac var removed=%.3f" % r["frac_var_removed"])
print("   C:", np.array(r["C"])[[0,1,2,3,4,5,7,14,15,29]], " fitS:", np.round(r["fitS"], 3),
      " spearS:", round(r["spearS"], 3))
print("e_full C:", np.array(res["e_full"]["C"])[[0,1,2,3,4,5,7,14,15,29]],
      " mean/rms:", res["e_full"]["mean"], res["e_full"]["rms"])
