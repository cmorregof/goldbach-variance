"""(1) Para e = r/(N/log^2 N) - S500 : fraccion de Var(e) explicada por S(N) (bloque a bloque) y C(h) del resto.
(2) Ley de varianza: Var(W - S N) / (S N log^2 N) por bloques diadicos, tras quitar la parte suave
    (media movil) que contiene la oscilacion de los ceros.  theta=2 <=> cociente constante en log N.
"""
import numpy as np, json
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
def conv(a):
    f = np.fft.rfft(a, L); return np.fft.irfft(f*f, L)[:lim]
W = conv(Lam); R = np.rint(conv(isp.astype(float)))
Nev = np.arange(0, lim, 2)
C2 = np.prod([1-1.0/(p-1)**2 for p in primes[1:]])
S = np.full(len(Nev), 2*C2)
for p in primes[1:]:
    if p > len(Nev): break
    S[0::p] *= (p-1)/(p-2)   # Nev[k]=2k ; p | 2k <=> p | k
sel = Nev >= 6; n = Nev[sel]; N = n.astype(float); Sf = S[sel]; lg = np.log(N)
hs = np.arange(1, H+1)
def C(y):
    y = y - y.mean(); d = (y*y).mean()
    return np.array([(y[:-h]*y[h:]).mean()/d for h in hs])
out = {}
# (1) fuga, con S completa (S500 difiere de S en rms 0.003, irrelevante)
e = R[n]/(N/lg**2) - Sf
r2_blocks = []; resid = np.empty_like(e)
lo = 4
while lo <= X:
    m = (N >= lo) & (N < 2*lo)
    if m.sum() > 20:
        A = np.vstack([Sf[m], np.ones(m.sum())]).T
        c, *_ = np.linalg.lstsq(A, e[m], rcond=None)
        rr = e[m] - A@c; resid[m] = rr
        r2_blocks.append((int(lo), float(1 - rr.var()/e[m].var())))
    lo *= 2
out["R2_e_on_S_by_block"] = r2_blocks
out["C_e"] = C(e).tolist(); out["C_resid"] = C(resid).tolist()
# residuo estandarizado por su escala local (para que no dominen los N pequenos)
z = resid / np.sqrt(Sf*lg**2/N * lg**2)  # ~ sd natural de r/(N/log^2N)
out["C_resid_std"] = C(z).tolist()
# (2) theta
eps = W[n] - Sf*N
k = 25  # media movil de 2k+1 pares: quita la oscilacion suave de los ceros
cs = np.cumsum(np.concatenate([[0], eps]))
sm = (cs[2*k+1:] - cs[:-2*k-1])/(2*k+1)
hp = eps[k:-k] - sm
Nh = N[k:-k]; Sh = Sf[k:-k]
theta = []
lo = 1 << 12
while lo < X:
    m = (Nh >= lo) & (Nh < 2*lo)
    v = (hp[m]**2).mean(); base = (Sh[m]*Nh[m]*np.log(Nh[m])**2).mean()
    theta.append((int(lo), float(v/base)))
    lo *= 2
out["var_ratio_by_block"] = theta
json.dump(out, open("fuga_theta_2e24.json", "w"), indent=1)
np.set_printoptions(precision=3, suppress=True)
ix = [0, 1, 3, 2, 5, 14]
print("R2 de e sobre S por bloque (ultimos 8):", [(b, round(r, 3)) for b, r in r2_blocks[-8:]])
print("C(e)       h=1,2,4,3,6,15:", np.array(out["C_e"])[ix])
print("C(resid)   h=1,2,4,3,6,15:", np.array(out["C_resid"])[ix])
print("C(resid z) h=1,2,4,3,6,15:", np.array(out["C_resid_std"])[ix], " max|C| h>=1:",
      np.abs(out["C_resid_std"]).max())
print("Var(high-pass(W - S N)) / mean(S N log^2 N), por bloque:")
for b, r in theta: print(f"   2^{int(np.log2(b))}: {r:.3f}   (log N ~ {np.log(1.5*b):.1f})")
