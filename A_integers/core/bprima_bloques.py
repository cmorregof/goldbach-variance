"""B' hasta 2^28 con convolucion por bloques (memoria acotada, ~4 GB pico).
W(n)=sum Lam(a)Lam(n-a), D2(n)=2 sum Lam(a)^2 Lam(n-a)^2, n par <= X; d = W - S n.
Por bloques diadicos: total = E[d^2]/E[S n log^2 n], diag = E[D2]/..., off = total - diag.
Uso: python3 bprima_bloques.py LOGX [--check]
"""
import sys, json, time, numpy as np, scipy.fft as sf
LOGX = int(sys.argv[1]); CHECK = "--check" in sys.argv
X = 1 << LOGX; lim = X + 1
NB = 8; B = X // NB; L = 2 * B
t0 = time.time()
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5) + 1):
    if isp[p]: isp[p*p::p] = False
primes = np.nonzero(isp)[0]; del isp
Lam = np.zeros(lim, np.float32)
Lam[primes] = np.log(primes).astype(np.float32)
for p in primes[:np.searchsorted(primes, int(lim**0.5)) + 1]:
    pk = int(p) * int(p)
    while pk < lim: Lam[pk] = np.float32(np.log(p)); pk *= int(p)
print("criba", round(time.time()-t0), "s", flush=True)

def bloque(i, sq):
    seg = np.zeros(B, np.float64); hi = min((i+1)*B, lim)
    seg[:hi - i*B] = Lam[i*B:hi]
    return seg*seg if sq else seg

def conv_pares(sq):
    out = np.zeros(X//2 + 1, np.float64)
    for i in range(NB):
        Fi = sf.rfft(bloque(i, sq), L, workers=2)
        for j in range(i, NB):
            base = (i + j) * B
            if base > X: break
            Fj = Fi if j == i else sf.rfft(bloque(j, sq), L, workers=2)
            c = sf.irfft(Fi * Fj, L, workers=2)
            if i != j: c *= 2
            k0 = base % 2                         # primer indice local con n par
            top = min(L - 1, X - base)
            idx = np.arange(k0, top + 1, 2)
            out[(base + idx) // 2] += c[idx]
            del c
    return out

W = conv_pares(False); print("W", round(time.time()-t0), "s", flush=True)
np.save("/home/claude/cap6/_W_tmp.npy", W); del W
D2 = conv_pares(True); D2 *= 2; print("D2", round(time.time()-t0), "s", flush=True)
W = np.load("/home/claude/cap6/_W_tmp.npy")
if CHECK:   # contraste con FFT directa
    f = np.fft.rfft(Lam.astype(np.float64), 2*X); Wd = np.fft.irfft(f*f, 2*X)[:lim:2]
    print("check max|W-Wd| =", np.abs(W - Wd).max(), " max W =", W.max(), flush=True)
del Lam
C2 = np.prod(1 - 1.0/(primes[1:].astype(float) - 1)**2)
S = np.full(X//2 + 1, 2*C2)
for p in primes[1:]:
    if p > X//2 + 1: break
    S[0::p] *= (p - 1)/(p - 2)
del primes
print("S", round(time.time()-t0), "s", flush=True)
# d = W - S n, en sitio
CH = 1 << 22
for s0 in range(0, len(W), CH):
    n = 2.0*np.arange(s0, min(s0 + CH, len(W)))
    W[s0:s0+CH] -= S[s0:s0+CH]*n
d = W
rows = []
lo = 1 << 12
while lo < X:
    a, b = lo//2, lo                              # indices de pares en [lo, 2lo)
    n = 2.0*np.arange(a, b); lg = np.log(n); s = S[a:b]
    b2 = (s*n*lg**2).mean(); v = (d[a:b]**2).mean(); dg = D2[a:b].mean()
    rows.append(dict(k=int(np.log2(lo)), meanlog=float(lg.mean()), total=float(v/b2),
                     diag=float(dg/b2), off=float((v - dg)/b2)))
    lo *= 2
a, b = X//4, X//2 + 1
n = 2.0*np.arange(a, b); lg = np.log(n); s = S[a:b]; cl = (2*np.arange(a, b)) % 6
off = (d[a:b]**2 - D2[a:b])/(n*lg**2)
clases = {int(c): dict(off_over_S=float(off[cl == c].mean()/s[cl == c].mean()),
                       off_over_S2=float(off[cl == c].mean()/(s[cl == c]**2).mean())) for c in (0, 2, 4)}
json.dump(dict(rows=rows, clases=clases), open(f"bprima_bloques_2e{LOGX}.json", "w"), indent=1)
print(" k  <logN>  total   diag    off")
for r in rows:
    print(f"{r['k']:2d}  {r['meanlog']:6.2f}  {r['total']:.4f}  {r['diag']:.4f}  {r['off']:+.4f}")
print("clases (n en [X/2,X]):", clases)
print("fin", round(time.time()-t0), "s")
