"""R3 pre-test: EXACT off-diagonal from primes [M], dyadic blocks up to 2^LOGX.
off(n) = d(n)^2 - D2(n),  d = W - S n,  D2 = 2 sum_a Lam(a)^2 Lam(n-a)^2   (same as cap6/bprima_bloques.py)
Two weightings: 'Lam' (von Mangoldt, prime powers included) and 'theta' (log p on primes only).
Block normalisation = ratio of sums (as bprima_bloques.py):  off_blk = sum off / sum S n log^2 n.
Per class c = n mod 6:  off/S := sum_c off / sum_c S n log^2 n ;  off/S^2 := sum_c off / sum_c S^2 n log^2 n.
Usage: python3 r3_exacto.py LOGX
"""
import sys, json, time, numpy as np, scipy.fft as sf
from r3_common import sieve, von_mangoldt, singular_S

LOGX = int(sys.argv[1]); X = 1 << LOGX; L = 2 * X
t0 = time.time()
primes = sieve(X)
S = singular_S(X, primes)
n_all = np.arange(X + 1, dtype=float)
out = {}
for name, pp in (("Lam", True), ("theta", False)):
    Lam = von_mangoldt(X, primes, prime_powers=pp)
    F = sf.rfft(Lam, L, workers=2)
    W = sf.irfft(F * F, L, workers=2)[:X + 1]
    F = sf.rfft(Lam ** 2, L, workers=2)
    D2 = 2 * sf.irfft(F * F, L, workers=2)[:X + 1]
    del F
    d = W - S * n_all
    off = d ** 2 - D2
    rows = []
    for k in range(10, LOGX):
        lo, hi = 1 << k, 1 << (k + 1)
        idx = np.arange(lo, hi, 2)
        nn = n_all[idx]; s = S[idx]; lg2 = np.log(nn) ** 2
        base = (s * nn * lg2).sum()
        r = dict(k=k, meanlog=float(np.log(nn).mean()),
                 total=float((d[idx] ** 2).sum() / base), diag=float(D2[idx].sum() / base),
                 off=float(off[idx].sum() / base))
        # standard error of the block off (treating n as independent; rough)
        z = off[idx] / (nn * lg2)
        r["off_se"] = float(z.std() / np.sqrt(len(z)) / s.mean())
        cls = {}
        for c in (0, 2, 4):
            m = (idx % 6) == c
            cls[c] = dict(off_over_S=float(off[idx][m].sum() / (s[m] * nn[m] * lg2[m]).sum()),
                          off_over_S2=float(off[idx][m].sum() / (s[m] ** 2 * nn[m] * lg2[m]).sum()),
                          meanS=float(s[m].mean()), N=int(m.sum()))
        r["clases"] = cls
        rows.append(r)
    out[name] = rows
    print(name, "done", round(time.time() - t0), "s", flush=True)
    for r in rows:
        print(f"  k={r['k']:2d} total={r['total']:.4f} diag={r['diag']:.4f} off={r['off']:+.4f} (se {r['off_se']:.4f})  "
              + " ".join(f"c{c}:{r['clases'][c]['off_over_S']:+.3f}" for c in (0, 2, 4)), flush=True)
    del W, D2, d, off, Lam
json.dump(out, open(f"/home/claude/ariadna/R3/r3_exacto_2e{LOGX}.json", "w"), indent=1)
print("fin", round(time.time() - t0), "s")
