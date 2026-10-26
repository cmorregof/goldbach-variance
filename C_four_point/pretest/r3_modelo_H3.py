"""R3, ad-hoc variant H3 [H] suggested by the band diagnostic (NOT a derived model):
'nearest-diagonal' assignment: a pair (a,a') with shifts h = a'-a, k = a+a'-n carries
    Cov = S^2 (r(n,h) - 1)  if |h| < |k|,     S^2 (r(n,k) - 1)  if |k| < |h|      (ties: either; O(n) points)
i.e. each dip is cut off where the other diagonal becomes closer.  By symmetry
    off_H3(n) = S^2 [ 4 sum_{h=1}^{n/2} (r(h)-1) c(h) - 2n + 3 ],   c(h) = #{m on line h : |k(m)| > h}.
Same sampling/seed as r3_modelo_lineal.py.  Usage: python3 r3_modelo_H3.py KMIN KMAX NSAMP
"""
import sys, json, numpy as np
import r3_common as C
KMIN, KMAX, NSAMP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(20261025)
NMAX = 1 << (KMAX + 1)
primes = C.sieve(max(2 * NMAX, 1 << 25))
gen = lambda p: np.log1p(-4.0 / (p - 2) ** 2)
c1 = lambda p: np.log(p / (p - 2)) - gen(p)
c3 = lambda p: np.log(p * (p - 3) / (p - 2) ** 2) - gen(p)
B5 = C.euler_sum(primes, gen, 5, -4.0)
G1 = C.additive(2 * NMAX, primes, c1); G3 = C.additive(2 * NMAX, primes, c3)
S = C.singular_S(NMAX, primes)


def rline(n):
    h = np.arange(1, n)
    oddp = C.odd_prime_factors(n, primes)
    L0 = B5 + sum(np.log1p(-1.0 / (p - 1) ** 2) - gen(float(p)) for p in oddp if p >= 5)
    lr = L0 + G1[1:n] + G3[n - h] + G3[n + h]
    for p in oddp:
        if p >= 5:
            lr[p - 1::p] += np.log(p / (p - 1)) - np.log1p(-1.0 / (p - 1) ** 2) - c1(float(p)) - 2 * c3(float(p))
    R = np.exp(lr) * np.where(h % 2 == 0, 2.0, 0.0)
    R *= np.where(h % 3 == 0, 1.5, 0.75) if n % 3 == 0 else np.where(h % 3 == 0, 3.0, 0.0)
    return R


rows = []
for k in range(KMIN, KMAX + 1):
    lo, hi = 1 << k, 1 << (k + 1)
    ns = np.unique(2 * rng.integers(lo // 2, hi // 2, NSAMP))
    num = 0.0; base = 0.0; cl = {c: [0.0, 0.0] for c in (0, 2, 4)}
    for n in ns:
        n = int(n); s = S[n]; R = rline(n)
        h = np.arange(1, n // 2 + 1)
        # line h (h>0): k = 2m+h-n, m = 1..n-1-h  -> k in [h+2-n, n-h-2] step 2 ; count |k| > h
        kmax = n - h - 2
        # number of k of the parity of h with h < k <= kmax is floor((kmax-h)/2); times 2 for the negative side
        cnt = np.where(kmax > h, 2 * ((kmax - h) // 2), 0)
        F = 4.0 * np.dot(R[:len(h)] - 1.0, cnt)
        off = s * s * (F - 2 * n + 3)
        b = s * n * np.log(n) ** 2
        num += off; base += b; cl[n % 6][0] += off; cl[n % 6][1] += b
    r = dict(k=k, H3=num / base, clases={c: cl[c][0] / cl[c][1] for c in cl if cl[c][1] > 0})
    rows.append(r)
    print(f"k={k:2d}  H3={r['H3']:+.4f}  " + " ".join(f"c{c}:{v:+.3f}" for c, v in r['clases'].items()), flush=True)
json.dump(rows, open(f"/home/claude/ariadna/R3/r3_modelo_H3_k{KMIN}-{KMAX}.json", "w"), indent=1)
