"""R3 pre-test, model (A) [H]: fixed-n Hardy-Littlewood 4-point model.

For fixed even n and shift h = a' - a, the four linear forms in m are
      m, m+h, n-m, n-m-h      (slopes +1,+1,-1,-1; distinct as forms iff h != 0)
nu_p(h) = #{0, -h, n, n-h} mod p, and   SS4(n,h) = prod_p (1 - nu_p/p)/(1 - 1/p)^4 = S(n)^2 r(n,h),
      r_p = (1 - nu_p/p) / (1 - nu'_p/p)^2 ,  nu'_p = #{0, n} mod p.
  p = 2           : r = 2 (h even), 0 (h odd)
  p = 3, 3 | n    : r = 3/2 (3|h), 3/4 otherwise
  p = 3, 3 ∤ n    : r = 3 (3|h), 0 otherwise       (h = +-n mod 3 kills a residue class)
  p >= 5, p | n   : r = p/(p-1) (p|h), 1 - 1/(p-1)^2 otherwise
  p >= 5, p ∤ n   : r = p/(p-2) (p|h),  p(p-3)/(p-2)^2 (p | n-h or p | n+h),  1 - 4/(p-2)^2 otherwise
Product over ALL primes (exact factors for p <= 2^25 + analytic tail).

Model of the off-diagonal (covariance form; E X_a = S(n)):
  off_model(n) = sum_{off pairs} Cov(X_a, X_a') - 2 sum_a (E X_a)^2 + (E d)^2
               = S(n)^2 [ F(n) - 2n + 3 ]  (+ O(log^4 n)),
  H1 ('h-lines only', literal):  F = M1(n) := sum_{0<|h|<n} w(h) (r(n,h) - 1)
  H2 ('dihedral-symmetrised') :  F = 2 M1(n)   (adds the identical k-line family, k = a+a'-n)
  w(h) = n-1-|h| - [h even, |h| <= n-2]   (the anti-diagonal point a' = n-a removed: it is in D2).
Usage: python3 r3_modelo_lineal.py KMIN KMAX NSAMP [seed]
"""
import sys, json, time, numpy as np
from r3_common import sieve, singular_S, additive, euler_sum, odd_prime_factors

KMIN, KMAX, NSAMP = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261025
rng = np.random.default_rng(seed)
t0 = time.time()
NMAX = 1 << (KMAX + 1)
primes = sieve(max(2 * NMAX, 1 << 25))
gen = lambda p: np.log1p(-4.0 / (p - 2) ** 2)
c1 = lambda p: np.log(p / (p - 2)) - gen(p)
c3 = lambda p: np.log(p * (p - 3) / (p - 2) ** 2) - gen(p)
B5 = euler_sum(primes, gen, 5, -4.0)
G1 = additive(2 * NMAX, primes, c1)
G3 = additive(2 * NMAX, primes, c3)
Sarr = singular_S(NMAX, primes)
print("setup", round(time.time() - t0, 1), "s; B5 =", B5, flush=True)


def M1(n):
    h = np.arange(1, n)
    oddp = odd_prime_factors(n, primes)
    L0 = B5
    for p in oddp:
        if p >= 5:
            L0 += np.log1p(-1.0 / (p - 1) ** 2) - gen(float(p))
    lr = L0 + G1[1:n] + G3[n - h] + G3[n + h]
    for p in oddp:
        if p >= 5:
            lr[p - 1::p] += np.log(p / (p - 1)) - np.log1p(-1.0 / (p - 1) ** 2) - c1(float(p)) - 2 * c3(float(p))
    R = np.exp(lr)
    R *= np.where(h % 2 == 0, 2.0, 0.0)
    if n % 3 == 0:
        R *= np.where(h % 3 == 0, 1.5, 0.75)
    else:
        R *= np.where(h % 3 == 0, 3.0, 0.0)
    w = (n - 1 - h).astype(float) - ((h % 2 == 0) & (h <= n - 2))
    return 2.0 * np.dot(w, R - 1.0), R


rows = []
for k in range(KMIN, KMAX + 1):
    lo, hi = 1 << k, 1 << (k + 1)
    ns = np.unique(2 * rng.integers(lo // 2, hi // 2, NSAMP))
    rec = []
    for n in ns:
        n = int(n); s = Sarr[n]
        m1, R = M1(n)
        rec.append((n, s, m1, R.mean() if n < (1 << 16) else np.nan))
    rec = np.array(rec)
    nn, s, m1 = rec[:, 0], rec[:, 1], rec[:, 2]
    lg2 = np.log(nn) ** 2
    H1 = s ** 2 * (m1 - 2 * nn + 3); H2 = s ** 2 * (2 * m1 - 2 * nn + 3)
    base = (s * nn * lg2).sum()
    r = dict(k=k, nsamp=len(nn), meanlog=float(np.log(nn).mean()),
             H1=float(H1.sum() / base), H2=float(H2.sum() / base),
             dip_per_family=float((s ** 2 * m1).sum() / base),          # S^2 M1 / (S n log^2 n)
             dip_over_S_per_family=float(((s * m1) / (nn * lg2)).mean()),  # = M1 S/(n log^2): should -> -1/2
             crossterm=float((-2 * s ** 2 * nn).sum() / base))
    # spread across n of the normalised value (for honesty about sampling)
    zz = (s * m1) / (nn * lg2)
    r["dip_over_S_sd"] = float(zz.std())
    cls = {}
    for c in (0, 2, 4):
        m = (nn % 6) == c
        if m.sum() == 0: continue
        cls[c] = dict(n=int(m.sum()),
                      H1_over_S=float(H1[m].sum() / (s[m] * nn[m] * lg2[m]).sum()),
                      H2_over_S=float(H2[m].sum() / (s[m] * nn[m] * lg2[m]).sum()),
                      H2_over_S2=float(H2[m].sum() / (s[m] ** 2 * nn[m] * lg2[m]).sum()))
    r["clases"] = cls
    rows.append(r)
    print(f"k={k:2d} ns={len(nn):3d} H1={r['H1']:+.4f} H2={r['H2']:+.4f}  S*M1/(n log^2)={r['dip_over_S_per_family']:+.4f}"
          f" (mean {r['dip_over_S_per_family']:+.4f}, sd {r['dip_over_S_sd']:.4f}) cross={r['crossterm']:+.4f}  "
          + " ".join(f"c{c}:{cls[c]['H2_over_S']:+.3f}/{cls[c]['H2_over_S2']:+.3f}" for c in cls)
          + f"  [{round(time.time() - t0)} s]", flush=True)
json.dump(dict(B5=B5, seed=seed, rows=rows),
          open(f"/home/claude/ariadna/R3/r3_modelo_lineal_k{KMIN}-{KMAX}.json", "w"), indent=1)
