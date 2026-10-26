"""R3 pre-test, model (B) [H]: block-averaged Hardy-Littlewood 4-point model (the coordinator's
literal route: shifts {0, h, k-h, k}, m free, n = 2m + k runs over the block).

Parametrisation of the off-diagonal pairs at even n:  a = m, a' = m + h, n - a = m + k, n - a' = m + k - h,
   so k = n - 2a (even), h = a' - a.  For fixed (h, k) the four LINEAR FORMS IN m have slope 1:
   standard prime k-tuple with shift set  H = {0, h, k-h, k}.
Degenerate (h,k) removed:
   h = 0      (a' = a)        -> in D2
   h = k      (a' = n - a)    -> in D2
   k = 0      (a = n/2)       -> row with Lam(n/2)^2; Cov ~ O(log^3 n) per n, dropped
   k = 2h     (a' = n/2)      -> idem
Local factors, p odd: nu_p = 1 if p | (h,k); 2 if p divides exactly one of h, k-h; 3 if exactly one of k, k-2h;
   4 otherwise (impossible for p = 3).  f_p(nu) = (1 - nu/p)/(1 - 1/p)^4.  p = 2 (k even): f_2 = 8 (h even), 0 (h odd).
Classes n = c mod 6: m restricted to m = 2(c - k) mod 3; then the p = 3 factor is the exact indicator
   1[3 does not divide m(m+h)(m+k-h)(m+k)] * (3/2)^4, and the count is over that residue class.
Model (covariance form):
   sum_{n in B, class} off_model(n) = sum_{(h,k) nondeg} [ SS4(h,k) * #m  -  sum_m S(2m+k)^2 ]  -  sum_n 2 n S(n)^2
Normalisation (ratio of sums, as the exact script):  off/S := sum off / sum S n log^2 n.
Usage: python3 r3_modelo_2d.py K [K ...]     (block [2^K, 2^(K+1)))
"""
import sys, json, time, numpy as np
from r3_common import sieve, singular_S, additive, euler_sum

t0 = time.time()
Ks = [int(a) for a in sys.argv[1:]]
HI = 1 << (max(Ks) + 1)
primes = sieve(max(4 * HI, 1 << 24))
f = lambda p, nu: np.log1p(-nu / p) - 4 * np.log1p(-1.0 / p)
L4 = euler_sum(primes, lambda p: f(p, 4), 5, -6.0)
c2 = lambda p: f(p, 2) - f(p, 4)
c3 = lambda p: f(p, 3) - f(p, 4)
g1 = lambda p: f(p, 1) - f(p, 4) - 2 * c2(p) - 2 * c3(p)
C2a = additive(3 * HI, primes, c2); C3a = additive(3 * HI, primes, c3); G1a = additive(HI, primes, g1)
S = singular_S(HI, primes)
nn = np.arange(HI + 1, dtype=float)
F3 = (1.5) ** 4
print("setup", round(time.time() - t0, 1), "s, L4 =", L4, flush=True)

res = {}
for K in Ks:
    lo, hi = 1 << K, 1 << (K + 1)
    # prefix sums of S(n)^2 over even n of class c (n = c mod 6), n < hi
    Q = {}
    for c in (0, 2, 4):
        v = np.where((np.arange(HI + 1) % 6 == c) & (np.arange(HI + 1) >= lo) & (np.arange(HI + 1) < hi), S ** 2, 0.0)
        Q[c] = np.concatenate([[0.0], np.cumsum(v)])          # Q[c][x+1] = sum_{n<=x}
    model = {c: 0.0 for c in (0, 2, 4)}; mainA = {c: 0.0 for c in (0, 2, 4)}; mainB = {c: 0.0 for c in (0, 2, 4)}
    h = np.arange(-(hi - 2), hi - 1)
    for k in range(-(hi - 4), hi - 3, 2):
        if k == 0: continue
        ok = (h != 0) & (h != k) & (2 * h != k)
        A = np.maximum.reduce([np.ones_like(h), 1 - h, 1 - k + h, np.full_like(h, 1 - k), np.full_like(h, (lo - k) // 2 + ((lo - k) % 2))])
        B = (hi - 2 - k) // 2
        ok &= (B >= A)
        if not ok.any(): continue
        hh = h[ok]; Aa = A[ok]
        lS = L4 + C2a[np.abs(hh)] + C2a[np.abs(k - hh)] + C3a[abs(k)] + C3a[np.abs(k - 2 * hh)] + G1a[np.gcd(np.abs(hh), abs(k))]
        S4 = np.where(hh % 2 == 0, 8.0, 0.0) * np.exp(lS)
        for c in (0, 2, 4):
            r = (2 * (c - k)) % 3
            cnt = (B - r) // 3 - (Aa - 1 - r) // 3
            f3 = ((r % 3 != 0) & ((r + hh) % 3 != 0) & ((r + k - hh) % 3 != 0) & ((r + k) % 3 != 0)) * F3
            nlo = 2 * Aa + k; nhi = 2 * B + k
            SS = Q[c][nhi + 1] - Q[c][nlo]
            a_ = (S4 * f3 * cnt).sum(); b_ = SS.sum()
            model[c] += a_ - b_; mainA[c] += a_; mainB[c] += b_
    out = {}
    tot_off = 0.0; tot_base = 0.0
    for c in (0, 2, 4):
        idx = np.arange(lo, hi); idx = idx[idx % 6 == c]
        s = S[idx]; n_ = nn[idx]; lg2 = np.log(n_) ** 2
        off = model[c] - (2 * n_ * s ** 2).sum()
        out[c] = dict(off_over_S=off / (s * n_ * lg2).sum(), off_over_S2=off / (s ** 2 * n_ * lg2).sum(),
                      cov_only_over_S=model[c] / (s * n_ * lg2).sum(),
                      main_rel_cancel=(mainA[c] - mainB[c]) / mainB[c])
        tot_off += off; tot_base += (s * n_ * lg2).sum()
    idx = np.arange(lo, hi, 2)
    res[K] = dict(K=K, meanlog=float(np.log(nn[idx]).mean()), off=tot_off / tot_base, clases=out)
    print(f"K={K} off_model={tot_off / tot_base:+.4f}  " + "  ".join(
        f"c{c}: off/S={out[c]['off_over_S']:+.4f} off/S2={out[c]['off_over_S2']:+.4f} (cancel {out[c]['main_rel_cancel']:+.2e})"
        for c in (0, 2, 4)) + f"  [{round(time.time() - t0)} s]", flush=True)
json.dump(dict(L4=L4, res=res), open(f"/home/claude/ariadna/R3/r3_modelo_2d_K{'-'.join(map(str, Ks))}.json", "w"), indent=1)
