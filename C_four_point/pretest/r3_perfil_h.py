"""R3 diagnostic: WHERE (in the shift h = a'-a) does the exact off-diagonal differ from model H2?
For every even n in the block (all of them, no sampling):
  exact  cov_n(h) = sum_{m on line h, m' != n-m} (X_m - S)(X_{m+h} - S)                      [M]
  model  P_n(h)   = S^2 [ w(h)(r(h)-1)  +  sum_{m on line h} (r(k(m)) - 1) ],  k(m) = 2m+h-n   [H2]
     (first term: h-family; second: the k-family evaluated on the points of line h)
Cumulative profiles Phi(t) = sum_{0<|h|<= t n} (.) / sum S n log^2 n, t in (0,1].
Usage: python3 r3_perfil_h.py K [STRIDE]   (STRIDE: use every STRIDE-th even n; default 1 = all)
"""
import sys, json, time, numpy as np, scipy.fft as sf
K = int(sys.argv[1]); STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
import r3_common as C
lo, hi = 1 << K, 1 << (K + 1)
t0 = time.time()
primes = C.sieve(max(4 * hi, 1 << 22))
Lam = C.von_mangoldt(hi, primes)
S = C.singular_S(hi, primes)
gen = lambda p: np.log1p(-4.0 / (p - 2) ** 2)
c1 = lambda p: np.log(p / (p - 2)) - gen(p)
c3 = lambda p: np.log(p * (p - 3) / (p - 2) ** 2) - gen(p)
B5 = C.euler_sum(primes, gen, 5, -4.0)
G1 = C.additive(2 * hi, primes, c1); G3 = C.additive(2 * hi, primes, c3)


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
    return R                       # R[h-1] = r(n,h), symmetric in h


tgrid = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
PhiE = np.zeros(len(tgrid)); PhiM = np.zeros(len(tgrid)); PhiM_h = np.zeros(len(tgrid)); base = 0.0
for n in range(lo, hi, 2 * STRIDE):
    s = S[n]
    X = Lam[1:n] * Lam[n - 1:0:-1] - s              # X_a - S, a = 1..n-1  (index a-1)
    F = sf.rfft(X, 2 * n); ac = sf.irfft(F * np.conj(F), 2 * n)[1:n - 1]   # h = 1..n-2 : sum_a Y_a Y_{a+h}
    h = np.arange(1, n - 1)
    # remove anti-diagonal point a = (n-h)/2 (h even): Y_a Y_{n-a} = Y_a^2
    ev = h % 2 == 0
    a_ad = (n - h[ev]) // 2
    ac[ev] -= X[a_ad - 1] ** 2
    covE = 2 * ac                                     # h and -h
    R = rline(n)
    r_h = R[:n - 2]
    w = (n - 1 - h).astype(float) - ev
    Mh = w * (r_h - 1)
    # k-family on line h: m = 1..n-1-h, k = 2m+h-n, k != 0 ; r(k) symmetric, r(0) excluded
    rk = np.concatenate([R[::-1], [0.0], R])          # index k + (n-1), k = -(n-1)..(n-1)
    rk1 = rk - 1.0; rk1[n - 1] = 0.0                  # k = 0 excluded
    cs = {}
    for par in (0, 1):
        v = np.where((np.arange(-(n - 1), n) % 2) == par, rk1, 0.0)
        cs[par] = np.concatenate([[0.0], np.cumsum(v)])
    k0 = 2 + h - n; k1 = n - h - 2
    Mk = np.where(h % 2 == 0, cs[0][k1 + n] - cs[0][k0 + n - 1], cs[1][k1 + n] - cs[1][k0 + n - 1])
    covM_h = 2 * s * s * Mh; covM = 2 * s * s * (Mh + Mk)
    cE = np.cumsum(covE); cM = np.cumsum(covM); cMh = np.cumsum(covM_h)
    idx = np.minimum((tgrid * (n - 2)).astype(int), n - 3)
    PhiE += cE[idx]; PhiM += cM[idx]; PhiM_h += cMh[idx]
    base += s * n * np.log(n) ** 2
res = dict(K=K, stride=STRIDE, t=tgrid.tolist(), exact=(PhiE / base).tolist(), H2=(PhiM / base).tolist(), H1=(PhiM_h / base).tolist())
json.dump(res, open(f"/home/claude/ariadna/R3/r3_perfil_h_K{K}.json", "w"), indent=1)
print(f"K={K}  ({round(time.time() - t0)} s)\n   t     exact     H2       H1     exact-H2")
for t, e, m, m1 in zip(tgrid, res['exact'], res['H2'], res['H1']):
    print(f"{t:6.3f}  {e:+.4f}  {m:+.4f}  {m1:+.4f}  {e - m:+.4f}")
