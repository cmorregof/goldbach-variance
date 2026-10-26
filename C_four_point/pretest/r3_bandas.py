"""R3 diagnostic: split the off-diagonal (covariance form) into
   hband(t) = pairs with 0<|h|<=t n,  kband(t) = pairs with 0<|k|<=t n (k = a+a'-n),
   diamond(t) = both,  bulk(t) = total - hband - kband + diamond,
for the EXACT data [M] and for models H1, H2 [H] at the same n (all even n in the block, or a stride).
Exact symmetry a' -> n-a' maps hband onto kband, so exact hband = kband (checked).
Normalisation: sum over n / sum S n log^2 n.
Usage: python3 r3_bandas.py K [STRIDE]
"""
import sys, json, time, numpy as np, scipy.fft as sf
import r3_common as C
K = int(sys.argv[1]); STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 1
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
    return np.concatenate([[np.nan], R])          # R[h] = r(n,h), h = 0..n-1


TS = [0.02, 0.05, 0.1, 0.2]
acc = {nm: {t: np.zeros(4) for t in TS} for nm in ("exact", "H1", "H2")}   # [total, hband, kband, diamond]
base = 0.0; checks = []
for n in range(lo, hi, 2 * STRIDE):
    s = S[n]
    Y = np.zeros(n + 1); Y[1:n] = Lam[1:n] * Lam[n - 1:0:-1] - s        # Y[a], a = 1..n-1
    a = np.arange(1, n)
    F = sf.rfft(Y, 2 * n + 2)
    ac = sf.irfft(F * np.conj(F), 2 * n + 2)                            # ac[h] = sum_a Y_a Y_{a+h}
    cv = sf.irfft(F * F, 2 * n + 2)                                     # cv[j] = sum_a Y_a Y_{j-a}
    R = rline(n)
    # exact: off pairs exclude h=0 (a'=a) and k=0 (a'=n-a)
    hline = np.zeros(n); hline[1:n - 1] = ac[1:n - 1]                   # h>0 line sums (incl. its k=0 point)
    kline = np.zeros(n); kline[1:n - 1] = cv[n + 1:2 * n - 1]           # k>0: a+a' = n+k  (incl. its h=0 point)
    hh = np.arange(n)
    # remove from each h-line its k=0 point (a=(n-h)/2, h even) and from each k-line its h=0 point (a=(n+k)/2, k even)
    ev = (hh % 2 == 0) & (hh >= 2) & (hh <= n - 2)
    hline[ev] -= Y[(n - hh[ev]) // 2] ** 2
    kline[ev] -= Y[(n + hh[ev]) // 2] ** 2
    tot_h = 2 * hline.sum()                                              # all off pairs (h != 0, k != 0)
    checks.append(abs(2 * kline.sum() - tot_h) / (abs(tot_h) + 1e-300))
    # model line sums: point value S^2[(r(h)-1) + (r(k)-1)] (H2) ; S^2 (r(h)-1) (H1)
    w = np.zeros(n); w[1:n - 1] = (n - 1 - hh[1:n - 1]) - ev[1:n - 1]   # points per h-line after removing k=0
    r1 = np.zeros(n); r1[1:] = R[1:] - 1.0
    # sum over points of line h of (r(k)-1): k = 2m+h-n, m = 1..n-1-h, k != 0
    rk1 = np.concatenate([r1[:0:-1], [0.0], r1[1:]])                       # index k+(n-1)
    cs = [np.concatenate([[0.0], np.cumsum(np.where(np.arange(-(n - 1), n) % 2 == par, rk1, 0.0))]) for par in (0, 1)]
    k0 = 2 + hh - n; k1 = n - hh - 2
    Mk = np.where(hh % 2 == 0, cs[0][np.clip(k1 + n, 0, 2 * n - 1)] - cs[0][np.clip(k0 + n - 1, 0, 2 * n - 1)],
                  cs[1][np.clip(k1 + n, 0, 2 * n - 1)] - cs[1][np.clip(k0 + n - 1, 0, 2 * n - 1)])
    Mk[0] = 0.0; Mk[n - 1:] = 0.0
    lineH1 = s * s * w * r1; lineH2 = lineH1 + s * s * Mk
    for t in TS:
        T = int(t * n)
        # diamond: |h|<=T and |k|<=T, h,k != 0 ; on line h>0: m with |2m+h-n|<=T, k != 0
        dE = 0.0; dM1 = 0.0; dM2 = 0.0
        for h_ in range(1, T + 1):
            kk = np.arange(-T + ((T + h_ + n) % 2), T + 1, 2)            # k with parity of h (k = 2m+h-n)
            kk = kk[kk != 0]
            m = (kk + n - h_) // 2
            m = m[(m >= 1) & (m + h_ <= n - 1)]
            kk = 2 * m + h_ - n
            dE += (Y[m] * Y[m + h_]).sum()
            dM1 += s * s * len(m) * r1[h_]
            dM2 += s * s * (len(m) * r1[h_] + r1[np.abs(kk)].sum())
        hb = {"exact": 2 * hline[1:T + 1].sum(), "H1": 2 * lineH1[1:T + 1].sum(), "H2": 2 * lineH2[1:T + 1].sum()}
        # kband = mirror of hband for exact and H2 (both symmetric); for H1 compute directly: k-line points carry r(h)-1
        # H1 on k-line k: points a with a+a'=n+k, h = a'-a = n+k-2a  -> sum of r(|h|)-1 over that segment
        kbH1 = 0.0
        for k_ in range(1, T + 1):
            aa = np.arange(max(1, k_ + 1), n)                            # a' = n+k-a in [1,n-1] -> a in [k+1, n-1]
            hv = n + k_ - 2 * aa
            hv = hv[hv != 0]
            kbH1 += s * s * r1[np.abs(hv)].sum()
        kb = {"exact": hb["exact"], "H1": 2 * kbH1, "H2": hb["H2"]}
        dm = {"exact": 2 * dE, "H1": 2 * dM1, "H2": 2 * dM2}
        totv = {"exact": tot_h, "H1": 2 * lineH1.sum(), "H2": 2 * lineH2.sum()}
        for nm in acc:
            acc[nm][t] += np.array([totv[nm], hb[nm], kb[nm], dm[nm]])
    base += s * n * np.log(n) ** 2
out = dict(K=K, stride=STRIDE, sym_check_max=float(max(checks)), res={})
print(f"K={K} stride={STRIDE}  symmetry check (exact hline vs kline totals) max rel diff = {max(checks):.1e}  [{round(time.time() - t0)} s]")
print("   t   model    total    hband    kband   diamond    bulk")
for t in TS:
    out["res"][t] = {}
    for nm in acc:
        tot, hb, kb, dm = acc[nm][t] / base
        out["res"][t][nm] = dict(total=tot, hband=hb, kband=kb, diamond=dm, bulk=tot - hb - kb + dm)
        print(f"{t:5.2f}  {nm:5s}  {tot:+.4f}  {hb:+.4f}  {kb:+.4f}  {dm:+.4f}  {tot - hb - kb + dm:+.4f}")
json.dump(out, open(f"/home/claude/ariadna/R3/r3_bandas_K{K}.json", "w"), indent=1)
