"""E1_lemas.py -- checks of the elementary lemmas used in E1_intento.md.
  (L3) double Abel summation, difference and sum kernels, with random a, b and a kernel h supported on [k0, N-1]
       (resp. h' on [k0, 2N]):  sum a_m b_n h(n-m) = - sum_{m<=N-1, n<=2N} A(m) B(n) D2h(n-m),
                                sum a_m b_n h'(m+n) = sum_{m,n<=2N} A(m) B(n) D2h'(m+n+1)   (A, B frozen beyond N).
  (L4) size of sum_k |D^2 h_l(k)| for h_l(k) = (l/k) Theta(k/l) rho_w(k), against N/(delta Y^2 l) + 1/Y.
  (L1) beta(2m) = (4 C2/m) sum_{e|m} gamma(e) and partial sums of |gamma(e)| e^{-sigma}.
Output: E1_lemas.json
"""
import json, math, numpy as np
rng = np.random.default_rng(1); OUT = {}
# ---------------- L3
N = 300; k0 = 7
a = np.zeros(N + 1); b = np.zeros(N + 1); a[1:] = rng.normal(size=N); b[1:] = rng.normal(size=N)
h = np.zeros(3 * N + 5); h[k0:N] = rng.normal(size=N - k0)            # h(k), index k >= 0; h(k)=0 for k<=0 and k>=N
lhs = sum(a[m] * b[n] * h[n - m] for m in range(1, N + 1) for n in range(m + 1, N + 1))
A = np.concatenate([np.cumsum(a), np.full(2 * N + 5, a.sum())]); B = np.concatenate([np.cumsum(b), np.full(2 * N + 5, b.sum())])
def hk(k): return h[k] if 0 <= k < len(h) else 0.0
def d2(f, k): return f(k + 1) - 2 * f(k) + f(k - 1)
rhs = -sum(A[m] * B[n] * d2(hk, n - m) for m in range(1, N) for n in range(1, 2 * N + 2))
hp = np.zeros(4 * N + 8); hp[k0:2 * N + 1] = rng.normal(size=2 * N + 1 - k0)
def hpk(k): return hp[k] if 0 <= k < len(hp) else 0.0
lhs2 = sum(a[m] * b[n] * hpk(m + n) for m in range(1, N + 1) for n in range(1, N + 1))
rhs2 = sum(A[m] * B[n] * d2(hpk, m + n + 1) for m in range(1, 2 * N + 1) for n in range(1, 2 * N + 1))
OUT['L3'] = dict(diff_kernel=[lhs, rhs], sum_kernel=[lhs2, rhs2])
print('L3 difference kernel:', lhs, rhs, '  sum kernel:', lhs2, rhs2)
# ---------------- L4
def smoothstep(u):
    u = np.clip(u, 0, 1); return u ** 3 * (10 - 15 * u + 6 * u * u)
res = []
for kN in [14, 18]:
    N = 2 ** kN; x = np.arange(N + 1) / N; w = 1 - x; w[0] = 0
    F = np.fft.rfft(w, 4 * N); rho = np.fft.irfft(F * np.conj(F), 4 * N)[:N + 3]; rho[N:] = 0
    for eps, delta in [(0.1, 0.05), (0.2, 0.05), (0.2, 0.2)]:
        Y = N ** (0.5 + eps); mx = 0.0
        for l in [1, 2, 5, int(N / Y)]:
            k = np.arange(N + 3, dtype=float); k[0] = 1
            hl = (l / k) * smoothstep((k / l - Y) / (delta * Y)) * rho; hl[0] = 0
            s = np.abs(hl[2:] - 2 * hl[1:-1] + hl[:-2]).sum()
            pred = N / (delta * Y * Y * l) + 1 / Y
            res.append(dict(N=N, eps=eps, delta=delta, l=l, sumD2=s, pred=pred, ratio=s / pred)); mx = max(mx, s / pred)
        print(f'L4 N=2^{kN} eps={eps} delta={delta}: max ratio sum|D2 h_l| / (N/(delta Y^2 l)+1/Y) = {mx:.3f}')
OUT['L4'] = res
# ---------------- L1
C2 = 0.660161815846869573927812110015
M = 20000
def factor(n):
    f = {}; p = 2
    while p * p <= n:
        while n % p == 0: f[p] = f.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: f[n] = f.get(n, 0) + 1
    return f
def gamma(e):
    g = 1.0
    for p, k in factor(e).items():
        if p == 2: g *= (-1.0 if k == 1 else 0.0)
        else: g *= {1: 2 / (p - 2), 2: -p / (p - 2)}.get(k, 0.0)
    return g
def beta(d):
    if d % 2: return 0.0
    f = factor(d)
    if any(k > 1 for k in f.values()): return 0.0
    out = 4 * C2
    for p in f:
        if p > 2: out /= (p - 2)
    return out
G = np.array([0.0] + [gamma(e) for e in range(1, M + 1)])
err = 0.0
for m in range(1, 3000):
    s = sum(G[e] for e in range(1, m + 1) if m % e == 0)
    err = max(err, abs(beta(2 * m) - 4 * C2 / m * s))
print('L1 max |beta(2m) - (4C2/m) sum_{e|m} gamma(e)|, m<3000:', err)
e = np.arange(1, M + 1)
part = {str(E): dict(sum_abs=float(np.abs(G[1:E + 1]).sum()), sum_abs_over_sqrtE=float(np.abs(G[1:E + 1]).sum() / math.sqrt(E)),
                     tail_abs_over_e=float((np.abs(G[E + 1:]) / e[E:]).sum()))
        for E in [10, 100, 1000, 10000]}
print('L1 partial sums of |gamma|:', part)
OUT['L1'] = dict(maxerr=err, partial=part)
json.dump(OUT, open('/home/claude/ariadna/R1/run2_E1/E1_lemas.json', 'w'), indent=1)
