"""Box: pieces 2 sum_{h!=0} M E and -2 sum S nu delta of X, in units N^2 L^2 (line 468 of v4)."""
import numpy as np, math, json, sys
C2 = 0.66016181584686957
def sieve(M):
    isp = np.ones(M + 1, bool); isp[:2] = False
    for p in range(2, int(M ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]
kmax = int(sys.argv[1]); Nmax = 1 << kmax
P = sieve(2 * Nmax)
S = np.zeros(2 * Nmax + 1); S[0::2] = 2 * C2
for p in P[1:]: S[p::p] *= (p - 1) / (p - 2)
Lam = np.zeros(Nmax + 1)
for p in P[P <= Nmax]:
    lp = math.log(p); q = p
    while q <= Nmax: Lam[q] = lp; q *= p
res = {}
for k in range(14, kmax + 1):
    N = 1 << k; L = math.log(N); lam = Lam[:N + 1].copy(); lam[0] = 0
    size = 1 << (k + 2); F = np.fft.rfft(lam, size)
    R = np.fft.irfft(F * F, size)[:2 * N + 1]; psi2 = np.fft.irfft(F * np.conj(F), size)[:N]
    h = np.arange(1, N); M = S[h] * (N - h); E = psi2[1:N] - M
    n = np.arange(2, 2 * N + 1); nu = np.where(n <= N + 1, n - 1, 2 * N - n + 1).astype(float)
    delta = R[2:] - S[n] * nu
    p1 = 2 * 2 * float(np.sum(M * E)); p2 = -2 * float(np.sum(S[n] * nu * delta))
    u = N * N * L * L
    res[k] = dict(piece_ME=p1 / u, piece_Snd=p2 / u, X=(p1 + p2) / u)
    print(k, json.dumps({a: round(b, 4) for a, b in res[k].items()}), flush=True)
json.dump(res, open('xpieces.json', 'w'), indent=1)
