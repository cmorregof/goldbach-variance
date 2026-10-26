"""Direct check of the refined Lemma kernels(b):
 rho_w(h) = N r(h/N) - w(0)w(h/N)/2 + O(1/N),  nu_w(n) = N g(n/N) - w(0)w(n/N)1_{n<=N} + O(1/N).
Reports max over sampled h,n of N*|error| (should stay bounded) and of |error| without the correction (should be ~1/2, ~1)."""
import numpy as np, math
xg, wg = np.polynomial.legendre.leggauss(400)
def W(name, x):
    x = np.asarray(x, float); out = np.zeros_like(x); m = (x >= 0) & (x < 1)
    if name == 'lin': out[m] = 1 - x[m]
    elif name == 'sq': out[m] = (1 - x[m]) ** 2
    else: out[m] = np.exp(-x[m] / (1 - x[m]))
    return out
def integ(f, a, b):
    t = 0.5 * (b - a) * xg + 0.5 * (b + a); return 0.5 * (b - a) * np.sum(wg * f(t))
def r(name, u): return integ(lambda x: W(name, x) * W(name, x + u), 0.0, 1 - u) if u < 1 else 0.0
def g(name, u):
    lo, hi = max(0.0, u - 1), min(1.0, u)
    return integ(lambda y: W(name, u - y) * W(name, y), lo, hi) if hi > lo else 0.0
for name in ['lin', 'sq', 'exp']:
    for k in [10, 12, 14, 16]:
        N = 1 << k; m = np.arange(N + 1); wm = W(name, m / N); wm[0] = 0; wm[N] = 0
        size = 1 << (k + 2); F = np.fft.rfft(wm, size)
        nu = np.fft.irfft(F * F, size)[:2 * N + 1]; rho = np.fft.irfft(F * np.conj(F), size)[:N]
        w0 = W(name, [0.0])[0]
        hs = np.unique(np.linspace(1, N - 1, 400).astype(int)); ns = np.unique(np.linspace(2, 2 * N, 800).astype(int))
        er = [rho[h] - (N * r(name, h / N) - 0.5 * w0 * W(name, [h / N])[0]) for h in hs]
        er0 = [rho[h] - N * r(name, h / N) for h in hs]
        en = [nu[n] - (N * g(name, n / N) - w0 * W(name, [n / N])[0] * (n <= N)) for n in ns]
        en0 = [nu[n] - N * g(name, n / N) for n in ns]
        print(f"{name:4s} N=2^{k}: max N|rho err|={N*max(map(abs,er)):.4f}  max|rho err, no corr|={max(map(abs,er0)):.4f}  "
              f"max N|nu err|={N*max(map(abs,en)):.4f}  max|nu err, no corr|={max(map(abs,en0)):.4f}")
