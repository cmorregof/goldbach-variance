"""arb_core.py -- independent referee computations for E1_intento.md (referee run 1).

Everything is recomputed from scratch (own sieve, own FFT convolutions, own multiple-sums).
Normalisation of manuscrito_v5_en.tex, Lemma dbasis:
  lambda = Lambda * w,  rho_w(h) = sum_m w_m w_{m+h},  nu_w(n) = sum_{a+b=n} w_a w_b,
  psi2^w(h) = sum_m lambda(m) lambda(m+h),  R_w(n) = sum_{a+b=n} lambda(a) lambda(b),
  E_w = psi2^w - S rho_w (h != 0),  delta_w = R_w - S nu_w,
  Delta_w(d) = 2 sum_{h>=1, d|h} rho_w E_w - sum_{n>=1, d|n} nu_w delta_w.
"""
import math, json, sys, time
import numpy as np

C2 = 0.6601618158468695739278121100145557784326233602847334133194484233354056423


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if s[p]: s[p * p::p] = False
    return np.flatnonzero(s)


def mangoldt(N):
    Lam = np.zeros(N + 1)
    for p in primes_upto(N):
        q = int(p); lp = math.log(p)
        while q <= N:
            Lam[q] = lp; q *= int(p)
    return Lam


def sing_series(K, pr):
    """S[n] = singular series for 0<n<=K (0 for odd n and n=0)."""
    S = np.zeros(K + 1); S[2::2] = 2 * C2
    for p in pr:
        if p == 2: continue
        if p > K: break
        S[p::p] *= (p - 1) / (p - 2)
    S[0] = 0.0
    return S


def beta_arr(K, pr):
    """beta(d) for d<=K: 4C2 prod_{p|d/2}(p-2)^{-1} for even squarefree d, else 0."""
    b = np.zeros(K + 1); b[2::2] = 4 * C2
    for p in pr:
        if p > K: break
        if p > 2: b[2 * p::2 * p] /= (p - 2)
        if p * p <= K: b[p * p::p * p] = 0.0
    return b


def gamma_arr(K, pr):
    """gamma = b * mu, b(m)=mu^2(m) 1_{m odd} prod_{p|m} p/(p-2)."""
    g = np.ones(K + 1); g[0] = 0
    g[2::2] *= -1.0
    if K >= 4: g[4::4] = 0.0
    for p in pr:
        if p == 2: continue
        if p > K: break
        g[p::p] *= 2.0 / (p - 2)
        if p * p <= K:
            g[p * p::p * p] *= -p / 2.0
            if p ** 3 <= K: g[p ** 3::p ** 3] = 0.0
    return g


def mult_sums(x, Dmax):
    """F[d] = sum_{j>=1} x[j d], 1<=d<=Dmax (x[0] ignored). Straightforward loop."""
    K = len(x) - 1; F = np.zeros(Dmax + 1)
    for d in range(1, min(Dmax, K) + 1):
        F[d] = x[d::d].sum()
    return F


def conv_data(N, w):
    """w: array of length N+1 with w[0]=0, w[N]=0. Returns dict of kernels."""
    L = 1
    while L < 2 * N + 2: L *= 2
    Lam = mangoldt(N); lam = Lam * w
    Fl = np.fft.rfft(lam, L); Fw = np.fft.rfft(w, L)
    R = np.fft.irfft(Fl * Fl, L)[:2 * N + 1]
    psi2 = np.fft.irfft(Fl * np.conj(Fl), L)[:N + 1]
    nu = np.fft.irfft(Fw * Fw, L)[:2 * N + 1]
    rho = np.fft.irfft(Fw * np.conj(Fw), L)[:N + 1]
    pr = primes_upto(2 * N + 2)
    S = sing_series(2 * N, pr)
    E = psi2 - S[:N + 1] * rho; E[0] = 0.0
    dlt = R - S * nu
    return dict(N=N, Lam=Lam, lam=lam, w=w, R=R, psi2=psi2, nu=nu, rho=rho, S=S, E=E, dlt=dlt, pr=pr)


def weight(name, N):
    x = np.arange(N + 1) / N
    if name == '1-x': w = 1 - x
    elif name == '(1-x)^2': w = (1 - x) ** 2
    elif name == 'cos': w = np.cos(np.pi * x / 2)
    elif name == '1-x^2': w = 1 - x ** 2
    else: raise ValueError(name)
    w = w.copy(); w[0] = 0.0; w[N] = 0.0
    return w


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)
