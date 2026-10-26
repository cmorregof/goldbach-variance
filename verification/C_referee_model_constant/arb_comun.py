"""Árbitro C92: implementación INDEPENDIENTE de r, rho, t_p, T_e a partir de la tabla de E1 §1 / §4.
Sólo usa numpy.  Factores locales exactos para p <= 2n+2 (todos los p que pueden dividir h, n-h, n+h con |h|<n),
y cola analítica para los p > 2n+2 (donde todos los factores son genéricos)."""
import math
import numpy as np

def primes_upto(lim):
    s = np.ones(lim + 1, bool); s[:2] = False
    for p in range(2, int(lim ** 0.5) + 1):
        if s[p]: s[p*p::p] = False
    return np.nonzero(s)[0]

_PTAIL = primes_upto(1 << 23).astype(float)
def tail_sums(P0):
    """sum_{p>P0} log(generic r_p), log(generic rho_p), p>=5 only (P0 >= 5)."""
    q = _PTAIL[_PTAIL > P0]
    lr = float(np.sum(np.log1p(-4.0 / (q - 2) ** 2)))
    lq = float(np.sum(np.log1p(-2.0 / ((q - 1) * (q - 2)))))
    pm = _PTAIL[-1]
    # resto p > 2^23: aprox -4/(pm log pm), -2/(pm log pm)
    return lr - 4.0 / (pm * math.log(pm)), lq - 2.0 / (pm * math.log(pm))

def factor(m, primes):
    out = []
    for p in primes:
        p = int(p)
        if p * p > m: break
        if m % p == 0:
            out.append(p)
            while m % p == 0: m //= p
    if m > 1: out.append(m)
    return out

def r_rho(n):
    """r(n,h), rho(n,h) para h=1..n-1 (n par)."""
    assert n % 2 == 0
    H = n - 1
    h = np.arange(1, H + 1)
    primes = primes_upto(2 * n + 2)
    pn = set(factor(n, primes))
    lr = np.zeros(H); lq = np.zeros(H); zero = np.zeros(H, bool)
    # p = 2 | n: r_2 = 2 si 2|h, 0 si no
    zero |= (h % 2 == 1); lr[h % 2 == 0] += math.log(2); lq[h % 2 == 0] += math.log(2)
    # p = 3
    if 3 in pn:
        v = np.where(h % 3 == 0, math.log(1.5), math.log(0.75)); lr += v; lq += v
    else:
        zero |= (h % 3 != 0); lr[h % 3 == 0] += math.log(3); lq[h % 3 == 0] += math.log(3)
    for p in primes:
        p = int(p)
        if p < 5: continue
        if p in pn:
            # r_p = rho_p = p/(p-1) si p|h ; p(p-2)/(p-1)^2 si no
            v = np.where(h % p == 0, math.log(p / (p - 1)), math.log(p * (p - 2) / (p - 1) ** 2))
            lr += v; lq += v
        else:
            # r_p: p/(p-2) si p|h ; p(p-3)/(p-2)^2 si p | n -+ h ; p(p-4)/(p-2)^2 si no
            # rho_p: p/(p-2) si p|h ; p(p-3)/((p-1)(p-2)) si no
            gen_r = math.log(p * (p - 4) / (p - 2) ** 2)
            gen_q = math.log(p * (p - 3) / ((p - 1) * (p - 2)))
            vr = np.full(H, gen_r); vq = np.full(H, gen_q)
            near = ((n - h) % p == 0) | ((n + h) % p == 0)
            vr[near] = math.log(p * (p - 3) / (p - 2) ** 2)
            div = (h % p == 0)
            vr[div] = math.log(p / (p - 2)); vq[div] = math.log(p / (p - 2))
            lr += vr; lq += vq
    tr, tq = tail_sums(2 * n + 2)
    r = np.exp(lr + tr); q = np.exp(lq + tq)
    r[zero] = 0; q[zero] = 0
    return r, q

def c2(p): return 2.0 / ((p - 2) * (p - 3))

def t_p(p, n, h):
    h = np.asarray(h)
    if n % p == 0: return np.zeros(h.shape)
    out = np.full(h.shape, -c2(p))
    out[((n - h) % p == 0) | ((n + h) % p == 0)] = 1.0 / (p - 2)
    out[h % p == 0] = 0.0
    return out

def T_e(ps, n, h):
    out = np.ones(np.asarray(h).shape)
    for p in ps: out = out * t_p(p, n, h)
    return out

def gamma_e(ps):
    g = 1.0
    for p in ps: g *= 2.0 * p / ((p - 2) * (p - 3))
    return g

def w_smooth(x):
    x = np.asarray(x, float); out = np.zeros(x.shape); m = (x > 1) & (x < 2)
    y = x[m]; out[m] = np.exp(4.0 - 1.0 / ((y - 1) * (2 - y)))
    return out

C2 = 0.66016181584686957
def sing(n):
    s = 2 * C2
    for p in factor(n, primes_upto(int(n ** 0.5) + 2)):
        if p > 2: s *= (p - 1) / (p - 2)
    return s
