"""run6_E1: utilidades exactas para R3-E1 (25 oct 2026).

r(n,h) = prod_p r_p(h), con r_p como en R3_pretest §2.1:
  p | n (incluye p=2):  r_p = p/(p-1) si p|h ;  p(p-2)/(p-1)^2 si no.          (p=2: 2 / 0)
  p ∤ n (p impar)    :  r_p = p/(p-2) si p|h ;  p(p-3)/(p-2)^2 si p | n∓h ;  p(p-4)/(p-2)^2 si no.
Producto sobre TODOS los primos: factores exactos p <= PMAX (PMAX >= 2n+2, así que ningún p > PMAX
divide h(n^2-h^2) para 0<|h|<n) más cola analítica sum_{p>PMAX} log(1-4/(p-2)^2) ≈ -4/(PMAX log PMAX).

rho(n,h) = prod_p rho_p(h): la "parte en h" (media condicional de r_p dado 1_{p|h}):
  p | n : rho_p = r_p ;   p ∤ n : rho_p = p/(p-2) si p|h ;  p(p-3)/((p-1)(p-2)) si no.
g multiplicativa: g(p) = r_p(0) - 1 = 1/(p-1) (p|n, incluye g(2)=1), 2/(p-2) (p∤n).
"""
import numpy as np

C2 = 0.66016181584686957392781211001455577843262336
EULER = 0.57721566490153286060651209008240243104215933593992


def sieve(lim):
    isp = np.ones(lim + 1, bool); isp[:2] = False
    for p in range(2, int(lim ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0]


_PR = {}


def primes_upto(lim):
    lim = max(lim, 1 << 16)
    for k, v in _PR.items():
        if k >= lim:
            return v[v <= lim]
    v = sieve(lim); _PR[lim] = v
    return v


def prime_factors(m, primes):
    out = []
    for p in primes:
        p = int(p)
        if p * p > m: break
        if m % p == 0:
            out.append(p)
            while m % p == 0: m //= p
    if m > 1: out.append(m)
    return out


def sing(n, primes):
    s = 2 * C2
    for p in prime_factors(n, primes):
        if p > 2: s *= (p - 1) / (p - 2)
    return s


def r_and_rho(n, H, primes=None):
    """arrays r[h], rho[h] for h = 1..H (index h-1), H <= n-1, n even."""
    assert n % 2 == 0 and 1 <= H <= n - 1
    PMAX = 2 * n + 2
    if primes is None: primes = primes_upto(max(PMAX, 1 << 20))
    ps = primes[primes <= max(PMAX, 1 << 20)]
    P = float(ps[-1])
    pf = set(prime_factors(n, ps))
    h = np.arange(1, H + 1)
    lr = np.zeros(H); lrho = np.zeros(H)
    # base: generic factor at every prime
    PB = max(PMAX, 1 << 24)
    pb = primes_upto(PB).astype(float); P = pb[-1]
    pb5 = pb[pb >= 5]
    base_r = float(np.sum(np.log1p(-4.0 / (pb5 - 2) ** 2)))
    base_rho = float(np.sum(np.log1p(-2.0 / ((pb5 - 1) * (pb5 - 2)))))
    for p in pf:
        if p >= 5:   # replace generic p∤n factor by the p|n factor
            g = np.log(p * (p - 2) / (p - 1) ** 2)
            base_r += g - np.log(p * (p - 4) / (p - 2) ** 2)
            base_rho += g - np.log(p * (p - 3) / ((p - 1) * (p - 2)))
    if 3 in pf:
        base_r += np.log(3 / 4); base_rho += np.log(3 / 4)
    base_r += -4.0 / (P * np.log(P)); base_rho += -2.0 / (P * np.log(P))
    lr += base_r; lrho += base_rho
    zero = np.zeros(H, bool)
    for p in ps:
        p = int(p)
        if p == 2:
            if True:  # 2 | n always
                zero |= (h % 2 == 1)
                lr[h % 2 == 0] += np.log(2.0); lrho[h % 2 == 0] += np.log(2.0)
            continue
        if p in pf:
            idx = np.arange(p - 1, H, p)  # h ≡ 0 mod p
            d = np.log(p / (p - 1)) - np.log(p * (p - 2) / (p - 1) ** 2)
            lr[idx] += d; lrho[idx] += d
            continue
        if p == 3:
            m3 = h % 3 == 0
            zero_r3 = ~m3
            lr[m3] += np.log(3.0); lrho[m3] += np.log(3.0)
            zero |= zero_r3       # r_3 = rho_3 = 0 off 3|h
            continue
        if p > H and p > 2 * n:   # cannot divide h or n±h
            continue
        g = np.log(p * (p - 4) / (p - 2) ** 2)
        gr = np.log(p * (p - 3) / ((p - 1) * (p - 2)))
        if p <= H:
            idx = np.arange(p - 1, H, p)
            lr[idx] += np.log(p / (p - 2)) - g
            lrho[idx] += np.log(p / (p - 2)) - gr
        for c in (n % p, (-n) % p):
            if c == 0: continue
            start = c if c >= 1 else p
            idx = np.arange(start - 1, H, p)
            if len(idx): lr[idx] += np.log(p * (p - 3) / (p - 2) ** 2) - g
    r = np.exp(lr); rho = np.exp(lrho)
    r[zero] = 0.0; rho[zero] = 0.0
    return r, rho


def g_array(n, Q, primes=None):
    """g(q) for q = 0..Q (g(0) unused), multiplicative, squarefree support."""
    if primes is None: primes = primes_upto(max(Q, 1 << 16))
    pf = set(prime_factors(n, primes))
    g = np.ones(Q + 1)
    for p in primes[primes <= Q]:
        p = int(p)
        gp = 1.0 / (p - 1) if p in pf else 2.0 / (p - 2)
        g[p::p] *= gp
        if p * p <= Q: g[p * p::p * p] = 0.0
    g[0] = 0.0
    return g


def mellin_consts(n, primes):
    """Sum_{q<=x} g(q) ~ E(0) [ (1/2) log^2 x + (2 gamma + E'/E(0)) log x + c0 ]  ;  E(0) = 1/S(n).
    Returns E0, b1 = E'/E(0) (by Euler product, exact factors p<=PMAX plus negligible tail)."""
    pf = set(prime_factors(n, primes))
    E0 = 1.0; b1 = 0.0
    for p in primes:
        p = int(p)
        gp = 1.0 / (p - 1) if p in pf else 2.0 / (p - 2)
        E0 *= (1 + gp) * (1 - 1.0 / p) ** 2
        lp = np.log(p)
        # d/ds log[(1+g p^-s)(1-p^-1-s)^2] at 0
        b1 += -gp * lp / (1 + gp) + 2 * lp / (p - 1)
    return E0, b1
