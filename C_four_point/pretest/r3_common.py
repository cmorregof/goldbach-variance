"""R3 pre-test, common machinery (25 oct 2026).

Conventions
-----------
n even. X_a = Lam(a) Lam(n-a), a = 1..n-1.  W(n) = sum_a X_a.
S(n) = 2 C2 prod_{p | n, p > 2} (p-1)/(p-2)  (twin-prime constant C2, full precision).
Local factors (Hardy-Littlewood, von Mangoldt weights, so no log factors):
   k-tuple of shifts with nu_p distinct residues:  f_p(nu) = (1 - nu/p) / (1 - 1/p)^k.
All infinite products are taken over ALL primes: exact Euler factors for p <= PMAX
(sieved) plus an analytic tail for p > PMAX (generic factor only, since no p > PMAX
divides any of the relevant nonzero integers when PMAX >= the range).
"""
import numpy as np

C2 = 0.66016181584686957392781211001455577843262336  # twin prime constant


def sieve(lim):
    isp = np.ones(lim + 1, bool); isp[:2] = False
    for p in range(2, int(lim ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0]


def von_mangoldt(lim, primes, prime_powers=True):
    Lam = np.zeros(lim + 1)
    Lam[primes[primes <= lim]] = np.log(primes[primes <= lim])
    if prime_powers:
        for p in primes[primes <= int(lim ** 0.5) + 1]:
            pk = int(p) * int(p)
            while pk <= lim:
                Lam[pk] = np.log(p); pk *= int(p)
    return Lam


def singular_S(lim, primes):
    """S(n) for all n = 0..lim (odd n -> 0)."""
    S = np.zeros(lim + 1); S[0::2] = 2 * C2
    for p in primes[1:]:
        if p > lim: break
        S[0::p] *= (p - 1) / (p - 2)
    return S


def additive(lim, primes, coef, pmin=5):
    """G(x) = sum_{p | x, p >= pmin} coef(p), for x = 0..lim (G(0) meaningless)."""
    G = np.zeros(lim + 1)
    ps = primes[(primes >= pmin) & (primes <= lim)]
    c = coef(ps.astype(float))
    for p, cp in zip(ps, c):
        G[p::p] += cp
    return G


def euler_sum(primes, fn, pmin, tail_coef):
    """sum_{p >= pmin} fn(p) over sieved primes plus tail  tail_coef * sum_{p>P} 1/p^2."""
    ps = primes[primes >= pmin].astype(float)
    P = ps[-1]
    return float(np.sum(fn(ps)) + tail_coef / (P * np.log(P)))


def odd_prime_factors(n, primes):
    out = []; m = n
    while m % 2 == 0: m //= 2
    for p in primes[1:]:
        p = int(p)
        if p * p > m: break
        if m % p == 0:
            out.append(p)
            while m % p == 0: m //= p
    if m > 1: out.append(m)
    return out
