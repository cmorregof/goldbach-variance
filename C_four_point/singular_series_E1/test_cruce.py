"""Cruce: r(n,h) de e1_comun contra rline() de r3_perfil_h (misma definición, código independiente)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/ariadna/R3')
import r3_common as C
from e1_comun import r_and_rho, primes_upto
primes = C.sieve(1 << 22)
gen = lambda p: np.log1p(-4.0 / (p - 2) ** 2)
c1 = lambda p: np.log(p / (p - 2)) - gen(p)
c3 = lambda p: np.log(p * (p - 3) / (p - 2) ** 2) - gen(p)
B5 = C.euler_sum(primes, gen, 5, -4.0)
G1 = C.additive(1 << 21, primes, c1); G3 = C.additive(1 << 21, primes, c3)
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
    return R
for n in (1000, 2310, 30030, 65536, 99990, 510510):
    R = rline(n); r, rho = r_and_rho(n, n - 1)
    print(n, "max rel diff", np.max(np.abs(R - r) / np.maximum(1e-300, np.abs(R) + 1e-12)), "mean r", r.mean(), "mean rho", rho.mean())
