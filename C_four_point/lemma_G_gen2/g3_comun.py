"""g3_comun.py — utilidades numéricas para gen2_fable_G (26 oct 2026).

Definiciones (E1 §1, §4):
  r_p(h), rho_p(h), kappa, ell, Gamma_n;  r = rho * tau;
  factorización triple: r(n,h) = C_n * A_n(h) * B_n(n-h) * B_n(n+h) para p>=5, p∤n:
     B_n(m) = prod_{p|m, p in P'} (1 + 1/(p-4)),  lambda'(p) = 1/(p-4)
     A_n(h) = prod_{p in {2,3} o p|n} r_p(h) * prod_{p|h, p in P'} (1 + 2/((p-1)(p-4)))
     C_n    = prod_{p in P'} p(p-4)/(p-2)^2
Todo se calcula con primos hasta PMAX y cola analítica para los productos convergentes.
"""
import numpy as np
from math import log

PMAX = 2_000_000  # tope de primos para productos de Euler truncados


def primes_upto(N):
    s = np.ones(N + 1, dtype=bool); s[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


PRIMES = primes_upto(PMAX)


def factor_small(m, plist):
    """primos de plist que dividen a m (m entero > 0)."""
    out = []
    for p in plist:
        if p * p > m:
            break
        if m % p == 0:
            out.append(p)
            while m % p == 0:
                m //= p
    if m > 1:
        out.append(m)
    return out


def sieve_interval(lo, hi, plist):
    """Factores primos (distintos) de cada entero en [lo, hi): devuelve lista de listas."""
    L = hi - lo
    facs = [[] for _ in range(L)]
    rem = np.arange(lo, hi, dtype=np.int64)
    for p in plist:
        if p * p > hi:
            break
        start = (-lo) % p
        idx = np.arange(start, L, p)
        if len(idx) == 0:
            continue
        for i in idx:
            facs[i].append(int(p))
        while True:
            mask = rem[idx] % p == 0
            if not mask.any():
                break
            rem[idx[mask]] //= p
    for i in range(L):
        if rem[i] > 1:
            facs[i].append(int(rem[i]))
    return facs


class Modelo:
    def __init__(self, n):
        assert n % 2 == 0
        self.n = n
        self.np_ = [int(p) for p in factor_small(n, PRIMES)]   # primos de n
        self.set_np = set(self.np_)
        self.ell = 2 if n % 3 == 0 else 6
        # Gamma_n = prod_{p in F} rho_p(gen): F = impares salvo el 3 cuando 3∤n
        P = PRIMES[PRIMES >= 5].astype(np.float64)
        gen = P * (P - 3) / ((P - 1) * (P - 2))        # rho_p(gen) para p∤n, p>=5
        logG = np.sum(np.log(gen))
        # corrección por p | n, p>=5: rho_p(gen) = r_p(p∤h) = p(p-2)/(p-1)^2
        for p in self.np_:
            if p >= 5:
                logG += log(p * (p - 2) / (p - 1) ** 2) - log(p * (p - 3) / ((p - 1) * (p - 2)))
        if n % 3 == 0:
            logG += log(3 * 1 / 4)   # p=3 | n: r_3(3∤h) = 3*1/2^2 = 3/4
        # cola p > PMAX: log(1 - 2/((p-1)(p-2))) ~ -2/p^2, sum_{p>P} 2/p^2 ~ 2/(P log P)
        logG -= 2.0 / (PMAX * log(PMAX))
        self.Gamma = float(np.exp(logG))
        # C_n = prod_{p in P'} p(p-4)/(p-2)^2 ;  cola ~ -4/(P log P)
        cn = P * (P - 4) / (P - 2) ** 2
        logC = np.sum(np.log(cn))
        for p in self.np_:
            if p >= 5:
                logC -= log(p * (p - 4) / (p - 2) ** 2)
        logC -= 4.0 / (PMAX * log(PMAX))
        self.C = float(np.exp(logC))

    # --- factores locales ---
    def r_p(self, p, h):
        n = self.n
        if p in self.set_np or p == 2:
            return p / (p - 1) if h % p == 0 else p * (p - 2) / (p - 1) ** 2
        if h % p == 0:
            return p / (p - 2)
        if (n - h) % p == 0 or (n + h) % p == 0:
            return p * (p - 3) / (p - 2) ** 2
        return p * (p - 4) / (p - 2) ** 2

    def kappa(self, p):
        return 1.0 / (p - 2) if p in self.set_np else 2.0 / (p - 3)

    def rho(self, h, facs_h):
        """rho(n,h) = ell*Gamma*1_{ell|h} * prod_{p|h, p in F}(1+kappa(p))."""
        h = abs(h)
        if h % self.ell:
            return 0.0
        v = self.ell * self.Gamma
        for p in facs_h:
            if p == 2 or (p == 3 and self.n % 3):
                continue
            v *= 1 + self.kappa(p)
        return v

    def B(self, facs_m):
        v = 1.0
        for p in facs_m:
            if p >= 5 and p not in self.set_np:
                v *= 1 + 1.0 / (p - 4)
        return v

    def A(self, h, facs_h):
        v = 1.0
        for p in [2, 3] + [q for q in self.np_ if q >= 5]:
            v *= self.r_p(p, h)
        for p in facs_h:
            if p >= 5 and p not in self.set_np:
                v *= 1 + 2.0 / ((p - 1) * (p - 4))
        return v

    def prepara_tau(self):
        # tau(h) = prod_{p∤2nh, p>=5} c_p * prod_{p | u v, p in P'} (p-3)/(p-4),
        # c_p = (p-1)(p-4)/((p-2)(p-3));  Ctau := prod_{p in P'} c_p (cola ~ -2/(P log P))
        P = PRIMES[PRIMES >= 5].astype(np.float64)
        cp = (P - 1) * (P - 4) / ((P - 2) * (P - 3))
        logc = np.sum(np.log(cp))
        for p in self.np_:
            if p >= 5:
                logc -= log((p - 1) * (p - 4) / ((p - 2) * (p - 3)))
        logc -= 2.0 / (PMAX * log(PMAX))
        self.Ctau = float(np.exp(logc))


def fejer(h, H):
    return max(0.0, 1 - abs(h) / H)


def arrays(n, H):
    """Devuelve h (array 0<|h|<H), rho, tau, r, phi. Exacto salvo colas de Euler ~1e-7."""
    M = Modelo(n); M.prepara_tau()
    hs = np.array([h for h in range(-H + 1, H) if h != 0], dtype=np.int64)
    small = PRIMES[PRIMES <= max(int((n + H) ** 0.5) + 2, 3)]
    facs_h = sieve_interval(1, H, small)                 # índices h-1 para h>=1
    facs_uv = sieve_interval(n - H + 1, n + H, small)    # índice m-(n-H+1)
    rho = np.empty(len(hs)); tau = np.empty(len(hs))
    for i, h in enumerate(hs):
        fh = facs_h[abs(h) - 1]
        fu = facs_uv[(n - h) - (n - H + 1)]
        fv = facs_uv[(n + h) - (n - H + 1)]
        rho[i] = M.rho(h, fh)
        t = 1.0
        for p in fh:
            if p >= 5 and p not in M.set_np:
                t *= 1 + 2.0 / ((p - 1) * (p - 4))
        for p in fu + fv:
            if p >= 5 and p not in M.set_np:
                t *= (p - 3.0) / (p - 4.0)
        tau[i] = M.Ctau * t
    phi = 1 - np.abs(hs) / H
    return M, hs, rho, tau, rho * tau, phi, facs_h, facs_uv
