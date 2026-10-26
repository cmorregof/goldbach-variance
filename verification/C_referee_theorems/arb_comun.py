"""Árbitro 3 (Fable): implementación independiente de r(n,h), rho, tau y constantes.
Todo se calcula desde la DEFINICIÓN (nu_p = #{0,-h,n,n-h} mod p), no desde la tabla de la sumisión."""
import numpy as np, math

C2 = 0.66016181584686957392781211001455577843262336
EULER = 0.57721566490153286060651209008240243104215933593992


def sieve(lim):
    isp = np.ones(lim + 1, bool); isp[:2] = False
    for p in range(2, int(lim ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0].astype(np.int64)


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


def sing(n, primes):
    s = 2 * C2
    for p in factor(n, primes):
        if p > 2: s *= (p - 1) / (p - 2)
    return s


def r_def(n, hs, primes):
    """r(n,h) = prod_p (1-nu_p/p)/(1-nu'_p/p)^2 desde la definición, para un array de h.
    Factores exactos en p <= primes[-1] (debe ser >= 2n+2 para que ningún primo mayor divida h(n^2-h^2)
    cuando 0<|h|<n), cola analítica para el factor genérico."""
    hs = np.asarray(hs, dtype=np.int64)
    P = primes.astype(np.float64)
    logr = np.zeros(len(hs))
    # cola: prod_{p>P_max} (1-4/p)/(1-2/p)^2 = prod (1 - 4/(p-2)^2)
    Pm = P[-1]
    tail = -4.0 / (Pm * math.log(Pm))
    for p in primes:
        p = int(p)
        a = np.zeros(len(hs))
        s0 = 0; s1 = (-hs) % p; s2 = n % p; s3 = (n - hs) % p
        nu = 1 + (s1 != 0).astype(int)
        nu += ((s2 != 0) & (s2 != s1)).astype(int)
        nu += ((s3 != 0) & (s3 != s1) & (s3 != s2)).astype(int)
        nup = 1 + (1 if n % p else 0)
        num = 1 - nu / p
        den = (1 - nup / p) ** 2
        with np.errstate(divide='ignore'):
            a = np.where(num > 0, np.log(np.maximum(num, 1e-300)) - math.log(den), -np.inf)
        logr += a
    return np.exp(logr + tail)


def rho_mult(n, hs, primes, Gam=None):
    """rho por la fórmula multiplicativa del Lema 4(ii): rho = l*Gamma*1_{l|h}*sum_{d|h, d in D} kappa(d)."""
    hs = np.asarray(hs, dtype=np.int64)
    pf = set(factor(n, primes))
    l = 2 if 3 in pf else 6
    if Gam is None: Gam = Gamma_n(n, primes)
    out = np.zeros(len(hs))
    for i, h in enumerate(hs):
        h = int(abs(h))
        if h % l: continue
        s = 1.0
        for p in factor(h, primes):
            if p == 2 or (p == 3 and 3 not in pf): continue
            s *= 1 + (1 / (p - 2) if p in pf else 2 / (p - 3))
        out[i] = l * Gam * s
    return out


def Gamma_n(n, primes):
    pf = set(factor(n, primes))
    lg = 0.0
    for p in primes:
        p = int(p)
        if p == 2: continue
        if p == 3 and 3 not in pf: continue
        if p in pf: lg += math.log(1 - 1 / (p - 1) ** 2)
        else: lg += math.log(1 - 2 / ((p - 1) * (p - 2)))
    Pm = float(primes[-1])
    return math.exp(lg - 2.0 / (Pm * math.log(Pm)))


def rho_def(n, hs, primes):
    """rho desde la definición como media condicional: rho_p = r_p si p|n o p=2 o (p=3, 3∤n);
    si p∤n, p>=5: p/(p-2) si p|h, p(p-3)/((p-1)(p-2)) si no."""
    hs = np.asarray(hs, dtype=np.int64)
    logr = np.zeros(len(hs))
    Pm = float(primes[-1])
    tail = -2.0 / (Pm * math.log(Pm))
    for p in primes:
        p = int(p)
        if n % p == 0 or p == 2 or p == 3:
            s1 = (-hs) % p; s2 = n % p; s3 = (n - hs) % p
            nu = 1 + (s1 != 0).astype(int)
            nu += ((s2 != 0) & (s2 != s1)).astype(int)
            nu += ((s3 != 0) & (s3 != s1) & (s3 != s2)).astype(int)
            nup = 1 + (1 if n % p else 0)
            num = 1 - nu / p; den = (1 - nup / p) ** 2
            with np.errstate(divide='ignore'):
                logr += np.where(num > 0, np.log(np.maximum(num, 1e-300)) - math.log(den), -np.inf)
        else:
            dv = (hs % p == 0)
            logr += np.where(dv, math.log(p / (p - 2)), math.log(p * (p - 3) / ((p - 1) * (p - 2))))
    return np.exp(logr + tail)
