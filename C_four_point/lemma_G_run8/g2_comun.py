"""run8_G2: utilidades ligeras (26 oct 2026).

r(n,h), rho(n,h) exactos para 1 <= h <= n-1 (n par), con los factores locales de
run6_E1/e1_comun.py, pero en una versión rápida pensada para barrer TODOS los n pares
de un bloque [N, 2N] (promedios en n).  Se cruza con e1_comun.r_and_rho en `test_cruce2.py`.

Convenciones (E1 §1, §4; G §1):
  p | n (incl. p=2):  r_p = p/(p-1) si p|h ; p(p-2)/(p-1)^2 si no        (p=2: 2 / 0)
  p ∤ n, p=3       :  r_3 = 3 si 3|h ; 0 si no
  p ∤ n, p>=5      :  r_p = p/(p-2) si p|h ; p(p-3)/(p-2)^2 si p | n∓h ; p(p-4)/(p-2)^2 si no
  rho_p = r_p salvo p ∤ n, p >= 5:  rho_p = p/(p-2) si p|h ; p(p-3)/((p-1)(p-2)) si no.
  t_p(n,h) (p>=5, p∤n): 0 si p|h ; 1/(p-2) si p | n∓h ; -c2(p) si no ;  t_p := 0 si p | n.
  c2(p) = 2/((p-2)(p-3)).
"""
import math
import numpy as np

_PR = {}


def primes_upto(lim):
    lim = int(lim)
    for k, v in _PR.items():
        if k >= lim:
            return v[v <= lim]
    s = np.ones(lim + 1, bool); s[:2] = False
    for p in range(2, int(lim ** 0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    v = np.nonzero(s)[0]
    _PR[lim] = v
    return v


def factor(m, primes):
    out = []
    for p in primes:
        p = int(p)
        if p * p > m:
            break
        if m % p == 0:
            out.append(p)
            while m % p == 0:
                m //= p
    if m > 1:
        out.append(m)
    return out


# constantes de cola: sumas sobre TODOS los p >= 5 de log(factor genérico)
_PB = primes_upto(1 << 24).astype(float)
_P5 = _PB[_PB >= 5]
_PMAX = _PB[-1]
LOG_GEN_R = float(np.sum(np.log1p(-4.0 / (_P5 - 2) ** 2))) - 4.0 / (_PMAX * math.log(_PMAX))
LOG_GEN_RHO = float(np.sum(np.log1p(-2.0 / ((_P5 - 1) * (_P5 - 2))))) - 2.0 / (_PMAX * math.log(_PMAX))


def c2(p):
    return 2.0 / ((p - 2) * (p - 3))


def r_rho(n, primes=None):
    """arrays r[h-1], rho[h-1] para h = 1..n-1 (n par)."""
    assert n % 2 == 0 and n >= 6
    H = n - 1
    if primes is None:
        primes = primes_upto(max(2 * n + 2, 1000))
    pf = set(factor(n, primes))
    lr = np.full(H, LOG_GEN_R); lq = np.full(H, LOG_GEN_RHO)
    zero = np.zeros(H, bool)
    h = np.arange(1, H + 1)
    # p = 2 (2 | n siempre)
    zero |= (h % 2 == 1); lr[1::2] += math.log(2.0); lq[1::2] += math.log(2.0)
    # p = 3
    if 3 in pf:
        a, b = math.log(1.5), math.log(0.75)
        lr += b; lq += b; lr[2::3] += a - b; lq[2::3] += a - b
    else:
        zero |= (h % 3 != 0); lr[2::3] += math.log(3.0); lq[2::3] += math.log(3.0)
    for p in primes:
        p = int(p)
        if p < 5:
            continue
        if p > 2 * n:
            break
        if p in pf:
            g = math.log(p * (p - 2) / (p - 1) ** 2)
            dr = g - math.log(p * (p - 4) / (p - 2) ** 2)
            dq = g - math.log(p * (p - 3) / ((p - 1) * (p - 2)))
            lr += dr; lq += dq
            d0 = math.log(p / (p - 1)) - g
            lr[p - 1::p] += d0; lq[p - 1::p] += d0
            continue
        gr = math.log(p * (p - 4) / (p - 2) ** 2)
        gq = math.log(p * (p - 3) / ((p - 1) * (p - 2)))
        if p <= H:
            lr[p - 1::p] += math.log(p / (p - 2)) - gr
            lq[p - 1::p] += math.log(p / (p - 2)) - gq
        d = math.log(p * (p - 3) / (p - 2) ** 2) - gr
        for c in {n % p, (-n) % p}:
            if c == 0:
                continue
            lr[c - 1::p] += d
    r = np.exp(lr); q = np.exp(lq)
    r[zero] = 0.0; q[zero] = 0.0
    return r, q


def t_p_arr(p, n, h):
    """t_p(n,h) vectorizado en h (array de enteros)."""
    h = np.asarray(h)
    out = np.full(h.shape, -c2(p))
    if n % p == 0:
        return np.zeros(h.shape)
    out[(n - h) % p == 0] = 1.0 / (p - 2)
    out[(n + h) % p == 0] = 1.0 / (p - 2)
    out[h % p == 0] = 0.0
    return out


def T_e_arr(e_primes, n, h):
    out = np.ones(np.asarray(h).shape)
    for p in e_primes:
        out = out * t_p_arr(p, n, h)
    return out


def w_smooth(x):
    """peso suave con soporte en [1,2]: exp(-1/((x-1)(2-x))) normalizado a max 1."""
    x = np.asarray(x, float)
    out = np.zeros(x.shape)
    m = (x > 1) & (x < 2)
    y = x[m]
    out[m] = np.exp(4.0 - 1.0 / ((y - 1) * (2 - y)))
    return out


def gamma_e(e_primes):
    """c2(e)*e = prod 2p/((p-2)(p-3))."""
    g = 1.0
    for p in e_primes:
        g *= 2.0 * p / ((p - 2) * (p - 3))
    return g
