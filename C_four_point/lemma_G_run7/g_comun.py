"""run7_G: utilidades para el Lema G (25 oct 2026).
Reutiliza r_and_rho de run6_E1/e1_comun.py (cruzado allí con r3_perfil_h.rline).
Añade: tau por bloques de primos, T_e exacto para e pequeño, caracteres pares.
tau_p (p∤n, p>=5):  1 si p|h ;  (p-1)/(p-2) si p | n∓h ;  (p-1)(p-4)/((p-2)(p-3)) en otro caso."""
import sys, math, numpy as np
sys.path.insert(0, "/home/claude/ariadna/R3/run6_E1")
from e1_comun import r_and_rho, primes_upto, prime_factors, sing  # noqa


def log_tau_blocks(n, H, bounds, primes):
    """Devuelve lista de arrays logtau_j[h-1], h=1..H, para primos p en [bounds[j], bounds[j+1]),
    p∤n, p>=5. El último bloque incluye la cola genérica analítica p > max primo usado."""
    h = np.arange(1, H + 1)
    pf = set(prime_factors(n, primes))
    out = [np.zeros(H) for _ in range(len(bounds) - 1)]
    PB = max(2 * n + 2, 1 << 24)
    pb = primes_upto(PB)
    for j in range(len(bounds) - 1):
        lo, hi = bounds[j], bounds[j + 1]
        sel = pb[(pb >= max(lo, 5)) & (pb < hi)]
        sel = np.array([p for p in sel if int(p) not in pf], dtype=np.int64)
        if len(sel) == 0:
            continue
        pf_ = sel.astype(float)
        gen = np.log1p(-2.0 / ((pf_ - 2) * (pf_ - 3)))
        out[j] += float(gen.sum())
        for p in sel:
            p = int(p)
            g = math.log1p(-2.0 / ((p - 2) * (p - 3)))
            if p > H and p > 2 * n:
                continue
            if p <= H:
                out[j][p - 1::p] -= g        # tau_p = 1 at p|h
            for c in {n % p, (-n) % p}:
                if c == 0:
                    continue
                idx = np.arange(c - 1, H, p)
                out[j][idx] += math.log((p - 1) / (p - 2)) - g
    if bounds[-1] >= PB:   # tail beyond the sieve range: generic factor only
        P = float(pb[-1]); out[-1] += -2.0 / (P * math.log(P))
    return out


def fejer_w(H):
    h = np.arange(1, H + 1)
    return 2.0 * (1 - h / H)   # factor 2: h and -h (r, rho even in h)
