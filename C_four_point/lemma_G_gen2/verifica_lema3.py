"""verifica_lema3.py — Lema 3 (i) y (iii): S_e en forma (d,k) y en forma de Kloosterman,
frente al cálculo directo S_e = sum_h phi rho T_e. También la identidad local Lema 1(iii)."""
import sys, json, numpy as np
from fractions import Fraction
sys.path.insert(0, '/home/claude/ariadna/R3/gen2_fable_G')
from g3_comun import *

out = {}
# Lema 1(iii) exacto en racionales para p = 5..97
ok = True
for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
    k = Fraction(2, p - 3); c2p = Fraction(2, (p - 1) * (p - 4)); kp = (1 + k) * (1 + c2p) - 1
    lp = Fraction(1, p - 4); cp = Fraction((p - 1) * (p - 4), (p - 2) * (p - 3))
    ok &= (cp * (1 + kp / p + 2 * lp / p) == 1 + k / p) and (kp == Fraction(2, p - 4))
out['lema1_iii_exacta_p_le_97'] = bool(ok)

def psi_e(e, x):
    v = 1.0
    for p in factor_small(e, PRIMES):
        xp = x % p
        v *= 1.0 / (p - 2) if xp in (1, p - 1) else -2.0 / ((p - 2) * (p - 3))
    return v

def psihat_e(e, j):
    v = 1.0
    for p in factor_small(e, PRIMES):
        p = int(p); jp = (j * pow(e // p, -1, p)) % p     # coordenada CRT: j/e = sum_p j_p/p
        if jp == 0:
            return 0.0
        v *= 2 * ((p - 1) * np.cos(2 * np.pi * jp / p) + 1) / (p * (p - 2) * (p - 3))
    return v

casos = [(1000002, 600, 7), (1000002, 600, 35), (1048576, 500, 143), (1048576, 500, 55), (30030, 300, 17), (30030, 300, 391)]
for (n, H, e) in casos:
    M, hs, rho, tau, r, phi, fh, fuv = arrays(n, H)
    # directo
    Te = np.ones(len(hs))
    for p in factor_small(e, PRIMES):
        t = np.full(len(hs), -2.0 / ((p - 2) * (p - 3)))
        t[(hs % p) == 0] = 0.0
        t[((n - hs) % p == 0) | ((n + hs) % p == 0)] = 1.0 / (p - 2)
        Te *= t
    S_dir = float(np.dot(phi * rho, Te))
    # (i): forma (d,k)
    ell = M.ell
    S_dk = 0.0
    for d in range(1, H // ell + 1):
        if d % e == 0 or any((d % p == 0) for p in factor_small(e, PRIMES)):
            continue
        fd = factor_small(d, PRIMES)
        # d en D: libre de cuadrados, primos en F
        if any(fd.count(p) > 1 for p in fd):
            continue
        sq = all(d % (p * p) for p in fd)
        inF = all((p % 2 == 1) and not (p == 3 and n % 3) for p in fd)
        if not (sq and inF):
            continue
        kd = 1.0
        for p in fd:
            kd *= M.kappa(p)
        for k in range(1, H // (ell * d) + 1):
            if ell * d * k >= H or any((k % p == 0) for p in factor_small(e, PRIMES)):
                continue
            x = pow(ell * d * k, -1, e)
            S_dk += 2 * ell * M.Gamma * kd * (1 - ell * d * k / H) * psi_e(e, (n * x) % e)
    # (iii): forma Kloosterman
    S_kl = 0.0
    for j in range(1, e):
        ph = psihat_e(e, j)
        if ph == 0.0:
            continue
        K = 0.0
        for d in range(1, H // ell + 1):
            fd = factor_small(d, PRIMES)
            if any((d % p == 0) for p in factor_small(e, PRIMES)) or not all(d % (p * p) for p in fd):
                continue
            if not all((p % 2 == 1) and not (p == 3 and n % 3) for p in fd):
                continue
            kd = 1.0
            for p in fd:
                kd *= M.kappa(p)
            for k in range(1, H // (ell * d) + 1):
                if ell * d * k >= H or any((k % p == 0) for p in factor_small(e, PRIMES)):
                    continue
                x = pow(ell * d * k, -1, e)
                K += kd * (1 - ell * d * k / H) * np.cos(2 * np.pi * ((j * n * x) % e) / e)
        S_kl += 2 * ell * M.Gamma * ph * K
    out[f'n{n}_H{H}_e{e}'] = dict(S_directo=S_dir, S_dk=S_dk, S_kloost=S_kl,
                                 rel_i=abs(S_dk - S_dir) / max(1e-300, abs(S_dir)),
                                 rel_iii=abs(S_kl - S_dir) / max(1e-300, abs(S_dir)))
    print(n, H, e, out[f'n{n}_H{H}_e{e}'])
print('Lema1(iii) exacta:', ok)
json.dump(out, open('/home/claude/ariadna/R3/gen2_fable_G/verifica_lema3.json', 'w'), indent=1)
