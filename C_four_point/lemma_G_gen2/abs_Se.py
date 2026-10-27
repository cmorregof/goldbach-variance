"""abs_Se.py — ¿puede la ruta «valor absoluto por conductor» dar O(log H)?
Calcula S_e = sum_h phi_H rho T_e para todo e libre de cuadrados (primos >=5, p∤n) con e <= E,
y compara  sum_e S_e  (debe ~ T2^F),  sum_e |S_e|,  y por bloques diádicos.
Uso: python3 abs_Se.py n H E
"""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/ariadna/R3/gen2_fable_G')
from g3_comun import *

n, H, E = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
M, hs, rho, tau, r, phi, fh, fuv = arrays(n, H)
w = phi * rho
T2 = float(np.sum(phi * (r - rho)))
P = [int(p) for p in PRIMES if 5 <= p <= E and n % p]

def t_arr(p):
    c2 = 2.0 / ((p - 2) * (p - 3))
    t = np.full(len(hs), -c2)
    t[(hs % p) == 0] = 0.0
    t[((n - hs) % p == 0) | ((n + hs) % p == 0)] = 1.0 / (p - 2)
    return t

nb = int(np.log2(E)) + 2
blk_signed = np.zeros(nb); blk_abs = np.zeros(nb); blk_cnt = np.zeros(nb, dtype=int)
blk_gamma = np.zeros(nb)
prime_abs = 0.0
Smax = (0.0, 1)

def gamma(e_primes):
    v = 1.0
    for p in e_primes:
        v *= 2.0 * p / ((p - 2) * (p - 3))
    return v

def dfs(start, e, T, eps):
    global prime_abs, Smax
    for i in range(start, len(P)):
        p = P[i]
        if e * p > E:
            break
        e2 = e * p
        T2_ = T * t_arr(p)
        S = float(np.dot(w, T2_))
        b = int(np.log2(e2))
        blk_signed[b] += S; blk_abs[b] += abs(S); blk_cnt[b] += 1
        blk_gamma[b] += gamma(eps + [p])
        if e == 1:
            prime_abs += abs(S)
        if abs(S) > Smax[0]:
            Smax = (abs(S), e2)
        dfs(i + 1, e2, T2_, eps + [p])

dfs(0, 1, np.ones(len(hs)), [])
tot_signed = float(blk_signed.sum()); tot_abs = float(blk_abs.sum())
res = dict(n=n, H=H, E=E, T2F=T2, sum_Se=tot_signed, sum_abs_Se=tot_abs, sum_abs_primes=prime_abs,
           logH=float(np.log(H)), ratio_abs_over_logH=tot_abs / np.log(H), Smax=Smax,
           blocks=[dict(e_range=[2 ** b, 2 ** (b + 1) - 1], count=int(blk_cnt[b]), signed=float(blk_signed[b]),
                        abs=float(blk_abs[b]), sum_gamma=float(blk_gamma[b])) for b in range(nb) if blk_cnt[b]])
print(json.dumps({k: v for k, v in res.items() if k != 'blocks'}, indent=1))
for bl in res['blocks']:
    print(bl)
json.dump(res, open(f'abs_Se_n{n}_H{H}_E{E}.json', 'w'), indent=1)
