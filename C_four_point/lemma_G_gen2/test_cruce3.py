"""Cruce: r(n,h) via factorización triple vs producto de Euler directo prod_{p<=PMAX} r_p(h)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/claude/ariadna/R3/gen2_fable_G')
from g3_comun import *
out = {}
for n in [1000002, 1048576, 9699690, 30030, 65536]:
    M, hs, rho, tau, r, phi, fh, fuv = arrays(n, 400)
    worst = 0.0
    for h in [1, 2, 3, 5, 6, 12, 35, 77, 210, 331, 399, -7, -30, -221]:
        i = int(np.where(hs == h)[0][0])
        # producto directo truncado en p <= 200000 (cola ~ -4/(P log P) para genéricos)
        P = PRIMES[PRIMES <= 200000]
        v = 1.0
        for p in P:
            v *= M.r_p(int(p), h)
        v *= np.exp(-4.0 / (200000 * np.log(200000)))
        rel = abs(v - r[i]) / r[i] if r[i] else abs(v)
        worst = max(worst, rel)
    # media de tau sobre h debería ser ~1, media de rho ~1
    out[n] = dict(max_rel_err_r=worst, mean_rho=float(rho.mean()), mean_tau=float(tau.mean()),
                  Gamma=M.Gamma, Ctau=M.Ctau, ell=M.ell)
    print(n, out[n])
json.dump(out, open('test_cruce3.json', 'w'), indent=1)
