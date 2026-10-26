"""T1: Lemma 2 of E1_intento.md -- |Delta_w(d)| << N^{2+eta} for ALL d <= 2N, including non-squarefree d
(the submission's `maxDelta` was restricted to squarefree d, where beta>0).
Also splits Delta_w = G_w - G_w^mod and reports both, and checks the identity X_w = sum beta Delta.
Usage: python3 arb_T1_lema2.py k [weights...]
"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, '/home/claude/ariadna/R1/arbitro1_fable')
from arb_core import *

k = int(sys.argv[1]); N = 2 ** k
wnames = sys.argv[2:] or ['1-x', '(1-x)^2']
out = dict(N=N, k=k)
pr = primes_upto(2 * N + 2)
beta = beta_arr(2 * N, pr)
sqf = np.ones(2 * N + 1, bool); sqf[0] = False
for p in pr:
    if p * p > 2 * N: break
    sqf[p * p::p * p] = False
even = np.zeros(2 * N + 1, bool); even[2::2] = True
d = np.arange(2 * N + 1)
for wn in wnames:
    t0 = time.time()
    w = weight(wn, N); D = conv_data(N, w)
    rho, nu, E, dlt, S, psi2, R = D['rho'], D['nu'], D['E'], D['dlt'], D['S'], D['psi2'], D['R']
    F1 = mult_sums(rho * E, 2 * N); F2 = mult_sums(nu * dlt, 2 * N)
    Delta = 2 * F1 - F2; Delta[0] = 0
    # G_w^mod(d) = rho(0) psi2(0) + 2 sum_{h>=1,d|h} S rho^2 - sum_{d|n} S nu^2 ; G_w = Delta + G^mod
    Gmod = D['rho'][0] * D['psi2'][0] + 2 * mult_sums(S[:N + 1] * rho ** 2, 2 * N) - mult_sums(S * nu ** 2, 2 * N)
    Gmod[0] = 0
    G = Delta + Gmod
    Xdir = 4 * (S[:N + 1] * rho * E).sum() - 2 * (S * nu * dlt).sum()
    Xd = (beta * Delta).sum()
    r = dict(X_direct_over_N2=Xdir / N ** 2, X_dbasis_over_N2=Xd / N ** 2, relerr=abs(Xd - Xdir) / abs(Xdir),
             min_G_over_N2=float(G[1:].min() / N ** 2))
    ranges = {'d<=sqrtN': (d >= 1) & (d <= math.isqrt(N)), 'sqrtN<d<=N': (d > math.isqrt(N)) & (d <= N),
              'N<d<=2N': (d > N) & (d <= 2 * N), 'd>N^0.55': d > N ** 0.55}
    classes = {'even_sqf': even & sqf, 'even_nonsqf': even & ~sqf, 'odd': ~even & (d >= 1), 'all': d >= 1}
    tab = {}
    for rn, rm in ranges.items():
        for cn, cm in classes.items():
            m = rm & cm
            if m.any():
                i = np.flatnonzero(m); j = i[np.argmax(np.abs(Delta[i]))]
                tab[f'{rn}|{cn}'] = dict(maxDelta=float(np.abs(Delta[i]).max() / N ** 2), argmax=int(j),
                                         maxG=float(G[i].max() / N ** 2), maxGmod=float(np.abs(Gmod[i]).max() / N ** 2),
                                         count=int(m.sum()))
    r['table'] = tab
    # a few specific non-squarefree d
    spec = {}
    for dd in [4, 8, 12, 16, 18, 24, 36, 48, 72, 100, 144, 360, 1024, 4096]:
        if dd <= 2 * N: spec[dd] = dict(Delta=float(Delta[dd] / N ** 2), G=float(G[dd] / N ** 2), Gmod=float(Gmod[dd] / N ** 2))
    r['specific_nonsqf'] = spec
    # Lemma 2 (ii) ingredient: F_M(d) - F_mod(d) individually vs N^2 for large d
    FM = 2 * mult_sums(S[:N + 1] * rho ** 2, 2 * N); Fmod = mult_sums(S * nu ** 2, 2 * N)
    big = d > N ** 0.9
    r['max_FM_over_N2_d>N^0.9'] = float(FM[big].max() / N ** 2); r['max_Fmod_over_N2_d>N^0.9'] = float(Fmod[big].max() / N ** 2)
    r['max|FM-Fmod|_over_N2_d>N^0.9'] = float(np.abs(FM - Fmod)[big].max() / N ** 2)
    r['rho0psi20_over_N2'] = float(D['rho'][0] * D['psi2'][0] / N ** 2)
    r['seconds'] = time.time() - t0
    out[wn] = r
    print(wn, json.dumps(r, indent=1), flush=True)
json.dump(out, open(f'/home/claude/ariadna/R1/arbitro1_fable/arb_T1_{k}.json', 'w'), indent=1)
