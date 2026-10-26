"""T4: Lemma vaughanM for NON-squarefree M, in the exact form used by Lemma 2(ii) of E1_intento.md.
Prediction (from vaughanM, w = 1-x, r0=1/3, h_w=-5/24, gamma_w=0.1081876):
  F_mod(d) - F_M(d) = r0^2 N^2 (L - log d) + N^2 [ r0^2 (gamma + log 2pi + sum_{p|d} log p/(p-1)) + h_w - gamma_w/2 ]
                      + O(d^{1/2+o(1)} N^{3/2} + N L),
with F_M(d) = 2 sum_{1<=h<N, d|h} S(h) rho_w(h)^2 and F_mod(d) = sum_{d|n<=2N} S(n) nu_w(n)^2.
The N^3/phi(d) terms cancel by Plancherel ONLY IF the density is 1/phi(M) for both sums (vaughanM for arbitrary M);
a wrong density for non-squarefree M (e.g. 1/phi(rad M) or 1/M) would leave an N^3 term.
Usage: python3 arb_T4_vaughanM.py k
"""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/claude/ariadna/R1/arbitro1_fable')
from arb_core import *

k = int(sys.argv[1]); N = 2 ** k; L = math.log(N)
w = weight('1-x', N); D = conv_data(N, w); rho, nu, S = D['rho'], D['nu'], D['S']
r0 = 1 / 3; hw = -5 / 24; gw = 0.1081876; EG = 0.5772156649015329
def phi(n):
    r = n; m = n; p = 2
    while p * p <= m:
        if m % p == 0:
            r -= r // p
            while m % p == 0: m //= p
        p += 1
    if m > 1: r -= r // m
    return r
def primes_of(n):
    out = []; p = 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0: n //= p
        p += 1
    if n > 1: out.append(n)
    return out
res = []
for d in [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 30, 32, 36, 48, 60, 64, 72, 100, 128, 144, 210, 256, 360, 420, 512, 720, 1024, 1296, 2048, 4096]:
    if d > N: continue
    FM = 2 * float((S[:N + 1] * rho ** 2)[d::d].sum()); Fmod = float((S * nu ** 2)[d::d].sum())
    cd = EG + math.log(2 * math.pi) + sum(math.log(p) / (p - 1) for p in primes_of(d))
    pred = r0 ** 2 * N ** 2 * (L - math.log(d)) + N ** 2 * (r0 ** 2 * cd + hw - gw / 2)
    # what the N^3 terms would be individually (to show the scale of the cancellation):
    N3 = N ** 3 / phi(d) * 2 * (1 / 15)      # 2 int_0^1 r^2 = 2 * (1/15)?  r(u)=(1-u)^2(2+u)/6 ; int r^2 computed numerically below
    u = np.linspace(0, 1, 200001); r = (1 - u) ** 2 * (2 + u) / 6; I = 2 * np.trapezoid(r ** 2, u)
    N3 = N ** 3 / phi(d) * I
    err = (Fmod - FM) - pred
    res.append(dict(d=d, phi=phi(d), sqfree=all(d % (p * p) for p in primes_of(d)), Fmod_minus_FM_over_N2=(Fmod - FM) / N ** 2, pred_over_N2=pred / N ** 2,
                    err_over_N2=err / N ** 2, N3term_over_N2=N3 / N ** 2, err_over_sqrtd_N15=err / (math.sqrt(d) * N ** 1.5)))
    print(f"d={d:5d} phi={phi(d):4d} sqf={res[-1]['sqfree']!s:5s}  (Fmod-FM)/N^2={(Fmod-FM)/N**2:+.5f}  pred={pred/N**2:+.5f}  err/N^2={err/N**2:+.2e}  err/(sqrt d N^1.5)={err/(math.sqrt(d)*N**1.5):+.3f}   [N^3/phi(d) term alone = {N3/N**2:.2f} N^2]")
json.dump(dict(N=N, rows=res), open(f'/home/claude/ariadna/R1/arbitro1_fable/arb_T4_{k}.json', 'w'), indent=1)
