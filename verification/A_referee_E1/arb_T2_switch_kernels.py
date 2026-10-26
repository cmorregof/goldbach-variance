"""T2: Steps 1-3 and Lemma 5 of E1_intento.md, independently.
 (a) Lemma 1: beta(2m) = (4C2/m) sum_{e|m} gamma(e); partial sums of |gamma|.
 (b) Step 1: S_Theta = 8 C2 sum_e gamma(e) T(2e), T(M0) = sum_{M0|d} Theta(d) Delta_w(d)/d.
 (c) Step 3 (switch): for several M0 (incl. non-squarefree 4, 12, 18),
       T(M0) = sum_l [ 2 sum_{M|k} h_l(k) E_w(k) - sum_{M|n} h'_l(n) delta_w(n) ],  M = M0 l.
 (d) Lemma 5(a): sum_{k>=2}|D2 rho_w|, sum_{n>=2}|D2 nu_w|, sup|grad rho|, sup|grad nu|, for several w.
 (e) Lemma 5(c): sum_k|D2 h_l| and sum_n|D2 h'_l| against N/(delta Y^2 l) + 1/Y, for several l, eps, delta, w.
     (the submission tested only h_l = phi_l rho_w, not the Goldbach kernel h'_l = phi_l nu_w).
Usage: python3 arb_T2_switch_kernels.py k
"""
import sys, json, math
import numpy as np
sys.path.insert(0, '/home/claude/ariadna/R1/arbitro1_fable')
from arb_core import *

k = int(sys.argv[1]); N = 2 ** k; out = dict(N=N)
pr = primes_upto(2 * N + 2); beta = beta_arr(2 * N, pr); gam = gamma_arr(N, pr)
# (a)
err = 0.0
for m in range(1, 5000):
    s = sum(gam[e] for e in range(1, m + 1) if m % e == 0)
    err = max(err, abs(beta[2 * m] - 4 * C2 / m * s))
part = {E: float(np.abs(gam[1:E + 1]).sum() / math.sqrt(E)) for E in [10, 100, 1000, 10000, N]}
tail = {E: float((np.abs(gam[E + 1:]) / np.arange(E + 1, N + 1)).sum() * math.sqrt(E)) for E in [10, 100, 1000, 10000]}
out['L1'] = dict(max_identity_error=err, sum_abs_gamma_over_sqrtE=part, tail_abs_over_e_times_sqrtE=tail,
                 sum_abs_gamma_over_phi2e=float((np.abs(gam[1:]) / np.array([m for m in range(1, N + 1)])).sum()))
print('L1', out['L1'])

def D2(f):  # second difference on an array indexed from 0: returns array g with g[j] = f[j+1]-2f[j]+f[j-1], j=1..len-2
    return f[2:] - 2 * f[1:-1] + f[:-2]

res_w = {}
for wn in ['1-x', '(1-x)^2', 'cos', '1-x^2']:
    w = weight(wn, N); D = conv_data(N, w)
    rho, nu, E, dlt = D['rho'], D['nu'], D['E'], D['dlt']
    r = {}
    # (d) Lemma 5(a)
    rr = np.concatenate([rho, [0.0, 0.0]]); nn = np.concatenate([nu, [0.0, 0.0]])
    d2r = D2(rr)   # index j -> k = j+1 ; k>=2 means j>=1
    d2n = D2(nn)   # n = j+1 ; n>=2 means j>=1
    r['L5a'] = dict(sum_D2rho_k2=float(np.abs(d2r[1:]).sum()), sum_D2nu_n2=float(np.abs(d2n[1:]).sum()),
                    D2nu_n1=float(d2n[0]), sup_grad_rho=float(np.abs(np.diff(rho)).max()),
                    sup_grad_nu=float(np.abs(np.diff(nu)).max()), sup_rho_over_N=float(rho.max() / N), sup_nu_over_N=float(nu.max() / N))
    if wn == '1-x':
        F1 = mult_sums(rho * E, 2 * N); F2 = mult_sums(nu * dlt, 2 * N); Delta = 2 * F1 - F2; Delta[0] = 0
        d = np.arange(2 * N + 1, dtype=float)
        sw = []
        for eps, delta in [(0.1, 0.05), (0.2, 0.2), (0.3, 0.1)]:
            Y = N ** (0.5 + eps); Th = smoothstep((d - Y) / (delta * Y))
            S_Theta = float((Th * beta * Delta).sum())
            z = np.zeros(2 * N + 1); z[1:] = Th[1:] * Delta[1:] / d[1:]
            T = mult_sums(z, 2 * N)
            expansion = 8 * C2 * float((gam[1:] * T[2:2 * N + 1:2][:N]).sum())
            # (c) switch for several M0
            swM = {}
            kk = np.arange(N + 1, dtype=float); nn2 = np.arange(2 * N + 1, dtype=float)
            for M0 in [2, 4, 6, 12, 18]:
                lmax = int(2 * N / Y) + 2; tot = 0.0; per_l = []
                for l in range(1, lmax + 1):
                    M = M0 * l
                    ks = kk[M::M]; ns = nn2[M::M]
                    pp = float(((l / ks) * smoothstep((ks / l - Y) / (delta * Y)) * rho[M::M] * E[M::M]).sum()) if len(ks) else 0.0
                    gb = float(((l / ns) * smoothstep((ns / l - Y) / (delta * Y)) * nu[M::M] * dlt[M::M]).sum()) if len(ns) else 0.0
                    tot += 2 * pp - gb; per_l.append((2 * pp - gb) / N ** 2)
                swM[M0] = dict(T_direct=float(T[M0] / N ** 2), T_switch=tot / N ** 2, abs_err=abs(T[M0] - tot) / N ** 2,
                               sum_abs_per_l=float(np.abs(per_l).sum()))
            sw.append(dict(eps=eps, delta=delta, S_Theta=S_Theta / N ** 2, expansion=expansion / N ** 2,
                           relerr=abs(S_Theta - expansion) / max(abs(S_Theta), 1e-300), switch=swM))
        r['steps1_3'] = sw
    # (e) Lemma 5(c)
    l5 = []
    for eps, delta in [(0.1, 0.05), (0.2, 0.05), (0.2, 0.2), (0.05, 0.02)]:
        Y = N ** (0.5 + eps)
        for l in sorted(set([1, 2, 3, 7, max(1, int(N / Y) // 2), max(1, int(2 * N / Y) - 1)])):
            kk = np.arange(N + 3, dtype=float); kk[0] = 1
            phi = (l / kk) * smoothstep((kk / l - Y) / (delta * Y)); phi[0] = 0
            hl = phi * np.concatenate([rho, [0, 0]]); s1 = float(np.abs(D2(hl)).sum())
            nn2 = np.arange(2 * N + 3, dtype=float); nn2[0] = 1
            phi2 = (l / nn2) * smoothstep((nn2 / l - Y) / (delta * Y)); phi2[0] = 0
            hp = phi2 * np.concatenate([nu, [0, 0]]); s2 = float(np.abs(D2(hp)).sum())
            pred = N / (delta * Y * Y * l) + 1 / Y
            l5.append(dict(eps=eps, delta=delta, l=l, sum_D2_h=s1, sum_D2_hprime=s2, pred=pred, ratio_h=s1 / pred, ratio_hprime=s2 / pred,
                           sup_h_over_NoverY=float(np.abs(hl).max() * Y / N), sup_hp_over_NoverY=float(np.abs(hp).max() * Y / N)))
    r['L5c'] = l5
    r['L5c_max_ratio_h'] = max(x['ratio_h'] for x in l5); r['L5c_max_ratio_hprime'] = max(x['ratio_hprime'] for x in l5)
    res_w[wn] = r
    print(wn, 'L5a', r['L5a'], 'L5c max ratios', r['L5c_max_ratio_h'], r['L5c_max_ratio_hprime'])
    if 'steps1_3' in r:
        for s in r['steps1_3']:
            print('  eps', s['eps'], 'delta', s['delta'], 'S_Theta', s['S_Theta'], 'relerr expansion', s['relerr'])
            for M0, v in s['switch'].items(): print('    M0', M0, v)
out['weights'] = res_w
json.dump(out, open(f'/home/claude/ariadna/R1/arbitro1_fable/arb_T2_{k}.json', 'w'), indent=1)
