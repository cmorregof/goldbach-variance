"""check_revision.py -- independent checks of claims in opus_moduli_medios.md (and fable_moduli_grandes.md when present).
Uses check_bloques.core() (own implementation of the F-sums), so this is not a copy of check_opus.py.
Usage: python3 -I check_revision.py        (~1 min)
"""
import sys, os, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bloques as cb

C2 = cb.C2

def beta_table(D):
    mu, phi, _ = cb.sieve_arith(D)
    # kappa_d = prod_{p not| d}(1 - 1/(p-1)^2) = C2-type product with the primes of d removed (p>2 only enters through (p-1)^2)
    pr = cb.sieve(D); kap_all = 1.0
    # product over ALL odd primes up to a large bound; tail handled by the zeta-type correction (error ~ 1/bound)
    big = cb.sieve(10 ** 7); odd = big[big > 2]
    kap_all = float(np.prod(1 - 1.0 / (odd - 1.0) ** 2))
    kap_all *= (1 - 1.0 / (1e7 * math.log(1e7)))
    beta = np.zeros(D + 1)
    for d in range(2, D + 1, 2):
        if mu[d] == 0: continue
        k = kap_all
        # remove the primes dividing d (p=2 factor does not occur in the product over odd primes)
        for p in cb.pfac(d):
            if p > 2: k /= (1 - 1.0 / (p - 1.0) ** 2)
        beta[d] = 2.0 * d * k / phi[d] ** 2
    return beta, mu, phi, kap_all

def check_lemma11(ar):
    D = 4096; beta, mu, phi, kap = beta_table(D)
    S = ar.S; mx = 0.0
    for k in range(1, 2001):
        s = 0.5 * sum(beta[d] for d in range(1, k + 1) if k % d == 0)
        mx = max(mx, abs(s - S[k]) / S[k] if S[k] > 0 else abs(s))
    print(f"[L1.1] max relative error of S(k)=1/2 sum_(d|k) beta(d), k<=2000 : {mx:.2e}   (kappa_all={kap:.10f}, C2={C2:.10f})")
    phi_f = np.maximum(phi, 1)
    print(f"[L1.1] sum_(d<=4096) beta/phi = {np.sum(beta[1:]/phi_f[1:]):.4f}   (2K = {2*2.300961544713:.4f}); "
          f"sum_(d<=y) beta - 2 log y at y=4096: {beta[1:].sum() - 2*math.log(4096):.3f}  (claimed ~1.15)")
    for D0 in (256, 1024, 2048):
        print(f"       sum_(D<d<=2D) beta, D={D0}: {beta[D0+1:2*D0+1].sum():.4f}  (2 log 2 = {2*math.log(2):.4f})")
    return beta

def check_thm12_and_FG(ar, beta_big, N, wname):
    cst = cb.weight_constants(cb.WEIGHTS[wname]); r0 = cst['r0']; cBw = cst['cBw']
    D = 2 * N
    sc = cb.core(ar, N, cb.WEIGHTS[wname], D)
    Delta = sc['FE'] - sc['Fd']
    mu, phi, _ = cb.sieve_arith(D)
    # beta on all d<=2N (computed with the product formula)
    kap = beta_big['kap']; beta = np.zeros(D + 1)
    big = cb.sieve(D + 5)
    # kappa_d via incremental removal
    for d in range(2, D + 1, 2):
        if mu[d] == 0: continue
        k = kap
        for p in cb.pfac(d):
            if p > 2: k /= (1 - 1.0 / (p - 1.0) ** 2)
        beta[d] = 2.0 * d * k / phi[d] ** 2
    Xd = float(np.sum(beta * Delta))
    print(f"[T1.2] w={wname} N={N}: X_w = {sc['X']:.6e}; sum_d beta(d) Delta(d) = {Xd:.6e}; rel.diff {abs(Xd-sc['X'])/abs(sc['X']):.2e}")
    # G >= 0 and FG term
    Gmod = sc['rho0'] * sc['psi20'] + sc['FM'] - sc['Fmod']
    G = Delta + Gmod
    ev = [d for d in range(2, D + 1, 2) if mu[d] != 0]
    ev_odd = [d for d in range(1, D + 1, 2) if mu[d] != 0]
    Gmin = min(G[d] / float(N) ** 2 for d in ev + ev_odd if d <= N // 2)
    print(f"[P1.3] min_(d<=N/2 squarefree) G_w(d)/N^2 = {Gmin:+.3e} (should be >= 0)")
    N2 = float(N) ** 2
    rows = []
    for d in (1, 2, 6, 10, 30, 70, 210, 330, 1010, 2002, 5002, 10010, 30030):
        if d > N // 2 or mu[d] == 0: continue
        wsum = 0.0
        for p in cb.pfac(d):
            if p > 2: wsum += math.log(p) / (p - 1.0)
        d0 = d // 2 if d % 2 == 0 else d
        GFG = r0 ** 2 * N2 * (math.log(d) - sum(math.log(p) / (p - 1.0) for p in cb.pfac(d))) - cBw * N2 if d % 2 == 0 else float('nan')
        rows.append((d, Gmod[d] / N2, GFG / N2, (Gmod[d] - GFG) / N2 if d % 2 == 0 else float('nan'), (Gmod[d] - GFG) / (d ** 0.5 * N ** 1.5) if d % 2 == 0 else float('nan')))
    print(f"[T2.1] w={wname} N={N}: G_mod(d)/N^2 vs G_FG(d)/N^2 (even d), difference, and difference/(d^1/2 N^3/2)")
    for r in rows: print(f"       d={r[0]:6d}  Gmod={r[1]:+9.5f}  GFG={r[2]:+9.5f}  diff={r[3]:+9.5f}  diff/(sqrt(d)N^1.5)={r[4]:+.4f}")
    return dict(Xd=Xd, X=sc['X'], Gmin=Gmin)

def block_cross(ar, N, wname):
    """q-block versus d-block: confirm the Moebius smearing statement (Opus sec 1): in the d-basis the same total, different blocks."""
    cst = cb.weight_constants(cb.WEIGHTS[wname]); D = 1 << 14
    tab = cb.sieve_arith(D); sc = cb.core(ar, N, cb.WEIGHTS[wname], D)
    bl, tail = cb.blocks_of(sc, D, cst, tab)
    print(f"[blocks] w={wname} N={N}: q-blocks (P+D)/N^2 for Q=16..8192: " + " ".join(f"{b['PD']:+.4f}" for b in bl if b['Q'] >= 16))

if __name__ == "__main__":
    ar = cb.Arith(1 << 16)
    beta = check_lemma11(ar)
    kap = float(np.prod(1 - 1.0 / (cb.sieve(10 ** 7)[1:] - 1.0) ** 2) * (1 - 1.0 / (1e7 * math.log(1e7))))
    for N, w in ((1 << 14, '1-x'), (1 << 15, '(1-x)^2'), (1 << 16, 'bump')):
        check_thm12_and_FG(ar, dict(kap=kap), N, w)
    block_cross(ar, 1 << 16, '1-x')
