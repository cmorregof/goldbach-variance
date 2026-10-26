"""T3: Steps 4-6 of E1_intento.md with actual Dirichlet characters, for M0 = 2e with e in {1, 2 (M0=4, non-squarefree), 3}
and all l < 2N/Y.  For each (e, l), M = 2 e l:
  prime-pair side:   Sigma_pp = sum_{M|k} h_l(k) psi2^w(k)   is split as
       (4a) pairs with (mn,M)>1, (4b) phi(M)^{-1} sum_{chi != chi0} S_chi, (4c) phi(M)^{-1} S_chi0,
       S_chi = sum_{m,n} conj(chi)(m) lambda(m) chi(n) lambda(n) h_l(n-m).
     Checks: the decomposition is exact; |S_chi| <= sum_k|D2 h_l| * sum_{u<=2N}|A_chi(u)|^2 (Lemma 4 + Lemma 6 shape);
     sizes of sum_u|A_chi(u)|^2/(N^2 L^2) (Lemma 6).
  deterministic side: Sigma_det = sum_{M|k} h_l(k) S(k) rho_w(k)  vs  phi(M)^{-1} sum_k h_l rho_w - (1/2) int f/x  (Steps 5-6, Lemma vaughanM
     with non-squarefree M when e=2).
  Goldbach side: the same with h'_l, n = m + n', chi(-1).
Then the total  2 sum_{M|k} h_l E_w - sum_{M|n} h'_l delta_w  is recombined from the pieces and compared with the direct value.
Usage: python3 arb_T3_caracteres.py k eps delta
"""
import sys, json, math, time
import numpy as np
sys.path.insert(0, '/home/claude/ariadna/R1/arbitro1_fable')
from arb_core import *

k = int(sys.argv[1]); eps = float(sys.argv[2]); delta = float(sys.argv[3])
N = 2 ** k; L = math.log(N); Y = N ** (0.5 + eps)
w = weight('1-x', N); D = conv_data(N, w)
rho, nu, E, dlt, S, psi2, R, lam, Lam = D['rho'], D['nu'], D['E'], D['dlt'], D['S'], D['psi2'], D['R'], D['lam'], D['Lam']
FL = 1
while FL < 4 * N + 4: FL *= 2


def factorize(n):
    f = {}; p = 2
    while p * p <= n:
        while n % p == 0: f[p] = f.get(p, 0) + 1; n //= p
        p += 1
    if n > 1: f[n] = f.get(n, 0) + 1
    return f


def primitive_root(q):  # q odd prime power
    f = factorize(q); p = list(f)[0]; phi = q // p * (p - 1)
    fac = factorize(phi)
    for g in range(2, q):
        if math.gcd(g, q) != 1: continue
        if all(pow(g, phi // r, q) != 1 for r in fac): return g
    raise RuntimeError


def characters(M):
    """Returns list of (chi array of length M, is_principal, chi(-1)). All characters mod M via CRT."""
    import itertools
    f = factorize(M); comps = []   # each comp: (q, list of functions x -> value)

    def mk_odd(q, g, phi):
        dlog = {pow(g, t, q): t for t in range(phi)}
        return [(lambda j, q=q, phi=phi, dlog=dlog: (lambda x: np.exp(2j * np.pi * j * dlog[x % q] / phi)))(j) for j in range(phi)]

    def mk_two(q, a):
        if a == 1: return [lambda x: 1.0 + 0j]
        if a == 2: return [lambda x: 1.0 + 0j, lambda x: (1.0 + 0j) if x % 4 == 1 else (-1.0 + 0j)]
        n2 = 2 ** (a - 2); tab = {}
        for s in range(2):
            for t in range(n2): tab[((-1) ** s * pow(5, t, q)) % q] = (s, t)
        return [(lambda j, u, q=q, n2=n2, tab=tab: (lambda x: ((-1) ** (j * tab[x % q][0])) * np.exp(2j * np.pi * u * tab[x % q][1] / n2)))(j, u)
                for j in range(2) for u in range(n2)]

    for p, a in f.items():
        q = p ** a
        if p == 2: comps.append((q, mk_two(q, a)))
        else: comps.append((q, mk_odd(q, primitive_root(q), q // p * (p - 1))))
    chis = []
    units = [x for x in range(M) if math.gcd(x, M) == 1]
    for combo in itertools.product(*[range(len(c[1])) for c in comps]):
        chi = np.zeros(M, complex)
        for x in units:
            v = 1.0 + 0j
            for (q, fs), j in zip(comps, combo): v *= fs[j](x)
            chi[x] = v
        chis.append(chi)
    # sanity: orthogonality
    U = np.array([c[units] for c in chis])
    G = U @ np.conj(U.T) / len(units)
    assert np.allclose(G, np.eye(len(chis)), atol=1e-9), f'characters mod {M} not orthonormal'
    assert len(chis) == len(units)
    out = []
    for chi in chis:
        princ = bool(np.allclose(chi[units], 1.0))
        out.append((chi, princ, chi[(M - 1) % M]))
    return out


def phi_l(x, l):
    x = np.asarray(x, float); xx = x.copy(); xx[xx == 0] = 1
    v = (l / xx) * smoothstep((xx / l - Y) / (delta * Y)); v[x == 0] = 0
    return v


def D2sum(f):
    return float(np.abs(f[2:] - 2 * f[1:-1] + f[:-2]).sum())


def sum_mult(x, M):  # sum_{j>=1} x[jM]
    return float(x[M::M].sum())


results = {}
kk = np.arange(N + 1, dtype=float); nn = np.arange(2 * N + 1, dtype=float)
idx = np.arange(N + 1)
t0 = time.time()
for e in [1, 2, 3]:
    M0 = 2 * e; lmax = int(2 * N / Y) + 1
    rows = []
    for l in range(1, lmax + 1):
        M = M0 * l
        hl = phi_l(kk, l) * rho                      # h_l(k), k=0..N
        hp = phi_l(nn, l) * nu                       # h'_l(n), n=0..2N
        if not (hl != 0).any() and not (hp != 0).any(): continue
        d2h = D2sum(np.concatenate([hl, [0, 0]])); d2hp = D2sum(np.concatenate([hp, [0, 0]]))
        # direct values
        Spp = sum_mult(hl * psi2, M); Sdet = sum_mult(hl * S[:N + 1] * rho, M)
        SG = sum_mult(hp * R, M); SGdet = sum_mult(hp * S * nu, M)
        # (4a) pairs with (mn,M)>1
        cop = np.array([math.gcd(int(m), M) == 1 for m in idx])
        lam_c = lam * cop; lam_nc = lam * (~cop)
        Fc = np.fft.rfft(lam_c, FL); Fnc = np.fft.rfft(lam_nc, FL)
        # cross-corr c(k) = sum_m a_m b_{m+k}: irfft(conj(Fa) Fb)
        corr_nc = np.fft.irfft(np.conj(Fnc) * Fnc, FL)[:N + 1]
        conv_nc = np.fft.irfft(Fnc * Fnc, FL)[:2 * N + 1]
        pp4a = sum_mult(hl * corr_nc, M); gb4a = sum_mult(hp * conv_nc, M)
        # characters
        chis = characters(M); phiM = len(chis)
        Snp = 0.0; SnpG = 0.0; S0 = 0.0; S0G = 0.0; maxratio = 0.0; maxA2 = 0.0; sum_abs_np = 0.0; sum_abs_npG = 0.0
        for chi, princ, chim1 in chis:
            chim = chi[idx % M]
            a = lam * chim                              # b_n = chi(n) lambda(n);  a_m = conj(chi)(m) lambda(m)
            Fa = np.fft.fft(a, FL)
            corr = np.fft.ifft(np.conj(Fa) * Fa, FL)[:N + 1]      # sum_m conj(a_m) a_{m+k} = sum conj(chi(m)) chi(m+k) lam lam
            Schi = complex((hl * corr).sum())
            A = np.cumsum(a); A2 = float((np.abs(A[1:]) ** 2).sum() + N * abs(A[N]) ** 2)   # sum_{u<=2N} |A_chi(u)|^2
            if princ:
                S0 = Schi.real
            else:
                Snp += Schi.real; sum_abs_np += abs(Schi)
                maxratio = max(maxratio, abs(Schi) / (d2h * A2)) if d2h > 0 else maxratio; maxA2 = max(maxA2, A2 / (N * N * L * L))
        # Goldbach side exactly: sum_{m+n = 0 mod M} lam lam h'(m+n) with (mn,M)=1 = phi^{-1} sum_chi chi(-1) sum_{m,n} conj(chi(m)) chi(n) lam lam h'(m+n)
        SnpG = 0.0; S0G = 0.0; maxratioG = 0.0
        for chi, princ, chim1 in chis:
            chim = chi[idx % M]
            a = lam * np.conj(chim); b = lam * chim
            Fa = np.fft.fft(a, FL); Fb = np.fft.fft(b, FL)
            conv = np.fft.ifft(Fa * Fb, FL)[:2 * N + 1]
            v = complex(chim1 * (hp * conv).sum())
            A2 = float((np.abs(np.cumsum(b)[1:]) ** 2).sum() + N * abs(np.cumsum(b)[N]) ** 2)
            if princ: S0G = v.real
            else:
                SnpG += v.real; sum_abs_npG += abs(v); maxratioG = max(maxratioG, abs(v) / (d2hp * A2)) if d2hp > 0 else maxratioG
        # principal-character main term and deterministic matching
        main_pp = float((hl * rho).sum()) / phiM; main_G = float((hp * nu).sum()) / phiM
        # secondary term: (1/2) int f/x, f = phi_l(x) N^2 r(x/N)^2 ; use the discrete sum of hl*rho/k as proxy of the integral
        xs = np.arange(1, N + 1, dtype=float)
        sec_pp = 0.5 * float((hl[1:] * rho[1:] / xs).sum())      # ~ (1/2) int phi_l rho^2 / x
        xs2 = np.arange(1, 2 * N + 1, dtype=float)
        sec_G = 0.5 * float((hp[1:] * nu[1:] / xs2).sum())
        # recombination check
        pp_from_pieces = pp4a + (Snp + S0) / phiM
        gb_from_pieces = gb4a + (SnpG + S0G) / phiM
        row = dict(l=l, M=M, phiM=phiM, d2h=d2h, d2hp=d2hp,
                   Spp=Spp / N ** 2, pp_recomb_err=(pp_from_pieces - Spp) / N ** 2, SG=SG / N ** 2, gb_recomb_err=(gb_from_pieces - SG) / N ** 2,
                   pp_4a=pp4a / N ** 2, pp_nonprinc=Snp / phiM / N ** 2, pp_nonprinc_absum=sum_abs_np / phiM / N ** 2,
                   pp_princ=S0 / phiM / N ** 2, pp_main=main_pp / N ** 2, pp_det=Sdet / N ** 2,
                   pp_princ_minus_main=(S0 / phiM - main_pp) / N ** 2,
                   pp_det_minus_pred=(Sdet - (main_pp - sec_pp)) / N ** 2, pp_sec=sec_pp / N ** 2,
                   gb_4a=gb4a / N ** 2, gb_nonprinc=SnpG / phiM / N ** 2, gb_nonprinc_absum=sum_abs_npG / phiM / N ** 2,
                   gb_princ=S0G / phiM / N ** 2, gb_main=main_G / N ** 2, gb_det=SGdet / N ** 2,
                   gb_princ_minus_main=(S0G / phiM - main_G) / N ** 2,
                   gb_det_minus_pred=(SGdet - (main_G - sec_G)) / N ** 2, gb_sec=sec_G / N ** 2,
                   bound_4b_per_char=d2h * N * N * L * L / N ** 2, max_ratio_Lemma4_pp=maxratio, max_ratio_Lemma4_gb=maxratioG,
                   max_A2_over_N2L2=maxA2,
                   E_side=(Spp - Sdet) / N ** 2, G_side=(SG - SGdet) / N ** 2)
        rows.append(row)
        print(f"e={e} l={l:3d} M={M:4d} phi={phiM:3d} Spp-Sdet={row['E_side']:+.3e} np={row['pp_nonprinc']:+.3e} (abs {row['pp_nonprinc_absum']:.3e}, bound/char {row['bound_4b_per_char']:.2e}) "
              f"princ-main={row['pp_princ_minus_main']:+.3e} det-pred={row['pp_det_minus_pred']:+.3e} sec={row['pp_sec']:.2e} | G: np={row['gb_nonprinc']:+.3e} princ-main={row['gb_princ_minus_main']:+.3e} det-pred={row['gb_det_minus_pred']:+.3e} "
              f"| r4={maxratio:.3f}/{maxratioG:.3f} A2={maxA2:.3f} recomb={row['pp_recomb_err']:.1e}/{row['gb_recomb_err']:.1e}  [{time.time()-t0:.0f}s]", flush=True)
    results[e] = rows
json.dump(dict(N=N, eps=eps, delta=delta, Y=Y, rows=results), open(f'/home/claude/ariadna/R1/arbitro1_fable/arb_T3_{k}_{eps}_{delta}.json', 'w'), indent=1)
