"""E1_bloques.py -- R1-E1, run 2: numerical checks for the d-tail L_w = sum_{d>N^{1/2+eps}} beta(d) Delta_w(d).

Normalisation is that of manuscrito_v5_en.tex (Lemma dbasis):
  Delta_w(d) = 2 sum_{h>=1, d|h} rho_w(h) E_w(h) - sum_{n>=1, d|n} nu_w(n) delta_w(n),
  beta(d)    = 4 C2 prod_{p | d/2} (p-2)^{-1}  (d even squarefree), 0 otherwise,
  X_w        = sum_d beta(d) Delta_w(d)  (checked against the direct formula).

Checks (all values in units of N^2 unless said otherwise):
  C1  identity X_w = sum beta Delta (relative error).
  C2  dyadic blocks B(D) = sum_{D<d<=2D} beta(d) Delta_w(d), D = 2^j up to 2N; compared with
      the Goldston-Vaughan-type scale N^{5/4} D^{3/4} suggested by the audit.
  C3  L_w(Y) for Y = N^{1/2+eps}, eps in EPS, and max_{d>Y} |Delta_w(d)|.
  C4  smooth-cutoff version S_Theta(Y,delta) and the expansion
        S_Theta = 8 C2 sum_e gamma(e) T(2e),   T(M0) = sum_{M0 | d} Theta(d) Delta_w(d) / d,
      with the truncation e <= E (tail size as a function of E).
  C5  kernel lemmas: sum_{k>=2} |D^2 rho_w(k)|, sum_n |D^2 nu_w(n)|, max |grad rho_w|, max |grad nu_w|.
  C6  after the switch, for e = 1 (M0 = 2): sum over l of |sum_{2l | k} h_l(k) E_w(k)| (prime-pair side) and the
      Goldbach analogue, with h_l(k) = (l/k) Theta(k/l) rho_w(k): no cancellation over l is used in the proof.
Usage: python3 E1_bloques.py k1 k2 ...   (N = 2^k).  Writes E1_bloques_<k>.json.
"""
import sys, json, math, time
import numpy as np, scipy.fft as sf

C2 = 0.660161815846869573927812110015
EPS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]


def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]


def flen(N):
    L = 1
    while L < 2 * N + 2: L *= 2
    return L


def weight(name, N):
    x = np.arange(N + 1) / N
    w = {'box': np.ones(N + 1), '1-x': 1 - x, '(1-x)^2': (1 - x) ** 2}[name]
    w = w.copy(); w[0] = 0.0
    return w


def mult_sums(x, Dmax):
    """F[d] = sum_{m>=1} x[m d] for d = 1..Dmax (x[0] ignored)."""
    K = len(x) - 1; F = np.zeros(Dmax + 1); D0 = int(math.isqrt(K))
    for d in range(1, min(D0, Dmax) + 1): F[d] = x[d::d].sum()
    if Dmax > D0:
        for m in range(1, K // (D0 + 1) + 1):
            hi = min(K // m, Dmax)
            if hi <= D0: break
            idx = np.arange(D0 + 1, hi + 1); F[idx] += x[m * idx]
    return F


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)   # C^2 step, 0 for u<=0, 1 for u>=1


def run(k):
    N = 2 ** k; t0 = time.time(); out = dict(N=N, k=k)
    pr = sieve(2 * N + 2)
    Lam = np.zeros(N + 1)
    for p in pr[pr <= N]:
        lp = math.log(p); pk = int(p)
        while pk <= N: Lam[pk] = lp; pk *= int(p)
    S = np.zeros(2 * N + 1); S[0::2] = 2 * C2
    for p in pr[1:]:
        if p <= 2 * N: S[0::p] *= (p - 1) / (p - 2)
    S[0] = 0.0
    # beta(d), d <= 2N ; gamma(e) for b = 1*gamma, e <= N
    D2 = 2 * N
    sqf = np.ones(D2 + 1, bool); sqf[0] = False
    for p in pr:
        if p * p <= D2: sqf[p * p::p * p] = False
    beta = np.zeros(D2 + 1); beta[2::2] = 4 * C2
    for p in pr[1:]:
        if p <= D2: beta[2 * p::2 * p] /= (p - 2)
    beta[~sqf] = 0.0
    gam = np.ones(N + 1); gam[0] = 0.0
    gam[2::2] *= -1.0; gam[4::4] = 0.0                         # gamma(2) = -1, gamma(2^k)=0 (k>=2)
    for p in pr[1:]:
        if p > N: break
        gam[p::p] *= 2.0 / (p - 2)                              # gamma(p) = 2/(p-2)
        if p * p <= N:
            gam[p * p::p * p] *= -p / 2.0                       # gamma(p^2) = -p/(p-2) = (2/(p-2)) * (-p/2)
            if p ** 3 <= N: gam[p ** 3::p ** 3] = 0.0
    out['sum_abs_gamma_over_e'] = float(np.sum(np.abs(gam[1:]) / np.arange(1, N + 1)))
    res = {}
    for wname in ['1-x', '(1-x)^2', 'box']:
        w = weight(wname, N); lam = Lam * w; L = flen(N)
        FL = sf.rfft(lam, L, workers=2); Fw = sf.rfft(w, L, workers=2)
        R = sf.irfft(FL * FL, L, workers=2)[:2 * N + 1]
        psi2 = sf.irfft(FL * np.conj(FL), L, workers=2)[:N + 1].copy()
        nu = sf.irfft(Fw * Fw, L, workers=2)[:2 * N + 1]
        rho = sf.irfft(Fw * np.conj(Fw), L, workers=2)[:N + 1].copy()
        E = psi2 - S[:N + 1] * rho; E[0] = 0.0
        dG = R - S * nu
        Xdir = 4 * (S[:N + 1] * rho * E).sum() - 2 * (S * nu * dG).sum()
        F1 = mult_sums(rho * E, D2); F2 = mult_sums(nu * dG, D2)
        Delta = 2 * F1 - F2; Delta[0] = 0.0
        Xd = float((beta * Delta).sum())
        r = dict(X_direct=Xdir / N ** 2, X_dbasis=Xd / N ** 2, relerr=abs(Xd - Xdir) / abs(Xdir))
        # C2 dyadic blocks
        blocks = []
        D = 1
        while D <= 2 * N:
            sl = slice(D + 1, min(2 * D, D2) + 1)
            B = float((beta[sl] * Delta[sl]).sum())
            Babs = float((beta[sl] * np.abs(Delta[sl])).sum())
            blocks.append(dict(D=D, B=B / N ** 2, B_abs=Babs / N ** 2,
                               GVscale=(N ** 1.25 * D ** 0.75) / N ** 2))
            D *= 2
        r['blocks'] = blocks
        # C3 L_w(Y)
        cum = np.cumsum((beta * Delta)[::-1])[::-1]             # cum[d] = sum_{d' >= d}
        LY = []
        for eps in EPS:
            Y = N ** (0.5 + eps); iY = int(math.floor(Y)) + 1
            LY.append(dict(eps=eps, Y=Y, L=float(cum[iY]) / N ** 2,
                           maxDelta=float(np.abs(Delta[iY:] * (beta[iY:] > 0)).max()) / N ** 2))
        r['L_Y'] = LY
        # C5 kernel second differences
        rr = np.concatenate([rho, np.zeros(3)]); nn = np.concatenate([nu, np.zeros(3)])
        d2r = rr[3:] - 2 * rr[2:-1] + rr[1:-2]                  # Delta^2 rho(k), k = 2..N+1
        d2n = nn[2:] - 2 * nn[1:-1] + nn[:-2]                   # Delta^2 nu(n), n = 1..2N+1
        r['kernel'] = dict(sum_d2rho_k2=float(np.abs(d2r).sum()), sum_d2nu=float(np.abs(d2n).sum()),
                           max_grad_rho=float(np.abs(np.diff(rho[1:])).max()),
                           max_grad_nu=float(np.abs(np.diff(nu)).max()))
        # C4 smooth cutoff and gamma expansion, and C6 switch sizes (only eps = 0.1, 0.2; weights 1-x)
        if wname != 'box':
            c4 = []
            for eps, delta in [(0.1, 0.05), (0.2, 0.05), (0.2, 0.2)]:
                Y = N ** (0.5 + eps); d = np.arange(D2 + 1)
                Th = smoothstep((d - Y) / (delta * Y))
                Ssm = float((Th * beta * Delta).sum())
                z = np.zeros(D2 + 1); z[1:] = Th[1:] * Delta[1:] / d[1:]
                T = mult_sums(z, D2)                            # T[M0] = sum_{M0|d} Theta Delta / d
                Tev = T[2:D2 + 1:2][:N]                         # T(2e), e = 1..N
                terms = 8 * C2 * gam[1:N + 1] * Tev
                full = float(terms.sum())
                trunc = []
                for Ee in [1, 2, 4, 8, 16, 64, 256, 1024]:
                    if Ee <= N: trunc.append(dict(E=Ee, partial=float(terms[:Ee].sum()) / N ** 2,
                                                  tail=float(full - terms[:Ee].sum()) / N ** 2))
                # C6 switch, e = 1 (M0 = 2): per-l sums
                kk = np.arange(N + 1, dtype=float); nn2 = np.arange(D2 + 1, dtype=float)
                lmax = int(2 * N / Y) + 1
                pp_abs = 0.0; gb_abs = 0.0; pp_sum = 0.0; gb_sum = 0.0
                for l in range(1, lmax + 1):
                    M = 2 * l
                    ks = kk[M::M]
                    if len(ks):
                        hl = (l / ks) * smoothstep((ks / l - Y) / (delta * Y)) * rho[M::M]
                        v = float((hl * E[M::M]).sum()); pp_abs += abs(v); pp_sum += v
                    ns = nn2[M::M]
                    hp = (l / ns) * smoothstep((ns / l - Y) / (delta * Y)) * nu[M::M]
                    v = float((hp * dG[M::M]).sum()); gb_abs += abs(v); gb_sum += v
                c4.append(dict(eps=eps, delta=delta, S_smooth=Ssm / N ** 2, expansion=full / N ** 2,
                               relerr=abs(full - Ssm) / max(abs(Ssm), 1e-300), trunc=trunc,
                               switch_e1=dict(T2=float(Tev[0]) / N ** 2, pp_sum_over_l=pp_sum / N ** 2,
                                              gb_sum_over_l=gb_sum / N ** 2,
                                              recomb=(2 * pp_sum - gb_sum) / N ** 2,
                                              pp_abs_over_l=pp_abs / N ** 2, gb_abs_over_l=gb_abs / N ** 2,
                                              lmax=lmax)))
            r['smooth'] = c4
        res[wname] = r
        del FL, Fw, R, psi2, nu, rho, E, dG, F1, F2
    out['res'] = res; out['seconds'] = time.time() - t0
    return out


if __name__ == '__main__':
    ks = [int(a) for a in sys.argv[1:]] or [14, 16]
    for k in ks:
        o = run(k)
        with open(f'/home/claude/ariadna/R1/run2_E1/E1_bloques_{k}.json', 'w') as f: json.dump(o, f, indent=1)
        print(k, 'done in', round(o['seconds'], 1), 's', flush=True)
