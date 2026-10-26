"""check_opus.py -- round 3, Opus: small and medium moduli.  See opus_moduli_medios.md.

Everything is exact (floating point) from Lambda; the singular series uses the FULL twin-prime constant.
Smoothed weights w(m/N), w(1)=0:  lam(m) = Lambda(m) w(m/N),  rho_w, nu_w, psi2_w, R_w as in lema_B_opus.md Sec. 1.1.
  E_w(h) = psi2_w(h) - S(h) rho_w(h)   (h != 0),   delta_w(n) = R_w(n) - S(n) nu_w(n).

THE DIVISOR (d-) BASIS (opus_moduli_medios.md Sec. 1).  With beta(d) = 2 d mu^2(d) kappa_d / phi(d)^2,
kappa_d = prod_{p not| d} (1 - (p-1)^-2)  (so beta(d) = 0 for odd d, beta(2d') = 4 C2 / prod_{p|d'} (p-2)),
one has S(k) = (1/2) sum_{d|k} beta(d) for k != 0, hence EXACTLY
      X_w = sum_{d <= 2N} beta(d) Delta_w(d),
      Delta_w(d) = sum_{h != 0, d|h} rho_w E_w(h) - sum_{d|n} nu_w delta_w(n)      (HL errors summed along d | h, d | n)
                 = G_w(d) - Gmod_w(d),
      G_w(d)    = sum_{d|h} rho_w psi2_w(h) - sum_{d|n} nu_w R_w(n)  >= 0      (windowed variance of primes mod d),
      Gmod_w(d) = rho_w(0) psi2_w(0) + sum_{h!=0,d|h} S rho_w^2 - sum_{d|n} S nu_w^2
                ~ G_FG(d) := r0^2 N^2 [log d0 - sum_{p|d0} log p/(p-1)] - cB_w N^2       (smoothed Friedlander-Goldston term)

Sections
  O1 (single N, several weights): identity X_w = sum beta Delta; d-blocks (D,2D] of beta*Delta, split into
      prime part P_d = sum beta G >= 0 and deterministic part D_d = -sum beta Gmod; q-block cross-check against
      check_B2 / check_bloques (P(Q), D(Q)); per-d statistics (rms of Delta, pair part, Goldbach part, sqrt(d)-scaling);
      the Friedlander-Goldston main term check.
  O2 (many N, w = 1-x): averaging in N.  For fixed d, how much of Delta_w(d;N) survives the N-average?
      Correlation of Delta_w(d;N)/N^2 in log N; block sums; d-average vs N-average.
  O3: scaling of |Delta_w(d)| with d (does |Delta_w(d)| decay like d^{-1/2}, as a random model of the HL errors predicts?).
  O4: long scan N in [2^14, 2^21]: spectrum of Delta_w(d;N)/N^2 in log N (zero-pair frequencies |gamma-gamma'|) and
      the fraction of it that survives averaging N over [Y,2Y] or [Y,8Y].
Usage: python3 check_opus.py [O1] [O2] [O3] [O4]      (all: ~5 min, < 1.5 GB)
Output: check_opus.json, stdout (saved as check_opus_output.txt).
"""
import sys, json, math, time
import numpy as np, scipy.fft as sf
from scipy.integrate import quad

C2 = 0.660161815846869573927812110015
GAMMA = 0.577215664901532860606512090082
OUTF = "/home/claude/pruebas/ronda3/check_opus.json"
OUT = {}

# --------------------------------------------------------------------------------------------- arithmetic
def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]

class Arith:
    def __init__(self, Nmax):
        pr = sieve(2 * Nmax + 2); self.pr = pr; self.Nmax = Nmax
        Lam = np.zeros(Nmax + 1)
        for p in pr[pr <= Nmax]:
            lp = math.log(p); pk = int(p)
            while pk <= Nmax: Lam[pk] = lp; pk *= int(p)
        self.Lam = Lam
        S = np.zeros(2 * Nmax + 2); S[0::2] = 2 * C2
        for p in pr[1:]: S[0::p] *= (p - 1) / (p - 2)
        S[0] = 0.0; self.S = S
        M = 2 * Nmax + 2                      # multiplicative data up to 2N+2
        mu = np.ones(M + 1, np.int8); phi = np.arange(M + 1, dtype=np.int64)
        g = np.ones(M + 1)                    # g(d') = prod_{p|d'} 1/(p-2)  (p odd), for beta
        lsum = np.zeros(M + 1)                # sum_{p | d, p odd} log p/(p-1)
        lodd = np.zeros(M + 1)                # log of odd part (for squarefree d)
        for p in pr[pr <= M]:
            p = int(p)
            mu[p::p] *= -1; mu[p * p::p * p] = 0
            phi[p::p] = phi[p::p] // p * (p - 1)
            if p > 2:
                g[p::p] /= (p - 2); lsum[p::p] += math.log(p) / (p - 1); lodd[p::p] += math.log(p)
        self.mu, self.phi, self.lsum, self.lodd = mu, phi, lsum, lodd
        beta = np.zeros(M + 1)
        d = np.arange(2, M + 1, 2)
        sqf = mu[d] != 0
        beta[d[sqf]] = 4 * C2 * g[d[sqf] // 2]
        self.beta = beta

def flen(N):
    L = 1
    while L < 2 * N + 2: L *= 2
    return L

# --------------------------------------------------------------------------------------------- weights
WEIGHTS = {
    '1-x':     lambda x: 1.0 - x,
    '(1-x)^2': lambda x: (1.0 - x) ** 2,
    'bump':    lambda x: np.exp(-x * x / np.maximum(1.0 - x * x, 1e-300)) * (x < 1),
}

_CST = {}
def constants(name):
    if name in _CST: return _CST[name]
    w = WEIGHTS[name]; wf = lambda x: float(w(np.array(x)))
    r = lambda u: quad(lambda x: wf(x) * wf(x + u), 0, 1 - u, limit=200)[0]
    g = lambda u: quad(lambda x: wf(x) * wf(u - x), max(0.0, u - 1), min(1.0, u), limit=200)[0]
    r0 = r(0.0)
    h = quad(lambda u: (r(u) ** 2 - r0 ** 2) / u, 0, 1, limit=200)[0]
    gam = quad(lambda u: g(u) ** 2 / u, 0, 2, points=[1.0], limit=200)[0]
    kap = quad(lambda x: wf(x) ** 2 * math.log(x), 0, 1, limit=200)[0]
    cBw = r0 ** 2 * (GAMMA + math.log(2 * math.pi)) - r0 * kap + h - gam / 2
    _CST[name] = dict(r0=r0, h=h, gamma=gam, kappa=kap, cBw=cBw)
    return _CST[name]

# --------------------------------------------------------------------------------------------- core
def core(ar, N, name):
    """E, delta, weighted arrays for weight `name` at N."""
    m = np.arange(N + 1); w = np.where(m >= 1, WEIGHTS[name](m / N), 0.0)
    lam = ar.Lam[:N + 1] * w; L = flen(N)
    FL = sf.rfft(lam, L, workers=2); Fw = sf.rfft(w, L, workers=2)
    R = sf.irfft(FL * FL, L, workers=2)[:2 * N + 1]
    psi2 = sf.irfft(FL * np.conj(FL), L, workers=2)[:N + 1].copy()
    nu = sf.irfft(Fw * Fw, L, workers=2)[:2 * N + 1]
    rho = sf.irfft(Fw * np.conj(Fw), L, workers=2)[:N + 1].copy()
    del FL, Fw
    S = ar.S
    aE = np.zeros(N + 1); aE[1:] = rho[1:] * (psi2[1:] - S[1:N + 1] * rho[1:])      # rho E   (h >= 1)
    bD = nu * (R - S[:2 * N + 1] * nu); bD[0] = 0.0                                   # nu delta (n >= 1)
    X = 4 * (S[1:N + 1] * aE[1:]).sum() - 2 * (S[:2 * N + 1] * bD).sum()
    out = dict(N=N, aE=aE, bD=bD, X=X, rho0=rho[0], psi20=psi2[0])
    out['aG'] = np.zeros(N + 1); out['aG'][1:] = rho[1:] * psi2[1:]
    out['bG'] = nu * R; out['bG'][0] = 0.0
    out['aM'] = np.zeros(N + 1); out['aM'][1:] = S[1:N + 1] * rho[1:] ** 2
    out['bM'] = S[:2 * N + 1] * nu * nu; out['bM'][0] = 0.0
    return out

def delta_all(c, dmax, full=False, even_only=False, sqf=None):
    """Delta(d) = 2 sum_{h>=1,d|h} rho E - sum_{n>=1,d|n} nu delta, its pair and Goldbach parts, and (if full)
       G(d), Gmod(d)."""
    aE, bD = c['aE'], c['bD']
    TE = np.zeros(dmax + 1); TD = np.zeros(dmax + 1)
    G = np.zeros(dmax + 1) if full else None; Gm = np.zeros(dmax + 1) if full else None
    base = c['rho0'] * c['psi20']
    rng = range(2, dmax + 1, 2) if even_only else range(1, dmax + 1)
    for d in rng:
        if sqf is not None and not sqf[d]: continue
        TE[d] = 2 * aE[d::d].sum(); TD[d] = bD[d::d].sum()
        if full:
            G[d] = base + 2 * c['aG'][d::d].sum() - c['bG'][d::d].sum()
            Gm[d] = base + 2 * c['aM'][d::d].sum() - c['bM'][d::d].sum()
    return TE, TD, G, Gm

def G_FG(ar, d, N, cst):
    """smoothed Friedlander-Goldston main term, d squarefree (array)."""
    return cst['r0'] ** 2 * N * N * (ar.lodd[d] - ar.lsum[d]) - cst['cBw'] * N * N

def dyadic_edges(lo, hi):
    e = [lo]
    while e[-1] * 2 <= hi: e.append(e[-1] * 2)
    return e

# --------------------------------------------------------------------------------------------- O1
def divisors_upto(D):
    divs = [[] for _ in range(D + 1)]
    for d in range(1, D + 1):
        for m in range(d, D + 1, d): divs[m].append(d)
    return divs

def O1(ar, logs=(16, 18, 20, 21), names=('1-x', '(1-x)^2', 'bump')):
    print("=" * 110); print("O1  d-basis: identity, d-blocks, q-block cross-check, per-d statistics, Friedlander-Goldston term"); print("=" * 110)
    res = {}
    Dq = 4096; divs = divisors_upto(Dq)
    for name in names:
        cst = constants(name)
        print(f"weight {name}: r0={cst['r0']:.6f} h={cst['h']:.6f} gamma_w={cst['gamma']:.6f} kappa={cst['kappa']:.6f} cB_w={cst['cBw']:.6f}")
        for lg in logs:
            if name != '1-x' and lg == 21: continue
            N = 1 << lg; t = time.time(); N2 = float(N) ** 2
            c = core(ar, N, name)
            M = 2 * N
            sqf = ar.mu[:M + 1] != 0
            # full Delta for all squarefree d <= Dfull (odd too, for q-blocks), even squarefree up to 2N
            Dfull = min(Dq, M)
            TE, TD, G, Gm = delta_all(c, Dfull, full=True, sqf=sqf)
            TE2, TD2, _, _ = delta_all(c, M, even_only=True, sqf=sqf)
            TE2[:Dfull + 1] = 0; TD2[:Dfull + 1] = 0
            TE2[:Dfull + 1] = TE; TD2[:Dfull + 1] = TD
            Dl = TE2 - TD2                                     # Delta(d), d <= 2N (odd d only up to Dfull)
            beta = ar.beta[:M + 1]
            Xd = float((beta * Dl).sum())
            r = dict(N=N, X_N2=c['X'] / N2, Xd_N2=Xd / N2, ident_relerr=(Xd - c['X']) / c['X'])
            # d-blocks
            blocks = []
            # first block d in {1,2} (only d=2 has beta != 0), then (D,2D] for D = 2,4,...
            edges = [0, 2] + [2 ** j for j in range(2, lg + 2)]
            lo = 0
            for hi in edges[1:]:
                ds = np.arange(lo + 1, hi + 1); ds = ds[(ds % 2 == 0) & sqf[ds]]
                if len(ds) == 0: lo = hi; continue
                bsum = float((beta[ds] * Dl[ds]).sum()) / N2
                row = dict(D=lo, n=int(len(ds)), sum=bsum, wsum=float(beta[ds].sum()),
                           pair=float((beta[ds] * TE2[ds]).sum()) / N2, gold=float(-(beta[ds] * TD2[ds]).sum()) / N2,
                           rms=float(np.sqrt(np.mean(Dl[ds] ** 2))) / N2, maxabs=float(np.abs(Dl[ds]).max()) / N2,
                           mean=float(np.mean(Dl[ds])) / N2, mean_phi=float(np.mean(Dl[ds] * ar.phi[ds])) / N2,
                           rms_sqrtd=float(np.sqrt(np.mean(ds * Dl[ds] ** 2))) / N2,
                           rms_pair_sqrtd=float(np.sqrt(np.mean(ds * TE2[ds] ** 2))) / N2,
                           rms_gold_sqrtd=float(np.sqrt(np.mean(ds * TD2[ds] ** 2))) / N2,
                           corr_pair_gold=float(np.corrcoef(TE2[ds], TD2[ds])[0, 1]) if (len(ds) > 2 and TE2[ds].std() > 0 and TD2[ds].std() > 0) else float('nan'))
                if hi <= Dfull:
                    row['P_d'] = float((beta[ds] * G[ds]).sum()) / N2
                    row['D_d'] = float(-(beta[ds] * Gm[ds]).sum()) / N2
                    row['D_FG'] = float(-(beta[ds] * G_FG(ar, ds, N, cst)).sum()) / N2
                    row['G_minus_FG_mean'] = float(np.mean(G[ds] - G_FG(ar, ds, N, cst))) / N2
                    row['Gm_minus_FG_mean'] = float(np.mean(Gm[ds] - G_FG(ar, ds, N, cst))) / N2
                    gfg = G_FG(ar, ds, N, cst); pos = gfg > 0.05 * N2
                    row['Gmin_over_FG'] = float(np.min(G[ds][pos] / gfg[pos])) if pos.any() else float('nan')
                blocks.append(row); lo = hi
            r['dblocks'] = blocks
            cum = np.cumsum([b['sum'] for b in blocks])
            # q-blocks from the Delta, G, Gmod of all d <= Dq (Moebius), for cross-check with check_B2/check_bloques
            mu, phi = ar.mu, ar.phi
            Xq = np.zeros(Dfull + 1); Pq = np.zeros(Dfull + 1)
            for q in range(1, Dfull + 1):
                if mu[q] == 0: continue
                cq = 2.0 / float(phi[q]) ** 2
                Xq[q] = cq * sum(d * int(mu[q // d]) * Dl[d] for d in divs[q])
                Pq[q] = cq * sum(d * int(mu[q // d]) * G[d] for d in divs[q])
            qb = []; Q = 1
            while 2 * Q <= Dfull:
                qs = [q for q in range(Q + 1, 2 * Q + 1) if mu[q] != 0] if Q > 1 else [1, 2]
                qb.append(dict(Q=Q if Q > 1 else 0, P=float(Pq[qs].sum()) / N2, D=float((Xq[qs] - Pq[qs]).sum()) / N2,
                               sum=float(Xq[qs].sum()) / N2))
                Q = 2 * Q if Q > 1 else 2
            r['qblocks'] = qb
            r['sum_q_le_Dq'] = float(Xq.sum()) / N2
            print(f"--- w={name:8s} N=2^{lg}: X_w/N^2={c['X']/N2:+.5f}   sum_d beta(d)Delta(d)/N^2={Xd/N2:+.5f}  (rel.err {r['ident_relerr']:.1e})   ({time.time()-t:.0f}s)")
            print("      d-block (D,2D] | #d | sum beta | sum beta*Delta/N^2 | cum | P_d=sum beta G | D_d=-sum beta Gmod | D_FG | mean(G-G_FG) | mean(Gmod-G_FG) | min G/G_FG | rms Delta, max|Delta| | rms sqrt(d)Delta (pair, Goldbach) | corr(pair,Gold)")
            for b, cu in zip(blocks, cum):
                extra = (f"{b['P_d']:8.4f} {b['D_d']:9.4f} {b['D_FG']:9.4f} {b['G_minus_FG_mean']:+8.4f} {b['Gm_minus_FG_mean']:+8.4f} {b['Gmin_over_FG']:6.3f}"
                         if 'P_d' in b else " " * 58)
                print(f"      {b['D']:8d} {b['n']:7d} {b['wsum']:7.4f} {b['sum']:+10.5f} {cu:+9.5f} | {extra} | {b['rms']:.2e} max {b['maxabs']:.3f} {b['rms_sqrtd']:.3f} ({b['rms_pair_sqrtd']:.3f},{b['rms_gold_sqrtd']:.3f}) {b['corr_pair_gold']:+.3f}")
            sN = math.sqrt(N); K2 = 2 * 2.300961544713
            med_d = sum(b['sum'] for b in blocks if b['D'] >= 16 and 2 * b['D'] <= sN)
            med_dabs = sum(b['wsum'] * b['rms'] for b in blocks if b['D'] >= 16 and 2 * b['D'] <= sN)
            r['medium_d_sum'] = med_d; r['medium_d_sum_beta_rms'] = med_dabs; r['X_over_2KcB'] = c['X'] / N2 / (K2 * cst['cBw'])
            print(f"      medium d-blocks (16 < d <= sqrt N): sum beta*Delta/N^2 = {med_d:+.5f}  vs  sum beta*rms|Delta|/N^2 = {med_dabs:.4f};"
                  f"   X_w/(2K cB_w N^2) = {r['X_over_2KcB']:.4f}")
            print("      q-blocks (Moebius from the same Delta; compare check_bloques):  Q: P  D  P+D")
            print("        " + "  ".join(f"[{b['Q']}: {b['P']:.4f} {b['D']:+.4f} {b['sum']:+.5f}]" for b in qb))
            res[f"{name}|{lg}"] = r; sys.stdout.flush()
            del c
    OUT['O1'] = res

# --------------------------------------------------------------------------------------------- O2
def O2(ar, lo=1 << 17, hi=1 << 19, K=64, name='1-x', dmax=2048):
    print("=" * 110); print(f"O2  averaging in N:  w={name}, {K} log-spaced N in [{lo},{hi}], even squarefree d <= {dmax}"); print("=" * 110)
    Ns = np.unique(np.round(np.exp(np.linspace(math.log(lo), math.log(hi), K))).astype(int))
    sqf = ar.mu != 0
    ds = np.array([d for d in range(2, dmax + 1, 2) if sqf[d]])
    beta = ar.beta[ds]
    Dmat = np.zeros((len(Ns), len(ds))); t = time.time()
    for i, N in enumerate(Ns):
        c = core(ar, int(N), name); N2 = float(N) ** 2
        TE, TD, _, _ = delta_all(c, dmax, even_only=True, sqf=sqf)
        Dmat[i] = (TE[ds] - TD[ds]) / N2
        del c
    print(f"  computed {len(Ns)} values of N ({time.time()-t:.0f}s)")
    logN = np.log(Ns.astype(float))
    res = dict(Ns=[int(x) for x in Ns], blocks=[])
    # per d: N-mean (bias), N-sd (fluctuation), autocorrelation in log N
    mean_N = Dmat.mean(0); sd_N = Dmat.std(0)
    resid = Dmat - mean_N
    def acorr(lag):
        a = resid[:-lag] if lag else resid; b = resid[lag:]
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    dlog = (logN[-1] - logN[0]) / (len(Ns) - 1)
    lags = [1, 2, 4, 8, 16, 32]
    edges = [2, 4] + [2 ** j for j in range(3, int(math.log2(dmax)) + 1)]
    print("  d-block | #d | rms_d of Delta(d;N)/N^2 (all N) | rms_d of N-mean | rms_d of N-sd | ratio N-mean/total |"
          " block sum: mean over N, sd over N | autocorr of resid at lags (in log N) " + " ".join(f"{l*dlog:.3f}" for l in lags))
    lo_e = 0
    for hi_e in edges:
        sel = (ds > lo_e) & (ds <= hi_e)
        if sel.sum() == 0: lo_e = hi_e; continue
        tot = float(np.sqrt(np.mean(Dmat[:, sel] ** 2))); mn = float(np.sqrt(np.mean(mean_N[sel] ** 2)))
        sdv = float(np.sqrt(np.mean(sd_N[sel] ** 2)))
        bs = (Dmat[:, sel] * beta[sel]).sum(1)
        ac = []
        for l in lags:
            a = resid[:-l, sel]; b = resid[l:, sel]
            ac.append(float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())))
        row = dict(D=lo_e, n=int(sel.sum()), rms_all=tot, rms_Nmean=mn, rms_Nsd=sdv, ratio=mn / tot,
                   block_mean=float(bs.mean()), block_sd=float(bs.std()), acorr=dict(zip([l * dlog for l in lags], ac)))
        res['blocks'].append(row)
        print(f"  ({lo_e:5d},{hi_e:5d}] {row['n']:4d}  {tot:.3e}  {mn:.3e}  {sdv:.3e}  {mn/tot:.3f}   {row['block_mean']:+.5f} {row['block_sd']:.5f}   "
              + " ".join(f"{a:+.2f}" for a in ac))
        lo_e = hi_e
    # cross-d correlation of the N-means (do biases of different d cancel?) and d-average vs N-average
    med = (ds > 16) & (ds <= 1024)
    tot_med = (Dmat[:, med] * beta[med]).sum(1)
    res['medium_total'] = dict(mean=float(tot_med.mean()), sd=float(tot_med.std()),
                               sum_abs_beta_Nmean=float((beta[med] * np.abs(mean_N[med])).sum()),
                               sum_beta_Nmean=float((beta[med] * mean_N[med]).sum()),
                               sum_beta_rms=float((beta[med] * np.sqrt((Dmat[:, med] ** 2).mean(0))).sum()))
    m = res['medium_total']
    print(f"  medium d in (16,1024]:  sum beta Delta/N^2 over N: mean {m['mean']:+.5f} sd {m['sd']:.5f};"
          f"  sum beta |N-mean of Delta| = {m['sum_abs_beta_Nmean']:.4f}  vs |sum beta N-mean| = {abs(m['sum_beta_Nmean']):.5f};"
          f"  sum beta rms(Delta) = {m['sum_beta_rms']:.4f}")
    OUT['O2'] = res

# --------------------------------------------------------------------------------------------- O4
def O4(ar, lo=1 << 14, hi=1 << 21, K=112, name='1-x', dmax=1024):
    """Long scan in log N: spectrum of Delta_w(d;N)/N^2 as a function of u = log N, averaged over d.
       Zero heuristic: Delta(d;N)/N^2 = sum over pairs of zeros of L(s,chi), chi mod d, of N^{i(gamma-gamma')} K(gamma,gamma');
       the frequencies in u are gamma-gamma'.  Dyadic N-averaging multiplies frequency f by sinc(f log2 / 2)."""
    print("=" * 110); print(f"O4  long scan: {K} log-spaced N in [{lo},{hi}], w={name}, even squarefree d <= {dmax}"); print("=" * 110)
    Ns = np.unique(np.round(np.exp(np.linspace(math.log(lo), math.log(hi), K))).astype(int))
    u = np.log(Ns.astype(float)); sqf = ar.mu != 0
    ds = np.array([d for d in range(2, dmax + 1, 2) if sqf[d]])
    Dm = np.zeros((len(Ns), len(ds))); t = time.time()
    for i, N in enumerate(Ns):
        c = core(ar, int(N), name); TE, TD, _, _ = delta_all(c, dmax, even_only=True, sqf=sqf)
        Dm[i] = (TE[ds] - TD[ds]) / float(N) ** 2; del c
    print(f"  ({time.time()-t:.0f}s)")
    res = {}
    du = (u[-1] - u[0]) / (len(u) - 1)
    for lab, sel in (('16<d<=128 (medium for all N in scan)', (ds > 16) & (ds <= 128)),
                     ('128<d<=1024', (ds > 128) & (ds <= 1024))):
        x = Dm[:, sel] - Dm[:, sel].mean(0)
        # detrend-free periodogram with a Hann taper, averaged over d
        win = np.hanning(len(u))[:, None]
        F = np.fft.rfft(x * win, axis=0); pw = (np.abs(F) ** 2).mean(1)
        fr = 2 * np.pi * np.fft.rfftfreq(len(u), du)            # angular frequency in u = log N, i.e. |gamma - gamma'|
        pw[0] = 0
        cum = np.cumsum(pw) / pw.sum()
        fmed = float(fr[np.searchsorted(cum, 0.5)]); fpk = float(fr[np.argmax(pw)])
        # dyadic moving average over [u, u+log 2]
        wlen = max(1, int(round(math.log(2) / du)))
        ker = np.ones(wlen) / wlen
        ma = np.array([np.convolve(Dm[:, j], ker, mode='valid') for j in np.nonzero(sel)[0]]).T
        raw = Dm[wlen - 1:, sel]
        surv = float(np.sqrt((ma ** 2).mean()) / np.sqrt((raw ** 2).mean()))
        # same for an 8-adic window
        wl8 = max(1, int(round(math.log(8) / du)))
        ma8 = np.array([np.convolve(Dm[:, j], np.ones(wl8) / wl8, mode='valid') for j in np.nonzero(sel)[0]]).T
        surv8 = float(np.sqrt((ma8 ** 2).mean()) / np.sqrt((Dm[wl8 - 1:, sel] ** 2).mean()))
        # the same for the fluctuating part only (per-d mean over the whole scan removed)
        Xc = Dm[:, sel] - Dm[:, sel].mean(0)
        mac = np.array([np.convolve(Xc[:, j], ker, mode='valid') for j in range(Xc.shape[1])]).T
        survc = float(np.sqrt((mac ** 2).mean()) / np.sqrt((Xc[wlen - 1:] ** 2).mean()))
        mac8 = np.array([np.convolve(Xc[:, j], np.ones(wl8) / wl8, mode='valid') for j in range(Xc.shape[1])]).T
        survc8 = float(np.sqrt((mac8 ** 2).mean()) / np.sqrt((Xc[wl8 - 1:] ** 2).mean()))
        mean_all = float(Dm[:, sel].mean())
        res[lab] = dict(n_d=int(sel.sum()), rms=float(np.sqrt((Dm[:, sel] ** 2).mean())), mean=mean_all,
                        freq_peak=fpk, freq_median_power=fmed, frac_power_below_1=float(cum[np.searchsorted(fr, 1.0)]),
                        dyadic_survival=surv, octo_survival=surv8, dyadic_survival_fluct=survc, octo_survival_fluct=survc8,
                        resolution=float(2 * np.pi / (u[-1] - u[0])),
                        spectrum=dict(freq=[float(f) for f in fr[:40]], power=[float(p) for p in (pw / pw.sum())[:40]]))
        print(f"  {lab}: #d={sel.sum()}  rms Delta/N^2={res[lab]['rms']:.4f}  mean {mean_all:+.4f}")
        print(f"     spectrum in u=log N: peak at |gamma-gamma'| ~ {fpk:.2f}, median of power at {fmed:.2f}, fraction of power below 1: {res[lab]['frac_power_below_1']:.2f}")
        print(f"     (frequency resolution {2*np.pi/(u[-1]-u[0]):.2f})   rms after averaging N over [Y,2Y]: {surv:.3f} of raw;   over [Y,8Y]: {surv8:.3f} of raw")
        print(f"     fluctuating part only (per-d scan mean removed): survives [Y,2Y]-averaging at {survc:.3f}, [Y,8Y]-averaging at {survc8:.3f}")
    OUT['O4'] = res

# --------------------------------------------------------------------------------------------- O3
def O3():
    print("=" * 110); print("O3  scaling of |Delta_w(d)| with d (from O1): fit rms(sqrt(d) Delta / N^2) and rms(Delta/N^2) ~ C d^-a"); print("=" * 110)
    res = {}
    for key, r in OUT.get('O1', {}).items():
        bl = [b for b in r['dblocks'] if b['D'] >= 16 and b['n'] >= 8]
        N = r['N']
        for rng_name, sel in (('medium d<=sqrtN', [b for b in bl if 2 * b['D'] <= math.sqrt(N)]),
                              ('sqrtN<d<=N/4', [b for b in bl if b['D'] >= math.sqrt(N) and 2 * b['D'] <= N / 4])):
            if len(sel) < 3: continue
            x = np.log([1.5 * b['D'] for b in sel]); y = np.log([b['rms'] for b in sel])
            a, c0 = np.polyfit(x, y, 1)
            res[f"{key}|{rng_name}"] = dict(exponent=float(-a), C=float(math.exp(c0)))
            print(f"  {key:16s} {rng_name:18s}: rms(Delta/N^2) ~ {math.exp(c0):.3f} d^-{-a:.3f}   (H_AP random model: exponent 1/2)")
    OUT['O3'] = res

if __name__ == "__main__":
    which = sys.argv[1:] or ['O1', 'O2', 'O3', 'O4']
    t0 = time.time(); ar = Arith(1 << 21)
    print(f"arith done ({time.time()-t0:.1f}s); C2={C2}")
    if 'O1' in which: O1(ar)
    if 'O2' in which: O2(ar)
    if 'O3' in which: O3()
    if 'O4' in which: O4(ar)
    try:
        old = json.load(open(OUTF))
    except Exception:
        old = {}
    old.update(OUT)
    json.dump(old, open(OUTF, "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")
