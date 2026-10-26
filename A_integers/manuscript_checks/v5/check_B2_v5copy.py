"""check_B2.py -- independent numerical checks for Lemma B (Opus, round 2).  See lema_B_opus.md.

All quantities are computed exactly (up to floating point) from Lambda, with the singular series built
from the FULL twin-prime constant C2 = 0.66016181584686957... (not the product over p <= 2N).

Notation (BRIEF.md): box [1,N]^2, X = 2 sum_{h!=0} M E - 2 sum_n S nu delta, X = sum_q X(q).
e(y) = psi(y) - y.  For squarefree q and (a,q)=1:  e_a(y) = sum_{m<=y} Lambda(m) e(am/q) - mu(q) y/phi(q).

Sections
  S1 (single N): the q = 1 formula X(1) = 2 Psi_1 + 2 c_B N^2 + err, with the exact split
        Psi_1 = (N-1)(e(N)-ubar)^2 + sum_k (u_k-ubar)^2 ,  u_k = e(k)+e(N-k),  ubar = mean u_k;
     per-dyadic-block split of sum_q X(q) into prime part  P = sum 2mu^2/phi^2 W(q) >= 0  and the
     deterministic part  D = sum det(q)  (GAP 3 made visible);  the character decomposition of W(3)
     (tests the chi(-1) sign in Fable's Prop. 5).
  S2 (many N, box): regression of X, X(1), X_{<=30}, X_{>30} against the endpoint functionals
        End_1   = 2(N-1)(e(N)-ubar)^2                         (principal character, q = 1)
        End_Q   = 2(N-1) sum_{q<=Q} mu^2/phi^2 sum_a |e_a(N) - vbar_a|^2     (all characters, q <= Q)
     predicted slopes: X(1) on End_1: 1;  X on End_1: K = 2.30;  X_{<=30} on End_30: 1;  X_{>30}: ~0.
     Also N-averages.
  S3 (smoothed box, w(x) = 1-x, so w(1) = 0): the identity VarG_w = A0_w + Ep_w + SS_w + X_w, the Mellin
     prediction for SS_w (Lemma A's method with the exact polynomial kernels), X_w(1) = 2[W_w(1) - ...],
     and the regression of X_w on the endpoint functionals (predicted slope 0).
  S0: Fable's T6 regression sample re-analysed with block-bootstrap standard errors.
Usage:  python3 check_B2.py            (everything; ~5 min on 2 cores)
        python3 check_B2.py S0|S1|S2|S3
Output: check_B2.json, and stdout (saved as check_B2_output.txt).
"""
import sys, json, math, time
import numpy as np, scipy.fft as sf
from numpy.polynomial import Polynomial as P

C2 = 0.660161815846869573927812110015
GAMMA = 0.577215664901532860606512090082
cB = 0.5 + GAMMA + math.log(math.pi / 2)
A2, A1, A0C, KCONST = -0.5, -2.109638600933, -2.654066259065, 2.300961544713   # Lemma A constants
OUT = {}

# ----------------------------------------------------------------------------------------- basics
def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]

class Arith:
    def __init__(self, Nmax):
        self.Nmax = Nmax
        pr = sieve(2 * Nmax + 2); self.pr = pr
        Lam = np.zeros(Nmax + 1)
        for p in pr[pr <= Nmax]:
            lp = math.log(p); pk = int(p)
            while pk <= Nmax: Lam[pk] = lp; pk *= int(p)
        self.Lam = Lam; self.cumpsi = np.cumsum(Lam)
        S = np.zeros(2 * Nmax + 2); S[0::2] = 2 * C2
        for p in pr[1:]: S[0::p] *= (p - 1) / (p - 2)
        S[0] = 0.0; self.S = S

def mobius_phi(D):
    mu = np.ones(D + 1, int); phi = np.arange(D + 1); isp = np.ones(D + 1, bool); isp[:2] = False
    for p in range(2, D + 1):
        if not isp[p]: continue
        isp[p * p::p] = False; mu[p::p] *= -1; mu[p * p::p * p] = 0; phi[p::p] = phi[p::p] // p * (p - 1)
    return mu, phi

def pfac(q):
    f = []; p = 2
    while p * p <= q:
        if q % p == 0:
            f.append(p)
            while q % p == 0: q //= p
        p += 1
    if q > 1: f.append(q)
    return f

def flen(N):
    L = 1
    while L < 2 * N + 2: L *= 2
    return L

def nu_box(N):
    n = np.arange(2 * N + 1); return np.where(n <= N + 1, np.maximum(n - 1, 0), 2 * N - n + 1).astype(float)

# ----------------------------------------------------------------------------------------- box core
def box_core(ar, N, D):
    """X, the F-sums per d <= D, X(q), det(q) for q <= D."""
    Lam = ar.Lam[:N + 1]; L = flen(N)
    F = sf.rfft(Lam, L, workers=2)
    R = sf.irfft(F * F, L, workers=2)[:2 * N + 1]
    psi2 = sf.irfft(F * np.conj(F), L, workers=2)[:N + 1].copy(); del F
    S = ar.S[:2 * N + 1]; nu = nu_box(N)
    h = np.arange(1, N + 1); M = S[1:N + 1] * (N - h); E = psi2[1:] - M; dG = R - S * nu
    X = 4 * (M * E).sum() - 2 * (S * nu * dG).sum()
    wE = (N - h) * E; wM = (N - h) * M; wd = nu * dG; wmod = nu * nu * S
    FE = np.zeros(D + 1); Fd = np.zeros(D + 1); FM = np.zeros(D + 1); Fmod = np.zeros(D + 1)
    for d in range(1, D + 1):
        FE[d] = 2 * wE[d - 1::d].sum(); FM[d] = 2 * wM[d - 1::d].sum()
        Fd[d] = wd[0::d].sum(); Fmod[d] = wmod[0::d].sum()
    return dict(X=X, psi20=psi2[0], FE=FE, Fd=Fd, FM=FM, Fmod=Fmod, N=N)

def per_q(core, D, mu, phi, divs):
    N = core['N']; FE, Fd, FM, Fmod, psi20 = core['FE'], core['Fd'], core['FM'], core['Fmod'], core['psi20']
    Xq = np.zeros(D + 1); Zq = np.zeros(D + 1); Zp = np.zeros(D + 1)
    for q in range(1, D + 1):
        if mu[q] == 0: continue
        c = 2.0 / phi[q] ** 2
        Xq[q] = c * sum(d * mu[q // d] * (FE[d] - Fd[d]) for d in divs[q])
        Zq[q] = c * sum(d * mu[q // d] * (Fmod[d] - FM[d]) for d in divs[q]) - 2 * N * psi20 / phi[q]
        Zp[q] = (2.0 / phi[q]) * N * N * (cB - sum(p * (p - 2) * math.log(p) / (p - 1) ** 2 for p in pfac(q) if p > 2))
    return Xq, Zq, Zp

def divisor_lists(D):
    divs = [[] for _ in range(D + 1)]
    for d in range(1, D + 1):
        for m in range(d, D + 1, d): divs[m].append(d)
    return divs

def psi1_split(ar, N):
    cp = ar.cumpsi; k = np.arange(1, N); eN = cp[N] - N
    ek = cp[k] - k; eNk = cp[N - k] - (N - k); u = ek + eNk
    Psi1 = ((cp[N] - cp[k] - cp[N - k]) ** 2).sum()
    ub = u.mean()
    return dict(Psi1=Psi1, eN=eN, ubar=ub, endpoint=(N - 1) * (eN - ub) ** 2, bulk=((u - ub) ** 2).sum())

def endpoint_functional(ar, N, Qmax, mu, phi):
    """End_q := (N-1) sum_{a*} |e_a(N) - vbar_a|^2 for squarefree q <= Qmax (exact)."""
    m = np.arange(1, N + 1); Lam = ar.Lam[1:N + 1]; out = {}
    for q in range(1, Qmax + 1):
        if mu[q] == 0: continue
        b0 = np.bincount(m % q, weights=Lam, minlength=q)
        b1 = np.bincount(m % q, weights=Lam * (N - m), minlength=q)
        r = np.arange(q); tot = 0.0
        for a in range(1, q + 1):
            if math.gcd(a, q) != 1: continue
            ph = np.exp(2j * np.pi * a * r / q)
            eaN = (ph * b0).sum() - mu[q] * N / phi[q]
            Za = (ph * b1).sum() - mu[q] / phi[q] * N * (N - 1) / 2
            vbar = 2 * Za.real / (N - 1)
            tot += abs(eaN - vbar) ** 2
        out[q] = (N - 1) * tot
    return out

# ----------------------------------------------------------------------------------------- S1
def S1(ar, logs=(16, 18, 20, 22)):
    print("=" * 100); print("S1  single N: q=1 formula, Psi_1 split, dyadic blocks (GAP 3), W(3) character test"); print("=" * 100)
    res = {}
    for lg in logs:
        N = 1 << lg; D = 8192 if lg <= 20 else 2048
        mu, phi = mobius_phi(D); divs = divisor_lists(D)
        t = time.time(); core = box_core(ar, N, D); Xq, Zq, Zp = per_q(core, D, mu, phi, divs)
        sp = psi1_split(ar, N); N2 = float(N) ** 2
        X1 = Xq[1]; pred1 = 2 * sp['Psi1'] + 2 * cB * N2
        r = dict(N=N, X_N2=core['X'] / N2, X1_N2=X1 / N2, pred1_N2=pred1 / N2, err1_N15=(X1 - pred1) / N ** 1.5,
                 eN=sp['eN'], eN2_N=sp['eN'] ** 2 / N, ubar_sqrtN=sp['ubar'] / math.sqrt(N),
                 Psi1_N2=sp['Psi1'] / N2, endpoint_N2=sp['endpoint'] / N2, bulk_N2=sp['bulk'] / N2)
        print(f"N=2^{lg}: X/N^2={r['X_N2']:.4f}  X(1)/N^2={r['X1_N2']:.4f}  (2Psi_1+2c_B N^2)/N^2={r['pred1_N2']:.4f}  "
              f"[X(1)-pred]/N^1.5={r['err1_N15']:+.2f}")
        print(f"     e(N)={sp['eN']:.1f}  e^2/N={r['eN2_N']:.4f}  ubar/sqrt(N)={r['ubar_sqrtN']:+.4f}  Psi_1/N^2={r['Psi1_N2']:.4f} = "
              f"endpoint (N-1)(e-ubar)^2/N^2 {r['endpoint_N2']:.4f} + bulk {r['bulk_N2']:.4f}")
        # dyadic blocks
        blocks = []; Q = 1
        while 2 * Q <= D:
            qs = [q for q in range(Q + 1, 2 * Q + 1) if mu[q] != 0] if Q > 1 else [1, 2]
            Pp = sum(Xq[q] - Zq[q] for q in qs); Dd = sum(Zq[q] for q in qs); Dpred = sum(Zp[q] for q in qs)
            blocks.append(dict(Q=Q if Q > 1 else 0, prime=Pp / N2, det=Dd / N2, det_pred=Dpred / N2, sum=(Pp + Dd) / N2))
            Q = 2 * Q if Q > 1 else 2
        tail = core['X'] - Xq[1:D + 1].sum()
        r['blocks'] = blocks; r['tail_gt_D_N2'] = tail / N2; r['D'] = D
        print("     dyadic blocks (q in (Q,2Q]; first row q in {1,2}):  prime part P=sum 2W/phi^2 >= 0 | det D | det predicted (Prop.3) | P+D")
        for b in blocks:
            print(f"       Q={b['Q']:5d}:  P/N^2={b['prime']:8.4f}   D/N^2={b['det']:8.4f}   Dpred/N^2={b['det_pred']:8.4f}   (P+D)/N^2={b['sum']:+8.4f}")
        print(f"       tail q>{D}: {r['tail_gt_D_N2']:+.4f} N^2;  2 log 2 = {2*math.log(2):.4f} (predicted growth of P per block per unit of log Q... see text)")
        # W(3) by characters (only for lg <= 20)
        if lg <= 20:
            r['W3'] = W3_test(ar, N, Xq[3] - Zq[3])
        print(f"     ({time.time()-t:.1f}s)"); res[lg] = r; sys.stdout.flush()
    OUT['S1'] = res

def W3_test(ar, N, Xminus_det):
    """W(3) computed (i) from X(3)-det(3) = 2/phi(3)^2 W(3), (ii) directly from e_a, (iii) by characters
       with the factor chi(-1) (correct) and without it (Fable's Prop. 5)."""
    q = 3; Lam = ar.Lam[:N + 1]; m = np.arange(N + 1); k = np.arange(1, N)
    tot = 0.0; ea = {}
    for a in (1, 2):
        c = np.cumsum(Lam * np.exp(2j * np.pi * a * m / q)); e = c - (-1 / 2) * m      # mu(3)/phi(3) = -1/2
        ea[a] = e
        tot += 2 * (e[N].imag) ** 2 + (np.abs(e[N] - e[k] - np.conj(e[N - k])) ** 2).sum()
    W_direct = tot; W_from_X = Xminus_det * 4 / 2
    chi = np.zeros(N + 1); chi[m % 3 == 1] = 1; chi[m % 3 == 2] = -1
    psichi = np.cumsum(Lam * chi); psi0 = np.cumsum(Lam * (m % 3 != 0))
    b = np.cumsum(Lam * (m % 3 == 0))                                   # powers of 3: e(a 3^j/3) = 1
    alpha = -0.5 * ((psi0[N] - psi0[k] - psi0[N - k]) - 0.0) + (b[N] - b[k] - b[N - k])   # real; the -y terms cancel
    tau2 = 3.0
    out = {}
    for s, name in ((-1.0, 'chi(-1)=-1 (correct)'), (1.0, 'no chi(-1) (Prop.5 as written)')):
        beta2 = tau2 / 4 * (psichi[N] - psichi[k] - s * psichi[N - k]) ** 2
        Wk = 2 * (alpha ** 2 + beta2).sum()
        out[name] = Wk
    Wk_direct = tot - sum(2 * (ea[a][N].imag) ** 2 for a in (1, 2))
    N2 = float(N) ** 2
    print(f"     W(3)/N^2: from X(3)-det(3): {W_from_X/N2:.6f}   direct: {W_direct/N2:.6f}   k>=1 part direct {Wk_direct/N2:.6f}  vs characters:"
          + "  ".join(f"[{nm}] {v/N2:.6f}" for nm, v in out.items()))
    return dict(W_from_X=W_from_X / N2, W_direct=W_direct / N2, Wk_direct=Wk_direct / N2, **{nm: v / N2 for nm, v in out.items()})

# ----------------------------------------------------------------------------------------- S2
def ols(x, y):
    A = np.vstack([np.ones_like(x), x]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c, float(np.std(y - A @ c))

def boot(x, y, B=2000, blk=8, seed=1):
    rng = np.random.default_rng(seed); n = len(x); nb = int(np.ceil(n / blk)); sl = []
    for _ in range(B):
        st = rng.integers(0, n - blk + 1, nb); idx = (st[:, None] + np.arange(blk)[None, :]).ravel()[:n]
        sl.append(ols(x[idx], y[idx])[0][1])
    return float(np.std(sl))

def S2(ar, ranges=((1 << 17, 1 << 18, 400), (1 << 20, 1 << 21, 100))):
    print("=" * 100); print("S2  many N (box): does the endpoint term survive in X?  regressions"); print("=" * 100)
    QE = 30; mu, phi = mobius_phi(QE); divs = divisor_lists(QE)
    Ksq = {Q: sum(mu[q] ** 2 / phi[q] ** 3 for q in range(1, Q + 1)) for Q in (1, 2, 3, 6, 10, 30)}
    res = {}
    for (lo, hi, cnt) in ranges:
        t = time.time(); rows = []
        for N in np.linspace(lo, hi, cnt).astype(int):
            N = int(N); core = box_core(ar, N, QE); Xq, Zq, _ = per_q(core, QE, mu, phi, divs)
            sp = psi1_split(ar, N); en = endpoint_functional(ar, N, QE, mu, phi); N2 = float(N) ** 2
            End = {Q: 2 * sum(mu[q] ** 2 / phi[q] ** 2 * en[q] for q in en if q <= Q) / N2 for Q in (1, 2, 3, 6, 10, 30)}
            rows.append(dict(N=N, X=core['X'] / N2, X1=Xq[1] / N2, Xle30=Xq[1:].sum() / N2, eN2=sp['eN'] ** 2 / N,
                             end1=2 * sp['endpoint'] / N2, **{f"End{Q}": v for Q, v in End.items()},
                             W1bulk=2 * sp['bulk'] / N2))
        cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        cols['Xgt30'] = cols['X'] - cols['Xle30']; cols['Xge2'] = cols['X'] - cols['X1']
        fits = {}
        def rep(yname, xname, pred):
            c, s = ols(cols[xname], cols[yname]); se = boot(cols[xname], cols[yname])
            fits[f"{yname}~{xname}"] = dict(a=float(c[0]), b=float(c[1]), b_se_block=se, resid_sd=s, sd=float(np.std(cols[yname])), pred_b=pred)
            print(f"  {yname:6s} ~ a + b*{xname:6s}: a={c[0]:7.3f}  b={c[1]:6.3f} (+-{se:.3f} block-bootstrap)  pred b={pred}  resid sd={s:.3f} (total sd {np.std(cols[yname]):.3f})")
        print(f"range N in [{lo},{hi}], {cnt} values; (e(N)^2/N) in [{cols['eN2'].min():.3f},{cols['eN2'].max():.3f}]; End_1 = 2(N-1)(e-ubar)^2/N^2 in [{cols['end1'].min():.3f},{cols['end1'].max():.3f}]")
        rep('X1', 'end1', '1'); rep('X1', 'eN2', '2')
        rep('X', 'eN2', f"2K={2*KCONST:.3f}"); rep('X', 'end1', f"K={KCONST:.3f}")
        rep('Xge2', 'end1', f"K-1={KCONST-1:.3f}")
        for Q in (1, 3, 6, 10, 30):
            rep('Xle30', f"End{Q}", '1 (Q=30)' if Q == 30 else f'~{Ksq[30]/Ksq[Q]:.2f} if only principal')
        rep('X', 'End30', '1'); rep('Xgt30', 'End30', '0'); rep('Xgt30', 'end1', f"K-K30={KCONST-Ksq[30]:.3f}")
        avg = {k: float(np.mean(v)) for k, v in cols.items() if k != 'N'}
        print("  N-averages: " + "  ".join(f"{k}={v:.3f}" for k, v in avg.items() if k in ('X', 'X1', 'Xle30', 'Xgt30', 'end1', 'End30', 'W1bulk')))
        print(f"  ({time.time()-t:.0f}s)"); sys.stdout.flush()
        res[f"{lo}-{hi}"] = dict(fits=fits, averages=avg, rows=[{k: float(v) for k, v in r.items()} for r in rows])
    OUT['S2'] = res

# ----------------------------------------------------------------------------------------- S3 smoothed
def smooth_core(ar, N, D=1):
    """w_m = (N-m)/N, 1<=m<=N (w(x)=1-x, w(1)=0).  Returns the five terms, X_w(1), W_w(1)."""
    m = np.arange(N + 1); w = np.where(m >= 1, (N - m) / N, 0.0)
    lam = ar.Lam[:N + 1] * w; L = flen(N)
    FL = sf.rfft(lam, L, workers=2); Fw = sf.rfft(w, L, workers=2)
    R = sf.irfft(FL * FL, L, workers=2)[:2 * N + 1]
    psi2 = sf.irfft(FL * np.conj(FL), L, workers=2)[:N + 1].copy()
    nu = sf.irfft(Fw * Fw, L, workers=2)[:2 * N + 1]
    rho = sf.irfft(Fw * np.conj(Fw), L, workers=2)[:N + 1].copy()
    Acorr = sf.irfft(FL * np.conj(Fw), L, workers=2)          # A(h) = sum_m lam(m) w(m-h) at index h mod L
    del FL, Fw
    S = ar.S[:2 * N + 1]; Mw = S[1:N + 1] * rho[1:]; E = psi2[1:] - Mw; dG = R - S * nu
    A0 = psi2[0] ** 2; Ep = 2 * (E ** 2).sum(); SS = 2 * (Mw ** 2).sum() - ((S * nu) ** 2).sum()
    X = 4 * (Mw * E).sum() - 2 * (S * nu * dG).sum(); Var = (dG ** 2).sum()
    X1 = 4 * (rho[1:] * E).sum() - 2 * (nu * dG).sum()
    k = np.arange(1, N); Ak = Acorr[k]; Amk = Acorr[L - k]
    W1 = ((Ak - Amk) ** 2).sum()
    FM1 = 2 * (S[1:N + 1] * rho[1:] ** 2).sum(); Fmod1 = (S * nu * nu).sum()
    det1 = 2 * (-rho[0] * psi2[0] - (FM1 - Fmod1))
    P2 = 2 * (S[1:N + 1] ** 2 * rho[1:] ** 2).sum(); G = ((S * nu) ** 2).sum()
    out = dict(N=N, A0=A0, Ep=Ep, SS=SS, X=X, Var=Var, X1=X1, W1=W1, det1=det1, rho0=rho[0], psi20=psi2[0],
               P2=P2, G=G, rho=rho, nu=nu)
    if D > 1:     # per-d sums for the q-decomposition of X_w (same algebra as the box, weights rho_w, nu_w)
        wE = rho[1:] * E; wM = rho[1:] * Mw; wd = nu * dG; wmod = nu * nu * S
        FE = np.zeros(D + 1); Fd = np.zeros(D + 1); FM = np.zeros(D + 1); Fmod = np.zeros(D + 1)
        for d in range(1, D + 1):
            FE[d] = 2 * wE[d - 1::d].sum(); FM[d] = 2 * wM[d - 1::d].sum()
            Fd[d] = wd[0::d].sum(); Fmod[d] = wmod[0::d].sum()
        out.update(FE=FE, Fd=Fd, FM=FM, Fmod=Fmod)
    return out

def smooth_blocks(sc, D, cst):
    """dyadic blocks for X_w: prime part sum 2W_w(q)/phi^2 (>=0) and det_w(q), with the smoothed Prop. 3 prediction
       det_w(q) ~ (2/phi(q)) N^2 [c_B^w - r0^2 sum_{p|q,p>2} w'_p]."""
    N = sc['N']; mu, phi = mobius_phi(D); divs = divisor_lists(D); N2 = float(N) ** 2
    Xq = np.zeros(D + 1); Zq = np.zeros(D + 1); Zp = np.zeros(D + 1)
    for q in range(1, D + 1):
        if mu[q] == 0: continue
        c = 2.0 / phi[q] ** 2
        Xq[q] = c * sum(d * mu[q // d] * (sc['FE'][d] - sc['Fd'][d]) for d in divs[q])
        Zq[q] = c * sum(d * mu[q // d] * (sc['Fmod'][d] - sc['FM'][d]) for d in divs[q]) - 2 * sc['rho0'] * sc['psi20'] / phi[q]
        Zp[q] = (2.0 / phi[q]) * N2 * (cst['cBw'] - cst['r0'] ** 2 * sum(p * (p - 2) * math.log(p) / (p - 1) ** 2 for p in pfac(q) if p > 2))
    blocks = []; Q = 1
    while 2 * Q <= D:
        qs = [q for q in range(Q + 1, 2 * Q + 1) if mu[q] != 0] if Q > 1 else [1, 2]
        Pp = sum(Xq[q] - Zq[q] for q in qs); Dd = sum(Zq[q] for q in qs); Dp = sum(Zp[q] for q in qs)
        blocks.append(dict(Q=Q if Q > 1 else 0, prime=Pp / N2, det=Dd / N2, det_pred=Dp / N2, sum=(Pp + Dd) / N2,
                           min_prime_q=float(min(Xq[q] - Zq[q] for q in qs) / N2)))
        Q = 2 * Q if Q > 1 else 2
    return blocks, (sc['X'] - Xq[1:].sum()) / N2

def poly_kernels(N):
    u = P([0, 1]); Nf = float(N)
    t = Nf * u; M = Nf - t
    rho = ((M - 1) * M * (2 * M - 1) / 6 + t * M * (M - 1) / 2) / Nf ** 2
    n = Nf * u
    nuL = ((n - 1) * (Nf * Nf - Nf * n) + n * n * (n - 1) / 2 - (n - 1) * n * (2 * n - 1) / 6) / Nf ** 2
    c = 2 * Nf - n + 1; x = n - Nf - 1
    S2N = Nf * (Nf + 1) * (2 * Nf + 1) / 6; S2x = x * (x + 1) * (2 * x + 1) / 6
    nuR = (c * (Nf * Nf - Nf * n) + n * n * c / 2 - (S2N - S2x)) / Nf ** 2
    return rho, nuL, nuR

def intpoly(Pu, a, b):      # int_a^b P(u) du
    Q = Pu.integ(); return Q(b) - Q(a)

def int_ulogu(Pu, a, b):    # int_a^b P(u) log u du
    tot = 0.0
    for kk, ck in enumerate(Pu.coef):
        F = lambda x: 0.0 if x == 0 else x ** (kk + 1) * (math.log(x) / (kk + 1) - 1.0 / (kk + 1) ** 2)
        tot += ck * (F(b) - F(a))
    return tot

def mellin_pred(N):
    """Lemma A's residue formula for P2_w = sum S^2 f1(k), f1 = 2 rho^2 on [0,N], and G_w = sum S^2 f2, f2 = nu^2."""
    rho, nuL, nuR = poly_kernels(N); Lg = math.log(N); Nf = float(N)
    f1 = 2 * rho * rho; f10 = f1(0.0)
    Q1 = P((f1 - f10).coef[1:])                       # (f1(Nu)-f1(0))/u
    H0 = intpoly(Q1, 0, 1); H1 = Lg * H0 + int_ulogu(Q1, 0, 1)
    int_f1 = Nf * intpoly(f1, 0, 1)
    P2p = KCONST * int_f1 + A2 * (f10 * Lg ** 2 / 2 + H1) + A1 * (f10 * Lg + H0) + A0C * f10
    I0 = 0.0; I1 = 0.0; int_f2 = 0.0
    for poly, ua, ub in ((nuL * nuL, 1 / Nf, (Nf + 1) / Nf), (nuR * nuR, (Nf + 1) / Nf, (2 * Nf + 1) / Nf)):
        p0 = poly(0.0); Qp = P((poly - p0).coef[1:])
        ta, tb = Nf * ua, Nf * ub
        I0 += p0 * math.log(tb / ta) + intpoly(Qp, ua, ub)
        I1 += p0 * (math.log(tb) ** 2 - math.log(ta) ** 2) / 2 + Lg * intpoly(Qp, ua, ub) + int_ulogu(Qp, ua, ub)
        int_f2 += Nf * intpoly(poly, ua, ub)
    Gp = KCONST * int_f2 + A2 * I1 + A1 * I0
    return dict(P2=P2p, G=Gp, SS=P2p - Gp, rho_poly=rho, nuL=nuL, nuR=nuR, f10=f10, int_f1=int_f1, int_f2=int_f2)

def smooth_constants():
    """continuous-kernel constants for w(x)=1-x: r0, h(0), gamma0, kappa, c_ss, c_w (see lema_B_opus.md, Thm S)."""
    from scipy.integrate import quad
    r = lambda u: (1 - u) ** 2 * (2 + u) / 6
    def g(u):
        lo, hi = max(0.0, u - 1), min(1.0, u)
        return quad(lambda x: (1 - x) * (1 - (u - x)), lo, hi)[0]
    r0 = r(0); h0 = quad(lambda u: (r(u) ** 2 - r0 ** 2) / u, 0, 1)[0]
    gam0 = quad(lambda u: g(u) ** 2 / u, 0, 2, points=[1])[0]
    kappa = quad(lambda x: (1 - x) ** 2 * math.log(x), 0, 1)[0]
    css = 2 * r0 ** 2 * A1 - h0 + gam0 / 2
    bM = -h0                                   # int_0^1 (r^2)'(u) log u du = -int_0^1 (r^2-r0^2)/u du
    cBw = r0 ** 2 * (GAMMA + math.log(2 * math.pi)) - r0 * kappa - bM - gam0 / 2
    return dict(r0=r0, h0=h0, gamma0=gam0, kappa=kappa, c_ss=css, c_w=2 * r0 * kappa + css, cBw=cBw, two_cBw=2 * cBw)

def S3(ar, logs=(16, 18, 20, 21), many=(1 << 17, 1 << 18, 400)):
    print("=" * 100); print("S3  smoothed box w(x)=1-x:  identity, Mellin prediction of SS_w, X_w, endpoint regressions"); print("=" * 100)
    cst = smooth_constants(); print("  constants:", {k: round(v, 6) for k, v in cst.items()})
    res = dict(constants=cst, single={})
    for lg in logs:
        N = 1 << lg; DB = 4096 if lg <= 20 else 1024
        sc = smooth_core(ar, N, D=DB); mp = mellin_pred(N); N2 = float(N) ** 2; Lg = math.log(N)
        rp = mp['rho_poly']; kk = np.arange(0, N + 1)
        rho_err = float(np.max(np.abs(rp(kk / N) - sc['rho'])))
        nn = np.arange(1, 2 * N + 1); nup = np.where(nn <= N + 1, mp['nuL'](nn / N), mp['nuR'](nn / N))
        nu_err = float(np.max(np.abs(nup - sc['nu'][1:])))
        ident = (sc['A0'] + sc['Ep'] + sc['SS'] + sc['X'] - sc['Var']) / sc['Var']
        r = dict(N=N, ident_relerr=ident, rho_poly_maxerr=rho_err, nu_poly_maxerr=nu_err,
                 Var_N2L2=sc['Var'] / N2 / Lg ** 2, A0_N2L2=sc['A0'] / N2 / Lg ** 2, Ep_N2L2=sc['Ep'] / N2 / Lg ** 2,
                 SS_N2L2=sc['SS'] / N2 / Lg ** 2, X_N2=sc['X'] / N2,
                 SS_minus_pred_N15=(sc['SS'] - mp['SS']) / N ** 1.5, P2_minus_pred_N15=(sc['P2'] - mp['P2']) / N ** 1.5,
                 G_minus_pred_N15=(sc['G'] - mp['G']) / N ** 1.5,
                 SS_asym_check=(sc['SS'] / N2 + 0.5 * cst['r0'] ** 2 * Lg ** 2 - cst['c_ss'] * Lg),
                 N3_cancel=KCONST * (mp['int_f1'] - mp['int_f2']) / N2,
                 X1_N2=sc['X1'] / N2, W1_N2=sc['W1'] / N2, det1_N2=sc['det1'] / N2,
                 X1_identity_err=(sc['X1'] - 2 * sc['W1'] - sc['det1']) / N2,
                 target_w=(sc['Var'] - sc['Ep'] - 0.5 * cst['r0'] ** 2 * N2 * Lg ** 2 - cst['c_w'] * N2 * Lg) / N2)
        print(f" N=2^{lg}: identity relerr {ident:.1e}; poly kernels exact to {rho_err:.1e},{nu_err:.1e};  Var {r['Var_N2L2']:.4f} = A0 {r['A0_N2L2']:.4f}"
              f" + Ep {r['Ep_N2L2']:.4f} + SS {r['SS_N2L2']:+.4f} + X {sc['X']/N2/Lg**2:+.5f}  (units N^2 L^2)")
        print(f"     SS_w - Mellin pred = {r['SS_minus_pred_N15']:+.3f} N^1.5  (P2 part {r['P2_minus_pred_N15']:+.3f}, G part {r['G_minus_pred_N15']:+.3f});"
              f"  [SS_w/N^2 + r0^2 L^2/2 - c_ss L] = {r['SS_asym_check']:+.4f} (should be O(1));  K(int f1 - int f2)/N^2 = {r['N3_cancel']:+.4f}")
        print(f"     X_w/N^2 = {r['X_N2']:+.4f};  X_w(1)/N^2 = {r['X1_N2']:+.4f} = 2W_w(1) {2*r['W1_N2']:.4f} + det_w(1) {r['det1_N2']:+.4f}"
              f"  (identity err {r['X1_identity_err']:.1e});  [VarG_w-Ep_w - r0^2N^2L^2/2 - c_w N^2 L]/N^2 = {r['target_w']:+.4f}")
        blocks, tail = smooth_blocks(sc, DB, cst); r['blocks'] = blocks; r['tail_N2'] = tail
        print(f"     dyadic blocks of X_w (first row q in {{1,2}}); predicted det per Thm S2 with c_B^w={cst['cBw']:.5f}:")
        for b in blocks:
            print(f"       Q={b['Q']:5d}: P_w/N^2={b['prime']:8.4f} (min per q {b['min_prime_q']:+.1e})  D_w/N^2={b['det']:8.4f}  Dpred/N^2={b['det_pred']:8.4f}  (P+D)/N^2={b['sum']:+8.4f}")
        print(f"       tail q>{DB}: {tail:+.4f} N^2")
        res['single'][lg] = r; sys.stdout.flush()
    # many N: smoothed vs box, regressions on the endpoint functionals
    lo, hi, cnt = many; QE = 30; mu, phi = mobius_phi(QE); divs = divisor_lists(QE); rows = []; t = time.time()
    for N in np.linspace(lo, hi, cnt).astype(int):
        N = int(N); sc = smooth_core(ar, N); core = box_core(ar, N, 1); sp = psi1_split(ar, N)
        en = endpoint_functional(ar, N, QE, mu, phi); N2 = float(N) ** 2; Lg = math.log(N)
        rows.append(dict(N=N, Xw=sc['X'] / N2, Xw1=sc['X1'] / N2, W1w=2 * sc['W1'] / N2, Xbox=core['X'] / N2,
                         end1=2 * sp['endpoint'] / N2, End30=2 * sum(mu[q] ** 2 / phi[q] ** 2 * en[q] for q in en) / N2,
                         target_w=(sc['Var'] - sc['Ep'] - 0.5 * cst['r0'] ** 2 * N2 * Lg ** 2 - cst['c_w'] * N2 * Lg) / N2))
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}; fits = {}
    print(f" many N in [{lo},{hi}], {cnt} values ({time.time()-t:.0f}s):")
    for y, x in (('Xbox', 'end1'), ('Xw', 'end1'), ('Xw', 'End30'), ('Xw1', 'end1'), ('W1w', 'end1'), ('target_w', 'end1')):
        c, s = ols(cols[x], cols[y]); se = boot(cols[x], cols[y])
        fits[f"{y}~{x}"] = dict(a=float(c[0]), b=float(c[1]), b_se_block=se, resid_sd=s, sd=float(np.std(cols[y])))
        print(f"  {y:8s} ~ a + b*{x:6s}: a={c[0]:7.3f} b={c[1]:7.3f} (+-{se:.3f})  resid sd {s:.3f}  total sd {np.std(cols[y]):.3f}  mean {np.mean(cols[y]):.3f}")
    res['many'] = dict(fits=fits, rows=[{k: float(v) for k, v in r.items()} for r in rows])
    OUT['S3'] = res

def S0():
    """Fable's T6 sample (check_B_regress.npy: N, X/N^2, e^2/N, Psi1/N^2, X1/N^2, e) with honest (block) errors."""
    print("=" * 100); print("S0  re-analysis of Fable's regression sample (check_B_regress.npy)"); print("=" * 100)
    r = np.load("/home/claude/pruebas/check_B_regress.npy"); x = r[:, 2]; res = {}
    for name, y in (('X', r[:, 1]), ('X1', r[:, 4])):
        c, s = ols(x, y); naive = s / np.std(x) / math.sqrt(len(x))
        b8, b25 = boot(x, y, blk=8), boot(x, y, blk=25)
        res[name] = dict(a=float(c[0]), b=float(c[1]), naive_se=naive, block8_se=b8, block25_se=b25)
        print(f"  {name}/N^2 = {c[0]:.3f} + {c[1]:.3f} e^2/N :  naive se {naive:.3f}, block-bootstrap se {b8:.3f} (block 8), {b25:.3f} (block 25)")
    res['lag1_autocorr_regressor'] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    print(f"  lag-1 autocorrelation of e^2/N in the sample: {res['lag1_autocorr_regressor']:.3f}")
    OUT['S0'] = res

if __name__ == "__main__":
    which = sys.argv[1:] or ['S0', 'S1', 'S2', 'S3']
    if 'S0' in which: S0()
    if which == ['S0']:
        old = json.load(open("/home/claude/pruebas/manuscrito/v5_scratch/check_B2_S2_rerun.json")); old.update(OUT)
        json.dump(old, open("/home/claude/pruebas/manuscrito/v5_scratch/check_B2_S2_rerun.json", "w"), indent=1, default=float); sys.exit(0)
    t0 = time.time(); ar = Arith(1 << 22 if 'S1' in which else 1 << 21)
    print(f"sieve done ({time.time()-t0:.1f}s); c_B = {cB:.6f}, K = {KCONST}, 2K = {2*KCONST:.4f}")
    if 'S1' in which: S1(ar)
    if 'S2' in which: S2(ar)
    if 'S3' in which: S3(ar)
    try:
        old = json.load(open("/home/claude/pruebas/manuscrito/v5_scratch/check_B2_S2_rerun.json"))
    except Exception:
        old = {}
    old.update({k: v for k, v in OUT.items()})
    json.dump(old, open("/home/claude/pruebas/manuscrito/v5_scratch/check_B2_S2_rerun.json", "w"), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")
