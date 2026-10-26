"""check_fable.py -- numerical checks for ronda3/fable_moduli_grandes.md (large moduli, smoothed weights).

Everything is computed exactly (floating point) from Lambda; the singular series uses the full twin-prime
constant C2.  Notation as in lema_B_opus.md / fable_moduli_grandes.md:

  w_m = w(m/N), lambda = Lambda*w, rho_w, nu_w, psi2^w, R_w, E_w = psi2^w - S rho_w, delta_w = R_w - S nu_w,
  X_w^{>Q} = sum_{q>Q} X_w(q) = 4 sum_{h>=1} St_Q(h) rho_w(h) E_w(h) - 2 sum_n St_Q(n) nu_w(n) delta_w(n),
  St_Q = tail of the singular series = sum_{d|k} c(d) beta_{>Q}(d),  c(d) = d mu^2(d)/phi(d)^2,
  beta_{>Q}(d) = sum_{r>Q/d,(r,d)=1} mu(r)/phi(r)^2 = eta(d) - gamma_Q(d),
  Delta_w(d) = 4 sum_{h>=1, d|h} rho_w E_w - 2 sum_{d|n} nu_w delta_w   (so X_w^{>Q} = sum_d c(d) beta_{>Q}(d) Delta_w(d)).

Tests
  T1  the tail X_w^{>Q}/N^2 for Q from N^{0.4} to 4N, its two halves (prime-pair side T1, Goldbach side T2),
      the dyadic blocks P+D = X^{>Q}-X^{>2Q}, the mean squares E_{p,w}, VarG_w, and the Cauchy-Schwarz bound
      of Theorem A with the *actual* mean squares;  GZHN (1.10) numerically.
  T2  the exact identity X_w^{>Q} = sum_d c(d) beta_{>Q}(d) Delta_w(d); sizes of Delta_w(d); the small-d part
      sum_{d<=Q/R} (Step 2 of Theorem B);  the per-d GRH-type size |Delta_w(d)|/(N^2 log^2 2d).
  T3  the deterministic matching of Theorem B, Step 5 (truncation level E in {1,2,6}, smooth cutoff Theta):
        Dpp_E(y) = sum_k rho_w(k)^2 [ S(k) sigma_E(k;y) - omega_E(k;y) ]      (prime-pair side)
        DG_E(y)  = sum_n nu_w(n)^2  [ S(n) sigma_E(n;y) - omega_E(n;y) ]      (Goldbach side)
      against the predicted Vaughan secondary term  -(1/2) Sigma_E sum_l sum_k f_l(k)/k.
  T4  the general-M Vaughan lemma (Lemma 1 of the file) for M in {1,2,3,4,6,8,9,12,30} with a smooth f.
Usage: python3 check_fable.py [logN ...]   (default 16 18 20; ~1-3 min each on 2 cores).  Output: check_fable.json
"""
import sys, json, math, time
import numpy as np, scipy.fft as sf

C2 = 0.660161815846869573927812110015
GAMMA = 0.577215664901532860606512090082
OUT = {}

# ------------------------------------------------------------------------------------------------ basics
def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]

def mobius_phi(D):
    mu = np.ones(D + 1, np.int64); phi = np.arange(D + 1, dtype=np.int64)
    for p in sieve(D):
        mu[p::p] *= -1; mu[p * p::p * p] = 0; phi[p::p] = phi[p::p] // p * (p - 1)
    return mu, phi

def flen(N):
    L = 1
    while L < 2 * N + 2: L *= 2
    return L

class Arith:
    def __init__(self, Nmax):
        self.Nmax = Nmax; pr = sieve(4 * Nmax + 4); self.pr = pr
        Lam = np.zeros(Nmax + 1)
        for p in pr[pr <= Nmax]:
            lp = math.log(p); pk = int(p)
            while pk <= Nmax: Lam[pk] = lp; pk *= int(p)
        self.Lam = Lam
        S = np.zeros(2 * Nmax + 2); S[0::2] = 2 * C2
        for p in pr[1:]:
            if p <= 2 * Nmax + 1: S[0::p] *= (p - 1) / (p - 2)
        S[0] = 0.0; self.S = S
        D = 4 * Nmax + 4; self.mu, self.phi = mobius_phi(D)
        mu, phi = self.mu, self.phi
        # c(d) = d mu^2/phi^2 ;  eta(d) = prod_{p not | d} (1-(p-1)^{-2}) = 0 (d odd), C2 prod_{p|d,p>2} (p-1)^2/(p(p-2)) (d even)
        D2 = 2 * Nmax + 2; d = np.arange(D2 + 1)
        self.c = np.where(mu[:D2 + 1] != 0, d / np.maximum(phi[:D2 + 1], 1) ** 2, 0.0); self.c[0] = 0
        eta = np.zeros(D2 + 1); eta[2::2] = C2
        for p in pr[1:]:
            if p <= D2: eta[p::p] *= (p - 1) ** 2 / (p * (p - 2))
        self.eta = eta
        # A0 = prod_p (1+(p-1)^{-2}) and theta(l) = prod_{p|l} (p^2-p+1)(p-1)/(p(p^2-2p+2))
        A0 = 1.0
        for p in pr: A0 *= 1 + 1 / (p - 1) ** 2
        P = pr[-1]; A0 *= math.exp(1 / (P * math.log(P)))      # tail of the product, ~ sum_{p>P} p^-2
        self.A0 = A0
        th = np.ones(D2 + 1)
        for p in pr:
            if p <= D2: th[p::p] *= (p * p - p + 1) * (p - 1) / (p * (p * p - 2 * p + 2))
        self.theta = th

def weight(name, N):
    m = np.arange(N + 1); x = m / N
    if name == 'box': w = np.ones(N + 1)
    elif name == '1-x': w = 1 - x
    elif name == '(1-x)^2': w = (1 - x) ** 2
    else: raise ValueError(name)
    w[0] = 0.0; return w

def core(ar, N, wname):
    w = weight(wname, N); lam = ar.Lam[:N + 1] * w; L = flen(N)
    FL = sf.rfft(lam, L, workers=2); Fw = sf.rfft(w, L, workers=2)
    R = sf.irfft(FL * FL, L, workers=2)[:2 * N + 1]
    psi2 = sf.irfft(FL * np.conj(FL), L, workers=2)[:N + 1].copy()
    nu = sf.irfft(Fw * Fw, L, workers=2)[:2 * N + 1]
    rho = sf.irfft(Fw * np.conj(Fw), L, workers=2)[:N + 1].copy()
    del FL, Fw
    S = ar.S[:2 * N + 1]
    E = psi2 - S[:N + 1] * rho; E[0] = 0.0                       # E_w(h), h = 0..N  (E[0] set to 0)
    dG = R - S * nu                                             # delta_w(n), n = 0..2N
    Ep = 2 * (E ** 2).sum(); Var = (dG ** 2).sum()
    X = 4 * (S[:N + 1] * rho * E).sum() - 2 * (S * nu * dG).sum()
    return dict(N=N, w=wname, rho=rho, nu=nu, E=E, dG=dG, Ep=Ep, Var=Var, X=X, psi20=psi2[0], rho0=rho[0],
                wE=rho * E, wd=nu * dG)

# ---------------------------------------------------------------------------------- divisor machinery
def mult_sums(x, Dmax):
    """F[d] = sum_{m>=1} x[m d]  (over md < len(x)), d = 1..Dmax.  x[0] is ignored."""
    K = len(x) - 1; F = np.zeros(Dmax + 1); D0 = int(math.isqrt(K))
    for d in range(1, min(D0, Dmax) + 1): F[d] = x[d::d].sum()
    if Dmax > D0:
        for m in range(1, K // (D0 + 1) + 1):
            hi = min(K // m, Dmax)
            if hi <= D0: break
            idx = np.arange(D0 + 1, hi + 1); F[idx] += x[m * idx]
    return F

def divisor_sieve(coef, K):
    """A[k] = sum_{d|k} coef[d], k = 0..K (A[0]=0); coef indexed 0..Dmax, Dmax <= K."""
    Dmax = len(coef) - 1; A = np.zeros(K + 1); D0 = int(math.isqrt(K))
    for d in range(1, min(D0, Dmax) + 1):
        if coef[d] != 0: A[d::d] += coef[d]
    if Dmax > D0:
        for m in range(1, K // (D0 + 1) + 1):
            hi = min(K // m, Dmax)
            if hi <= D0: break
            idx = np.arange(D0 + 1, hi + 1); A[m * idx] += coef[idx]
    return A

def gamma_Q(ar, Q, Dmax):
    """gamma_Q(d) = sum_{r<=Q/d, (r,d)=1} mu(r)/phi(r)^2, d = 1..Dmax."""
    mu, phi = ar.mu, ar.phi; R0 = 64; QQ = max(Q, R0); mphi2 = np.zeros(QQ + 2); r = np.arange(1, QQ + 1)
    mphi2[1:QQ + 1] = mu[1:QQ + 1] / phi[1:QQ + 1].astype(float) ** 2
    g = np.zeros(Dmax + 1); dA = min(Dmax, Q // R0)
    for d in range(1, dA + 1):
        if mu[d] == 0: continue
        Z = Q // d; rr = r[:Z]; g[d] = mphi2[rr][np.gcd(rr, d) == 1].sum()
    hi = min(Q, Dmax)
    if hi > dA:
        dd = np.arange(dA + 1, hi + 1); gg = np.zeros(len(dd))
        for rr in range(1, min(R0, Q) + 1):
            if mu[rr] == 0: continue
            msk = (dd <= Q // rr) & (np.gcd(dd, rr) == 1); gg[msk] += mphi2[rr]
        g[dd] = gg
    g[ar.mu[:Dmax + 1] == 0] = 0.0
    return g

def tail_series(ar, Q, K):
    """St_Q(k), k = 0..K, and the coefficient array cb[d] = c(d) beta_{>Q}(d), d <= K."""
    g = gamma_Q(ar, Q, K); cb = ar.c[:K + 1] * (ar.eta[:K + 1] - g)
    return divisor_sieve(cb, K), cb

# ------------------------------------------------------------------------------------------------ T1, T2
def T12(ar, cr):
    N = cr['N']; N2 = float(N) ** 2; L = math.log(N); K = 2 * N
    S = ar.S[:K + 1]; rho, nu, E, dG = cr['rho'], cr['nu'], cr['E'], cr['dG']
    wE = np.zeros(K + 1); wE[:N + 1] = cr['wE']; wd = cr['wd']
    print(f"--- w={cr['w']:8s} N=2^{round(math.log2(N))}  X_w/N^2={cr['X']/N2:+.4f}  E_p,w/N^2={cr['Ep']/N2:.4f}  VarG_w/N^2={cr['Var']/N2:.4f}"
          f"  (E_p,w+VarG_w)/N^{{5/2}}={(cr['Ep']+cr['Var'])/N**2.5:.2e}")
    # sanity: sum_{d|k} c(d) eta(d) = S(k)
    St0 = divisor_sieve(ar.c[:K + 1] * ar.eta[:K + 1], K)
    print(f"    check  max_k |sum_{{d|k}} c(d)eta(d) - S(k)| = {np.abs(St0 - S).max():.2e}")
    # Delta_w(d) for all d <= 2N
    Delta = 4 * mult_sums(wE, K) - 2 * mult_sums(wd, K)
    cd = ar.c[:K + 1]; d = np.arange(K + 1); ld = np.log(2 * np.maximum(d, 1))
    r2 = dict(max_Delta_N2=float(np.abs(Delta).max() / N2), max_Delta_over_N2log2=float((np.abs(Delta) / (N2 * ld ** 2))[1:].max()),
              sum_c_absDelta_N2=float((cd * np.abs(Delta)).sum() / N2))
    print(f"    Delta_w(d):  max|Delta|/N^2={r2['max_Delta_N2']:.3f}   max|Delta|/(N^2 log^2 2d)={r2['max_Delta_over_N2log2']:.3f}"
          f"   sum_d c(d)|Delta(d)|/N^2={r2['sum_c_absDelta_N2']:.3f}  (log^3 N = {L**3:.0f})")
    # tails
    j0 = int(math.floor(0.4 * math.log2(N))); j1 = int(math.log2(N)) + 2
    rows = []; prevX = None
    print("    Q         Q/sqrtN   X^{>Q}/N^2   T1/N^2    T2/N^2   (X^{>Q}-X^{>2Q})/N^2  CSbound/N^2  CS(5/2)/N^2  ident.err  small-d(R=8)/N^2  GZHN(1.10)")
    for j in range(j0, j1 + 1):
        Q = 2 ** j; St, cb = tail_series(ar, Q, K)
        T1 = 4 * (St[:N + 1] * wE[:N + 1]).sum(); T2 = 2 * (St * wd).sum(); Xt = T1 - T2
        ident = (cb * Delta).sum()
        n1 = math.sqrt((St[:N + 1] ** 2 * rho ** 2).sum()); n2 = math.sqrt((St ** 2 * nu ** 2).sum())
        cs = 2 * n1 * math.sqrt(cr['Ep']) + 2 * n2 * math.sqrt(cr['Var'])                      # Theorem A with actual mean squares
        cs52 = 2 * (n1 + n2) * N ** 1.25 * L ** 2.5                                              # with the GRH-size mean square N^{5/2} L^5
        gz = (St[1:N + 1] ** 2).sum() * Q * Q / (N * L)
        smalld = (cb * Delta)[:Q // 8 + 1].sum()
        blk = (prevX - Xt) / N2 if prevX is not None else float('nan')
        print(f"  {Q:8d}  {Q/math.sqrt(N):8.3f}   {Xt/N2:+.5f}  {T1/N2:+.5f}  {T2/N2:+.5f}      {blk:+.5f}          {cs/N2:9.4f}   {cs52/N2:9.2e}   {abs(ident-Xt)/max(abs(Xt),1e-9*N2):.1e}    {smalld/N2:+.5f}      {gz:.3f}")
        rows.append(dict(Q=Q, Xtail_N2=Xt / N2, T1_N2=T1 / N2, T2_N2=T2 / N2, block_prev_N2=blk, CS_N2=cs / N2, CS52_N2=cs52 / N2,
                         ident_relerr=abs(ident - Xt) / max(abs(Xt), 1e-9 * N2), smalld_N2=smalld / N2, gzhn=gz))
        prevX = Xt
    return dict(rows=rows, Delta=r2, Ep_N2=cr['Ep'] / N2, Var_N2=cr['Var'] / N2, X_N2=cr['X'] / N2)

# ------------------------------------------------------------------------------------------------ T3
def gamma_star(ar, E):
    """gamma*(e) = (g * mu)(e), g(d) = d^2 mu^2(d)/phi(d)^2, e <= E (multiplicative, supported on cube-free)."""
    gs = np.zeros(E + 1); gs[1] = 1.0
    for e in range(2, E + 1):
        v = 1.0; x = e; p = 2
        while p * p <= x:
            if x % p == 0:
                k = 0
                while x % p == 0: x //= p; k += 1
                if k == 1: v *= p * p / (p - 1) ** 2 - 1
                elif k == 2: v *= -p * p / (p - 1) ** 2
                else: v = 0.0
            p += 1
        if x > 1: v *= x * x / (x - 1) ** 2 - 1
        gs[e] = v
    return gs

def theta_cut(u, delta=0.25):
    """smooth step: 0 for u<=1, 1 for u>=1+delta, C^1 in between (cubic)."""
    t = np.clip((u - 1) / delta, 0, 1); return t * t * (3 - 2 * t)

def T3(ar, cr, Es=(1, 2, 6), ys=None):
    N = cr['N']; N2 = float(N) ** 2; K = 2 * N; S = ar.S[:K + 1]; rho, nu = cr['rho'], cr['nu']
    rho2 = np.zeros(K + 1); rho2[:N + 1] = rho ** 2; nu2 = nu ** 2
    if ys is None: ys = [int(N ** a) for a in ((0.4, 0.5, 0.6, 0.75) if N <= (1 << 16) else (0.5, 0.6, 0.75))]
    print(f"--- T3 deterministic matching, w={cr['w']}  N=2^{round(math.log2(N))}:  rows = (E, y):  [Dpp, DG, pred](/N^2) and residuals")
    res = []
    d = np.arange(K + 1, dtype=float); phi = ar.phi; k = d.copy(); k[0] = 1
    Emax = max(Es); gs = gamma_star(ar, Emax)
    gE = {E: divisor_sieve(gs[:E + 1], K) for E in Es}          # gE[E][d] = sum_{e|d,e<=E} gamma*(e)
    for y in ys:
        # omega_E(k;y) = sum_l (l/k) Theta(k/(y l)) Gamma_E(l),  Gamma_E(l) = sum_{e<=E} gamma*(e)/phi(e l)   (all E at once)
        Lmax = int(K / y) + 1; om = {E: np.zeros(K + 1) for E in Es}; predpp = 0.0; predG = 0.0
        for l in range(1, Lmax + 1):
            k0 = int(y * l)
            if k0 >= K: break
            kk = k[k0:]; th = theta_cut(kk / (y * l)); fl = l * th / kk
            cum = 0.0
            for E in Es:
                cum_e = sum(gs[e] / phi[e * l] for e in range(1, E + 1) if gs[e] != 0)
                om[E][k0:] += fl * cum_e
            predpp += (rho2[k0:] * fl / kk).sum(); predG += (nu2[k0:] * fl / kk).sum()
        for E in Es:
            SigE = gs[:E + 1].sum()
            coef = np.zeros(K + 1); coef[1:] = theta_cut(d[1:] / y) * gE[E][1:] / d[1:]     # sigma_E(k;y) = sum_{d|k} coef[d]
            sig = divisor_sieve(coef, K)
            Dpp = (rho2 * (S * sig - om[E])).sum(); DG = (nu2 * (S * sig - om[E])).sum()
            ppred = -0.5 * SigE * predpp; gpred = -0.5 * SigE * predG
            print(f"   E={E} Sigma_E={SigE:+.3f}  y={y:7d} (N^{math.log(y)/math.log(N):.2f}):  Dpp={Dpp/N2:+.4f} pred={ppred/N2:+.4f} resid={(Dpp-ppred)/N2:+.4f} |"
                  f"  DG={DG/N2:+.4f} pred={gpred/N2:+.4f} resid={(DG-gpred)/N2:+.4f} |  N/y^2={N/y**2:.3f}")
            res.append(dict(E=E, y=y, Dpp_N2=Dpp / N2, Dpp_pred_N2=ppred / N2, DG_N2=DG / N2, DG_pred_N2=gpred / N2))
    return res

# ------------------------------------------------------------------------------------------------ T4
def T4(ar, N, Ms=(1, 2, 3, 4, 6, 8, 9, 12, 30)):
    """Lemma 1 (general M): sum_{M|n<=X} S(n) f(n) = (1/phi(M)) int f + (1/2) int f' log x - f(0)/2 (1 + c#_M - log M) + O(M^{1/2+eps} X^{1/2} Var f').
       f(x) = x (X-x)^2 / X^2  on [0,X]  (f(0)=0, f(X)=f'(X)=0),  X = 2N."""
    X = 2 * N; n = np.arange(X + 1, dtype=float); f = n * (X - n) ** 2 / X ** 2; S = ar.S[:X + 1]
    intf = X ** 2 / 12.0                                         # int_0^X x (X-x)^2/X^2 dx = X^2/12
    # int f' log x dx = -int f/x dx = -int (X-x)^2/X^2 dx = -X/3
    sec = 0.5 * (-X / 3.0)
    print(f"--- T4 Lemma 1 (general M, smooth f = x(X-x)^2/X^2, X=2N={X}):  err/(sqrt(M X))")
    out = []
    for M in Ms:
        lhs = (S[M::M] * f[M::M]).sum(); main = intf / ar.phi[M] + sec
        err = lhs - main; out.append(dict(M=M, lhs=lhs, main=main, err=err, err_norm=err / math.sqrt(M * X)))
        print(f"    M={M:3d}  lhs={lhs:.6e}  main={main:.6e}  err={err:+.4e}  err/sqrt(MX)={err/math.sqrt(M*X):+.4f}")
    return out

# ------------------------------------------------------------------------------------------------ main
if __name__ == '__main__':
    logs = [int(a) for a in sys.argv[1:]] or [16, 18, 20]
    Nmax = 1 << max(logs); t0 = time.time(); ar = Arith(Nmax)
    print(f"sieve+tables {time.time()-t0:.1f}s;  A0 = prod_p(1+(p-1)^-2) = {ar.A0:.8f}")
    for lg in logs:
        N = 1 << lg
        for wn in ('1-x', '(1-x)^2', 'box'):
            t1 = time.time(); cr = core(ar, N, wn)
            r = T12(ar, cr); OUT[f'T12_{wn}_{lg}'] = r
            if wn != 'box': OUT[f'T3_{wn}_{lg}'] = T3(ar, cr)
            print(f"    ({time.time()-t1:.0f}s)")
        OUT[f'T4_{lg}'] = T4(ar, N)
    json.dump(OUT, open('check_fable.json', 'w'), indent=1, default=float)
    print(f"total {time.time()-t0:.0f}s")
