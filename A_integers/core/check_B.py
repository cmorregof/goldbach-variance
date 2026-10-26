"""check_B.py -- numerical tests for Lemma B (the cross term X).  Fable.

Notation as in BRIEF.md and lema_B_fable.md.  Exact decomposition  X = sum_q X(q)  with

   X(q) = 2 mu(q)^2/phi(q)^2 * sum_{d|q} d mu(q/d) [ F_E(d) - F_delta(d) ],
   F_E(d) = sum_{d|h, h!=0} (N-|h|) E(h),   F_delta(d) = sum_{d|n} nu(n) delta(n),

E(h) = psi2(h) - S(h)(N-|h|),  delta(n) = R(n) - S(n) nu(n).

Window identity (Prop. 2 of the write-up), for every d >= 1:

   F_psi(d) - F_R(d) = Psi_d - N psi2(0),
   Psi_d = sum_{k=1}^{N-1} sum_{r mod d} (psi(W_k;d,r) - psi(W_{-k};d,-r))^2
           + (1/2) sum_r (psi(N;d,r) - psi(N;d,-r))^2  >= 0,
   W_k = (k,N], W_{-k} = [1,N-k].

Deterministic part (Prop. 3), c_B = 1/2 + gamma + log(pi/2), w'_p = p(p-2)log p/(p-1)^2 :

   X(q) = 2mu^2/phi^2 * W(q) + 2mu^2/phi(q) * N^2 [c_B - sum_{p|q,p>2} w'_p] + O(N^{3/2} q^{-1/2+eps}),
   W(q) = sum_{d|q} d mu(q/d) Psi_d >= 0,   and for q=1:  X(1) = 2 Psi_1 + 2 c_B N^2 + o(N^2),
   Psi_1 = sum_{k=1}^{N-1} (psi(N) - psi(k) - psi(N-k))^2 = N (psi(N)-N)^2 + O(N^2) under RH.

Tests (all quantities exact up to floating point, N = 2^LOGN):
  T1  convergence of sum_{q<=Q} X(q) to X, and the tail.
  T2  window identity for d <= 12.
  T3  deterministic part: numerical  Ztilde(q)  versus the closed formula, q <= 30.
  T4  X(1) = 2 Psi_1 + 2 c_B N^2 ;  Psi_1 versus N (psi(N)-N)^2.
  T5  W(q) >= mu^2/phi(q) * Psi_1  (the principal-character part of W(q)), q <= 30.
  T6  (option --regress) X/N^2 against (psi(N)-N)^2/N over many N: slope should be about 2K,
      K = prod_p (1 + 1/(p-1)^3) = 2.30.
  T7  the per-d model sums of Lemma 3.2/3.1:  B_1^{(d)}(x), B_2^{(d)}(x) (Vaughan's lemma) and the
      reflection identity  sum_n nu(n)^2 g(n) = B_2^g(2N+1) - 4N B_1^g(N+1), for several squarefree d.
Output: check_B.json (and check_B_regress.npy with --regress).

Companion script: check_B_lambda.py tests the completed-square identity (Prop. 1) term by term.
"""
import sys, json, numpy as np, scipy.fft as sf
from math import log, pi, sqrt

C2 = 0.6601618158468696          # twin prime constant, full precision (the product over p <= 2N is NOT enough)
GAMMA = 0.5772156649015329
cB = 0.5 + GAMMA + log(pi/2)

def sieve(lim):
    isp = np.ones(lim, bool); isp[:2] = False
    for p in range(2, int(lim**0.5) + 1):
        if isp[p]: isp[p*p::p] = False
    return np.nonzero(isp)[0]

def mobius_phi(D):
    mu = np.ones(D + 1, int); phi = np.arange(D + 1)
    isp = np.ones(D + 1, bool); isp[:2] = False
    for p in range(2, D + 1):
        if not isp[p]: continue
        isp[p*p::p] = False
        mu[p::p] *= -1; mu[p*p::p*p] = 0
        phi[p::p] = phi[p::p] // p * (p - 1)
    return mu, phi

def prime_factors(q):
    f = []; p = 2
    while p*p <= q:
        if q % p == 0:
            f.append(p)
            while q % p == 0: q //= p
        p += 1
    if q > 1: f.append(q)
    return f

def basic(N):
    """Lambda, R, psi2, S, nu for the box [1,N]^2."""
    lim = 2*N + 1
    pr = sieve(lim + 1)
    Lam = np.zeros(N + 1)
    for p in pr[pr <= N]:
        lp = log(p); pk = int(p)
        while pk <= N: Lam[pk] = lp; pk *= int(p)
    L = 4*N
    F = sf.rfft(Lam, L, workers=2)
    R = sf.irfft(F*F, L, workers=2)[:lim]
    psi = sf.irfft(F*np.conj(F), L, workers=2)[:N+1].copy(); del F
    S = np.zeros(lim); S[0::2] = 2*C2
    for p in pr[1:]:
        S[0::p] *= (p-1)/(p-2)
    S[0] = 0.0
    n = np.arange(lim); nu = np.where(n <= N+1, np.maximum(n-1, 0), 2*N - n + 1).astype(float)
    return Lam, R, psi, S, nu

def run(LOGN, D=2000, DPSI=30):
    N = 1 << LOGN
    Lam, R, psi, S, nu = basic(N)
    lim = 2*N + 1
    h = np.arange(1, N+1); M = S[1:N+1]*(N - h); E = psi[1:] - M
    dG = R - S*nu
    X1 = 4*(M*E).sum(); X2 = -2*(S*nu*dG).sum(); X = X1 + X2
    psi20 = psi[0]; psiN = Lam.sum(); eN = psiN - N
    norm = N**2*log(N)**2
    # ---- per-d sums
    wE = (N - h)*E; wpsi = (N - h)*psi[1:]; wM = (N - h)*M
    wdel = nu*dG; wR = nu*R; wmod = nu*nu*S
    mu, phi = mobius_phi(D)
    FE = np.zeros(D+1); Fd = np.zeros(D+1); Fpsi = np.zeros(D+1); FR = np.zeros(D+1); FM = np.zeros(D+1); Fmod = np.zeros(D+1)
    for d in range(1, D + 1):
        FE[d] = 2*wE[d-1::d].sum(); Fpsi[d] = 2*wpsi[d-1::d].sum(); FM[d] = 2*wM[d-1::d].sum()
        Fd[d] = wdel[0::d].sum(); FR[d] = wR[0::d].sum(); Fmod[d] = wmod[0::d].sum()
    Xq = np.zeros(D+1); Zt_num = np.zeros(D+1); Zt_pred = np.zeros(D+1)
    for q in range(1, D + 1):
        if mu[q] == 0: continue
        divs = [d for d in range(1, q + 1) if q % d == 0]
        c = 2*mu[q]**2/phi[q]**2
        Xq[q] = c*sum(d*mu[q//d]*(FE[d] - Fd[d]) for d in divs)
        Zt_num[q] = c*sum(d*mu[q//d]*(Fmod[d] - FM[d]) for d in divs) - 2*N*psi20/phi[q]
        Zt_pred[q] = (2/phi[q])*N**2*(cB - sum(p*(p-2)*log(p)/(p-1)**2 for p in prime_factors(q) if p > 2))
    cum = np.cumsum(Xq)
    # ---- T2/T5: Psi_d for d <= DPSI via residue classes
    k = np.arange(1, N)
    Psi = {}
    for d in range(1, DPSI + 1):
        cls = np.zeros((d, N + 1))
        for r in range(d):
            v = np.zeros(N + 1); v[r::d] = Lam[r::d]; cls[r] = np.cumsum(v)
        tot = 0.0
        for r in range(d):
            A = cls[r, N] - cls[r, k]; B = cls[(-r) % d, N - k]
            tot += ((A - B)**2).sum()
        for r in range(d):
            tot += 0.5*(cls[r, N] - cls[(-r) % d, N])**2
        Psi[d] = tot
    win_err = {d: abs((Fpsi[d] - FR[d]) - (Psi[d] - N*psi20))/abs(Fpsi[d] - FR[d]) for d in range(1, 13)}
    Wq = {}
    for q in range(1, DPSI + 1):
        if mu[q] == 0: continue
        Wq[q] = sum(d*mu[q//d]*Psi[d] for d in range(1, q+1) if q % d == 0)
    # ---- T4: Psi_1 decomposition, e(x) = psi(x) - x
    cum_psi = np.cumsum(Lam); e = cum_psi - np.arange(N + 1)
    Psi1 = Psi[1]
    parts = [(N-1)*eN**2, -2*eN*(e[k] + e[N-k]).sum(), ((e[k] + e[N-k])**2).sum()]
    out = dict(N=N, logN=log(N), X_over_N2log2N=X/norm, X_over_N2=X/N**2,
               psiN_minus_N=float(eN), eN2_over_N=float(eN**2/N), psi20_over_N=float(psi20/N), logN_minus_1=log(N)-1,
               T1=dict(Xq_over_N2={str(q): float(Xq[q]/N**2) for q in range(1, 31) if mu[q] != 0},
                       cum_over_N2={str(Q): float(cum[Q]/N**2) for Q in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000] if Q <= D},
                       tail_over_N2=float((X - cum[D])/N**2), D=D),
               T2=dict(window_identity_relerr={str(d): float(v) for d, v in win_err.items()}),
               T3=dict(Ztilde_num_over_N2={str(q): float(Zt_num[q]/N**2) for q in range(1, 31) if mu[q] != 0},
                       Ztilde_pred_over_N2={str(q): float(Zt_pred[q]/N**2) for q in range(1, 31) if mu[q] != 0},
                       max_abs_diff_over_N2_q_le_30=float(max(abs(Zt_num[q]-Zt_pred[q]) for q in range(1, 31) if mu[q] != 0)/N**2),
                       sum_q_le_30_num=float(sum(Zt_num[q] for q in range(1, 31))/N**2), sum_q_le_30_pred=float(sum(Zt_pred[q] for q in range(1, 31))/N**2)),
               T4=dict(X1_over_N2=float(Xq[1]/N**2), twoPsi1_plus_2cB=float((2*Psi1 + 2*cB*N**2)/N**2), twocB=2*cB,
                       Psi1_over_N2=float(Psi1/N**2), Psi1_parts_over_N2=[float(p/N**2) for p in parts],
                       NeN2_over_N2=float(N*eN**2/N**2)),
               T5=dict(W_over_N2={str(q): float(Wq[q]/N**2) for q in Wq},
                       principal_part_over_N2={str(q): float(Psi1/phi[q]/N**2) for q in Wq},
                       ratio={str(q): float(Wq[q]/(Psi1/phi[q])) for q in Wq}))
    return out

def model_sums(LOGN):
    """T7: per-d singular-series sums versus the closed formulas of Lemma 3.2 and the reflection identity."""
    import math
    N = 1 << LOGN
    Lam, R, psi, S, nu = basic(N)
    lim = 2*N + 1; n = np.arange(lim)
    out = {}
    for d in [1, 2, 3, 5, 6, 7, 10, 15, 30, 105]:
        d0 = d if d % 2 else d // 2
        f = prime_factors(d0); ph0 = d0
        for p in f: ph0 = ph0 // p * (p - 1)
        c = GAMMA - 1 + log(2*pi) + sum(log(p)/(p-1) for p in f)
        def B1(x): m = n[(n <= x) & (n % d == 0)]; return math.fsum((x - m)*S[m])
        def B2(x): m = n[(n <= x) & (n % d == 0)]; return math.fsum((x - m)**2*S[m])
        x = N
        pred1 = x*x/(2*ph0) - 0.5*x*(log(x/d0) + c)
        pred2 = x**3/(3*ph0) - 0.5*x*x*log(x/d0) - 0.5*(c - 0.5)*x*x
        Fmod = math.fsum(S[n % d == 0]*nu[n % d == 0]**2); ident = B2(2*N + 1) - 4*N*B1(N + 1); FM = 2*B2(N)
        pred_diff = N*N*log(N/d0) + N*N*(GAMMA - 0.5 + log(pi/2) + sum(log(p)/(p-1) for p in f))
        out[str(d)] = dict(B1_err_over_sqrt_dx=float((B1(x) - pred1)/sqrt(d*x)), B2_err_over_sqrt_d_x15=float((B2(x) - pred2)/sqrt(d)/x**1.5),
                           reflection_relerr=float((Fmod - ident)/Fmod), Fmod_minus_FM_over_N2_minus_pred=float((Fmod - FM - pred_diff)/N**2))
        print(f"  T7 d={d:3d}: B1 err/(dx)^.5={out[str(d)]['B1_err_over_sqrt_dx']:7.3f}  B2 err/(d^.5 x^1.5)={out[str(d)]['B2_err_over_sqrt_d_x15']:7.3f}  reflection relerr={out[str(d)]['reflection_relerr']:.1e}  [(Fmod-FM)/N^2 - formula]={out[str(d)]['Fmod_minus_FM_over_N2_minus_pred']:+.4f}")
    return out

def regress(Nmin, Nmax, step, fname):
    pr = sieve(2*Nmax + 2)
    LamAll = np.zeros(Nmax + 1)
    for p in pr[pr <= Nmax]:
        lp = log(p); pk = int(p)
        while pk <= Nmax: LamAll[pk] = lp; pk *= int(p)
    Sall = np.zeros(2*Nmax + 2); Sall[0::2] = 2*C2
    for p in pr[1:]: Sall[0::p] *= (p-1)/(p-2)
    Sall[0] = 0
    cumpsi = np.cumsum(LamAll)
    rows = []
    for N in range(Nmin, Nmax + 1, step):
        Lam = LamAll[:N+1]; lim = 2*N + 1; L = 1
        while L < 4*N: L *= 2
        F = sf.rfft(Lam, L); R = sf.irfft(F*F, L)[:lim]; psi = sf.irfft(F*np.conj(F), L)[:N+1]
        S = Sall[:lim]; n = np.arange(lim); nu = np.where(n <= N+1, np.maximum(n-1, 0), 2*N - n + 1).astype(float)
        h = np.arange(1, N+1); M = S[1:N+1]*(N - h); E = psi[1:] - M; dG = R - S*nu
        X = 4*(M*E).sum() - 2*(S*nu*dG).sum()
        X1 = 4*((N - h)*E).sum() - 2*(nu*dG).sum()
        eN = cumpsi[N] - N
        k = np.arange(1, N); Psi1 = ((cumpsi[N] - cumpsi[k] - cumpsi[N-k])**2).sum()
        rows.append((N, X/N**2, eN**2/N, Psi1/N**2, X1/N**2, eN))
    rows = np.array(rows); np.save(fname, rows)
    x = rows[:, 2]; A = np.vstack([np.ones_like(x), x]).T
    res = {}
    for name, col in [("X", 1), ("X1", 4)]:
        y = rows[:, col]; coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res[name] = dict(a=float(coef[0]), b=float(coef[1]), resid_std=float(np.std(y - A@coef)), std=float(np.std(y)), corr=float(np.corrcoef(x, y)[0, 1]))
        print(f"  fit {name}/N^2 = a + b*(psi(N)-N)^2/N : a={coef[0]:.4f} b={coef[1]:.4f} resid_std={res[name]['resid_std']:.4f} (std {res[name]['std']:.4f}, corr {res[name]['corr']:.3f})")
    K = 1.0
    for p in pr[pr <= 10**5]: K *= 1 + 1/(p-1)**3
    print(f"  prediction for the slope of X: 2K = {2*K:.3f};  for X(1): 2.  Range of (psi(N)-N)^2/N: {x.min():.3f}..{x.max():.3f}; mean X/N^2 = {rows[:,1].mean():.3f}")
    res['twoK'] = 2*K; res['count'] = len(rows); res['Nmin'] = Nmin; res['Nmax'] = Nmax
    return res

if __name__ == "__main__":
    args = sys.argv[1:]
    do_reg = '--regress' in args
    args = [int(a) for a in args if a != '--regress'] or [16, 18, 20]
    res = {}
    for LOGN in args:
        r = run(LOGN, DPSI=30 if LOGN <= 20 else 12); res[LOGN] = r
        N = r['N']
        print(f"N=2^{LOGN}  X/(N^2 log^2 N)={r['X_over_N2log2N']:.4f}  X/N^2={r['X_over_N2']:.3f}  psi(N)-N={r['psiN_minus_N']:.1f}  (psi(N)-N)^2/N={r['eN2_over_N']:.3f}")
        print("  T1 X(q)/N^2, q<=12:", {q: round(v, 3) for q, v in r['T1']['Xq_over_N2'].items() if int(q) <= 12})
        print("  T1 cumulative:", {Q: round(v, 3) for Q, v in r['T1']['cum_over_N2'].items()}, " tail(q>2000)/N^2 =", round(r['T1']['tail_over_N2'], 4))
        print("  T2 window identity max rel.err (d<=12): %.1e" % max(r['T2']['window_identity_relerr'].values()))
        print("  T3 deterministic part Ztilde(q)/N^2 num vs pred, q=1,2,3,5,6,7:", [(q, round(r['T3']['Ztilde_num_over_N2'][q], 3), round(r['T3']['Ztilde_pred_over_N2'][q], 3)) for q in ['1', '2', '3', '5', '6', '7']],
              " max|diff|/N^2 (q<=30) = %.4f" % r['T3']['max_abs_diff_over_N2_q_le_30'])
        print("  T4 X(1)/N^2 = %.4f  vs 2Psi_1/N^2 + 2c_B = %.4f  (2c_B = %.4f);  Psi_1/N^2 = %.4f, N(psi(N)-N)^2/N^2 = %.4f, parts = %s" %
              (r['T4']['X1_over_N2'], r['T4']['twoPsi1_plus_2cB'], r['T4']['twocB'], r['T4']['Psi1_over_N2'], r['T4']['NeN2_over_N2'], [round(p, 4) for p in r['T4']['Psi1_parts_over_N2']]))
        print("  T5 W(q)/N^2 and ratio W(q)/(Psi_1/phi(q)) for q<=15:", [(q, round(r['T5']['W_over_N2'][q], 3), round(r['T5']['ratio'][q], 2)) for q in r['T5']['W_over_N2'] if int(q) <= 15])
        sys.stdout.flush()
    print(f"T7 model sums (Lemma 3.2 and reflection identity) at N=2^{args[0]}:")
    res['T7'] = model_sums(args[0])
    if do_reg:
        print("T6 regression over N in [40000, 80000] step 97:")
        res['regress'] = regress(40000, 80000, 97, "/home/claude/pruebas/check_B_regress.npy")
    json.dump(res, open("/home/claude/pruebas/check_B.json", "w"), indent=1)
