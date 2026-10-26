"""check_A.py -- numerical check of Lemma A (see lema_A_opus.md).

Lemma A (proved there, unconditionally):
    SS := 2 sum_{k=1}^{N-1} (N-k)^2 S(k)^2  -  sum_{n=2}^{2N} S(n)^2 nu(n)^2
        = 2N^2 (a2 e2 + a1 e1 + a0) - a2 I1(N) - a1 I0(N) + O(N^{15/8})          (exact-residue form)
        = N^2 ( -1/2 log^2 N + c1 log N + c0 ) + O(N^{15/8})                         (expanded form)
where D(s) = sum_k S(k)^2 k^{-s} = a2/s^2 + a1/s + a0 + O(s) near s = 0, a2 = -1/2,
    e1 = L - 3/2, e2 = L^2/2 - 3L/2 + 7/4   (L = log N),
    I0 = int_1^{2N+1} g(t)^2 dt/t,  I1 = int_1^{2N+1} g(t)^2 log t dt/t,  g = tent with g(n) = nu(n).
Separately (each side):
    P2 := 2 sum (N-k)^2 S(k)^2       = (2K/3) N^3 + 2N^2 (a2 e2 + a1 e1 + a0) + O(N^{15/8})
    G  := sum S(n)^2 nu(n)^2         = (2K/3) N^3 + a2 I1 + a1 I0               + O(N^{15/8})
with K = sum_q mu(q)^2/phi(q)^3 = prod_p (1 + 1/(p-1)^3).
Also checks GZHN Theorem 4 (their (6.2)) and Theorem 2 at y = 1.

Usage: python3 check_A.py [logN ...]     (default 12 14 16 18 20 22 24)
Needs numpy; mpmath optional (only to cross-check the hard-coded constants).
"""
import sys, math, json
import numpy as np

# ---- hard-coded classical constants (30-digit values from mpmath) ----------------------------
C2     = 0.660161815846869573927812110015      # twin prime constant prod_{p>2} (1 - 1/(p-1)^2)
GAMMA  = 0.577215664901532860606512090082      # Euler gamma
GAMMA1 = -0.0728158454836767248605863758749    # Stieltjes gamma_1
Z1     = -0.918938533204672741780329736406     # zeta'(0)  = -log(2 pi)/2
Z2     = -2.00635645590858485121010002673 / 2  # zeta''(0)/2
Z0     = -0.5                                  # zeta(0)
try:
    import mpmath as mp
    mp.mp.dps = 30
    assert abs(float(mp.twinprime) - C2) < 1e-15 and abs(float(mp.stieltjes(1)) - GAMMA1) < 1e-15
    assert abs(float(mp.zeta(0, derivative=2)) / 2 - Z2) < 1e-15
except ImportError:
    pass


def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0]


def constants(P=2 * 10 ** 8):
    """Laurent coefficients a2, a1, a0 of D(s) at s = 0, K, and derived c1, c0."""
    pr = sieve(P).astype(np.float64)
    po = pr[1:]                                   # odd primes
    lp = np.log(po)
    # S1 = sum_{p>2} log p/(p-1)^2  (tail ~ 1/P by the PNT)
    S1 = math.fsum(lp / (po - 1) ** 2) + 1.0 / P
    # second derivative of log J at 0, odd-prime part:
    #   log^2 p * [ (2p-3)(p-2)^2/(p-1)^4 - 2p/(p-1)^2 ]   ~ -7 log^2 p / p^2   (tail ~ -7 (log P + 1)/P)
    t2 = lp ** 2 * ((2 * po - 3) * (po - 2) ** 2 / (po - 1) ** 4 - 2 * po / (po - 1) ** 2)
    S2 = math.fsum(t2) - 7 * (math.log(P) + 1) / P
    L2 = math.log(2)
    # log( 2^{-s} J(s)/J(0) ) = l1 s + l2 s^2 + ...
    l1 = -L2 + 2 * L2 + S1                         # = log 2 + S1
    l2 = 0.5 * (-4 * L2 ** 2 + S2)
    b1, b2 = l1, l2 + l1 ** 2 / 2                  # 2^{-s} Jhat(s) = 1 + b1 s + b2 s^2 + ...
    # s^2 zeta(s+1)^2 = 1 + 2 gamma s + (gamma^2 - 2 gamma1) s^2 + ...
    q1, q2 = 2 * GAMMA, GAMMA ** 2 - 2 * GAMMA1
    a2 = Z0
    a1 = Z1 + Z0 * (q1 + b1)
    a0 = Z2 + Z1 * (q1 + b1) + Z0 * (q2 + q1 * b1 + b2)
    # K = prod_p (1 + 1/(p-1)^3)
    K = math.exp(math.fsum(np.log1p(1.0 / (pr - 1) ** 3)))
    B = 2 * L2 ** 2 - 6 * L2 + 3                   # int_0^2 phi(u)^2 log u du/u, phi = tent
    A = 4 * L2 - 2                                 # int_0^2 phi(u)^2 du/u
    c1 = 2 * a1 - 3 * a2 - a2 * A                  # = 2 a1 + 2 log 2 + 1/2
    c0 = 2 * (7 * a2 / 4 - 1.5 * a1 + a0) - a2 * B - a1 * A
    # GZHN constants
    Sall = S1 + L2                                 # sum over all p of log p/(p-1)^2
    Spm2 = math.fsum(lp * (po - 2) / (po * (po - 1) ** 2)) + 1.0 / P   # sum_p (p-2) log p / (p (p-1)^2)
    th4 = 0.75 - GAMMA - 0.5 * math.log(2 * math.pi) - 0.5 * Sall       # GZHN (6.2), N^2 log N coeff.
    cG = 0.75 - 0.5 * math.log(2 * math.pi) + 0.5 * Spm2               # GZHN Theorem 2 constant c
    return dict(a2=a2, a1=a1, a0=a0, K=K, c1=c1, c0=c0, S1=S1, beta1=b1,
                thm4_coef=th4, c_GZHN=cG, c_true_y1=th4 + 1.0, A=A, B=B)


def I0I1(N):
    """Exact I0 = int g^2/t, I1 = int g^2 log t /t over [1, 2N+1] (closed forms, mpmath if present)."""
    try:
        import mpmath as mp
        mp.mp.dps = 40
        lg, f = mp.log, mp.mpf
    except ImportError:
        lg, f = math.log, float
    N = f(N); X = 2 * N + 1; a = N + 1
    # first piece: (t-1)^2/t = t - 2 + 1/t on [1, a]
    I0 = (a ** 2 - 1) / 2 - 2 * (a - 1) + lg(a)
    F = lambda t: t ** 2 / 2 * lg(t) - t ** 2 / 4 - 2 * (t * lg(t) - t) + lg(t) ** 2 / 2
    I1 = F(a) - F(f(1))
    # second piece: (X-t)^2/t = X^2/t - 2X + t on [a, X]
    I0 += X ** 2 * lg(X / a) - 2 * X * (X - a) + (X ** 2 - a ** 2) / 2
    G = lambda t: X ** 2 * lg(t) ** 2 / 2 - 2 * X * (t * lg(t) - t) + t ** 2 / 2 * lg(t) - t ** 2 / 4
    I1 += G(X) - G(a)
    return float(I0), float(I1)


def singular_series_sq(M, pr):
    """S(k) for k = 0..M with the exact C2 (S(0) unused)."""
    S = np.zeros(M + 1)
    S[0::2] = 2 * C2
    for p in pr[1:]:
        if p > M:
            break
        S[0::2 * p] *= (p - 1) / (p - 2)          # odd p | k, k even
    return S


def run(logN, cst):
    N = 1 << logN; M = 2 * N + 1
    pr = sieve(M)
    S = singular_series_sq(M, pr)
    k = np.arange(1, N, dtype=np.float64)
    Sk = S[1:N]
    w = (N - k) ** 2
    P2 = 2 * np.sum(w * Sk ** 2)
    Q = np.sum(w * Sk)
    T1 = np.sum(w * (Sk - 1) ** 2)                  # GZHN Theorem 2 at y = 1 (k = N term is 0)
    n = np.arange(2, 2 * N + 1, dtype=np.float64)
    nu = np.where(n <= N + 1, n - 1, 2 * N - n + 1)
    G = np.sum((S[2:2 * N + 1] * nu) ** 2)
    SS = P2 - G
    L = math.log(N); N2 = float(N) ** 2
    a2, a1, a0, K = cst['a2'], cst['a1'], cst['a0'], cst['K']
    e1, e2 = L - 1.5, L * L / 2 - 1.5 * L + 1.75
    I0, I1 = I0I1(N)
    res0P = 2 * N2 * (a2 * e2 + a1 * e1 + a0)
    res0G = a2 * I1 + a1 * I0
    main = 2 * K / 3 * float(N) ** 3
    SS_pred = res0P - res0G
    SS_asym = N2 * (-0.5 * L * L + cst['c1'] * L + cst['c0'])
    # GZHN Theorem 2 at y=1: T1 - (K-1)N^3/3 + N^2 L^2/4, per N^2 L
    gz = (T1 - (K - 1) * float(N) ** 3 / 3 + N2 * L * L / 4) / (N2 * L)
    # same, after removing the predicted N^2 constant of T1 = P - 2Q + R (see lema_A_opus.md, Sec. 6)
    b1 = Z1 + Z0 * GAMMA                       # Laurent coeff. of sum S(k) k^{-s}: (-1/2)/s + b1 + ...
    CT = (7 * a2 / 4 - 1.5 * a1 + a0) - 2 * (0.75 + b1) - 0.5
    gzc = gz - CT / L
    r = dict(logN=L, N=N,
             SS_norm=SS / (N2 * L * L),
             SS_pred_norm=SS_pred / (N2 * L * L),
             SS_asym_norm=SS_asym / (N2 * L * L),
             SS_err_over_N15=(SS - SS_pred) / float(N) ** 1.5,
             SS_err_over_N2=(SS - SS_pred) / N2,
             SS_coef_L=(SS + 0.5 * N2 * L * L - cst['c0'] * N2) / (N2 * L),   # -> c1
             P2_err_over_N15=(P2 - main - res0P) / float(N) ** 1.5,
             G_err_over_N15=(G - main - res0G) / float(N) ** 1.5,
             G_minus_main_over_N2L2=(G - main) / (N2 * L * L),
             G_minus_main_over_N2L=(G - main) / (N2 * L),
             P2_minus_main_over_N2L2=(P2 - main) / (N2 * L * L),
             Q_check=(Q - float(N) ** 3 / 3 + 0.5 * N2 * L) / N2,   # -> 3/4 - gamma/2 - log(2pi)/2 (GZHN Lemma 4 integrated)
             GZHN_thm2_y1_coefL=gz,
             GZHN_thm2_y1_coefL_corrected=gzc)
    return r


if __name__ == "__main__":
    logs = [int(a) for a in sys.argv[1:]] or [12, 14, 16, 18, 20, 22, 24]
    cst = constants()
    print("constants:", json.dumps({k: round(v, 12) for k, v in cst.items()}, indent=1))
    print("Q_check limit 3/4 - gamma/2 - log(2pi)/2 =", 0.75 - GAMMA / 2 - 0.5 * math.log(2 * math.pi))
    out = {"constants": cst, "runs": {}}
    hdr = ("logN   SS/N2L2   pred(exact-res)  pred(asym)   (SS-pred)/N^1.5  (SS-pred)/N^2  coefL->c1"
           "   P2err/N^1.5  Gerr/N^1.5  (G-main)/N2L2  (G-main)/N2L  GZHN-y1 coefL (raw, corrected)  Q_check")
    print(hdr)
    for ln in logs:
        r = run(ln, cst)
        out["runs"][ln] = r
        print(f"2^{ln:<3d} {r['SS_norm']:+.6f}  {r['SS_pred_norm']:+.6f}       {r['SS_asym_norm']:+.6f}   "
              f"{r['SS_err_over_N15']:+.4f}          {r['SS_err_over_N2']:+.2e}     {r['SS_coef_L']:+.5f}   "
              f"{r['P2_err_over_N15']:+.4f}      {r['G_err_over_N15']:+.4f}     {r['G_minus_main_over_N2L2']:+.5f}"
              f"        {r['G_minus_main_over_N2L']:+.5f}      {r['GZHN_thm2_y1_coefL']:+.5f} {r['GZHN_thm2_y1_coefL_corrected']:+.6f}  {r['Q_check']:+.5f}",
              flush=True)
    json.dump(out, open("check_A.json", "w"), indent=1)
