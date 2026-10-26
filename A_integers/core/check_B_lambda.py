"""check_B_lambda.py -- test of the exact 'completed square' identity (lema_B_fable.md, Prop. 1):

for ANY lambda supported on [1,N], f(alpha)=sum lambda(m)e(m alpha), psi2^lam(h)=sum_m lam(m)lam(m+h),
R^lam(n)=sum_{a+b=n} lam(a)lam(b):

  X = 4 int_0^1 (Im S conj f)^2                                   (I)
      - 2 psi2(0) ||lambda||^2                                    (II)
      + 2[ sum_n S(n)nu(n)R^lam(n) - sum_{h!=0} M(h) psi2^lam(h) ] (III)
      + 2 sum_{h!=0} E(h) (M(h) - psi2^lam(h))                    (IV)
      - 2 sum_n delta(n) (S(n)nu(n) - R^lam(n))                   (V)

We take lambda = Lambda*_Q(m) = sum_{q<=Q} mu(q) c_q(m)/phi(q) (truncated Ramanujan expansion of Lambda)
and print each term / N^2 for several Q, for N = 2^LOGN.  (I) is computed by Parseval:
(I) = 2 sum_h psi2(h) psi2^lam(h) - 2 sum_n R(n) R^lam(n).
"""
import sys, json, numpy as np, scipy.fft as sf
from math import log, sqrt

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

def lambda_Q(N, Q, mu, phi):
    """lambda(m) = sum_{d|m, d<=Q} d mu(d)/phi(d) * L_d(Q/d),  L_d(x)=sum_{r<=x,(r,d)=1} mu^2(r)/phi(r)."""
    lam = np.zeros(N + 1)
    r = np.arange(Q + 1)
    musq_over_phi = np.zeros(Q + 1); musq_over_phi[1:] = (mu[1:Q+1]**2)/phi[1:Q+1]
    from math import gcd
    for d in range(1, Q + 1):
        if mu[d] == 0: continue
        x = Q // d
        if d == 1:
            Ld = musq_over_phi[1:x+1].sum()
        else:
            rr = np.arange(1, x + 1)
            g = np.gcd(rr, d)
            Ld = musq_over_phi[1:x+1][g == 1].sum()
        lam[d::d] += d*mu[d]/phi[d]*Ld
    lam[0] = 0
    return lam

def run(LOGN, Qs):
    N = 1 << LOGN; lim = 2*N + 1
    pr = sieve(lim)
    Lam = np.zeros(N + 1)
    for p in pr[pr <= N]:
        lp = log(p); pk = int(p)
        while pk <= N: Lam[pk] = lp; pk *= int(p)
    L = 4*N
    F = sf.rfft(Lam, L, workers=2)
    R = sf.irfft(F*F, L, workers=2)[:lim]
    psi = sf.irfft(F*np.conj(F), L, workers=2)[:N+1].copy(); del F
    C2 = 0.6601618158468696      # exact twin prime constant
    S = np.zeros(lim); S[0::2] = 2*C2
    for p in pr[1:]:
        S[0::p] *= (p-1)/(p-2)
    S[0] = 0.0
    n = np.arange(lim); nu = np.where(n <= N+1, np.maximum(n-1, 0), 2*N - n + 1).astype(float)
    h = np.arange(1, N+1); M = S[1:N+1]*(N - h); E = psi[1:] - M
    dG = R - S*nu
    X = 4*(M*E).sum() - 2*(S*nu*dG).sum()
    psi20 = psi[0]
    Dmax = max(Qs); mu, phi = mobius_phi(Dmax)
    out = dict(logN=log(N), X_over_N2=X/N**2, terms={})
    print(f"N=2^{LOGN}  X/N^2 = {X/N**2:.4f}   (log N = {log(N):.2f})")
    print(f"{'Q':>6} {'(I)':>9} {'(II)':>9} {'(III)':>9} {'(IV)':>9} {'(V)':>9} {'sum':>9} | {'||lam||^2/N':>11} {'L(Q)':>7} {'(II)+(III)':>10}")
    for Q in Qs:
        lam = lambda_Q(N, Q, mu, phi)
        Fl = sf.rfft(lam, L, workers=2)
        Rl = sf.irfft(Fl*Fl, L, workers=2)[:lim]
        psil = sf.irfft(Fl*np.conj(Fl), L, workers=2)[:N+1].copy(); del Fl
        I = 2*(psi20*psil[0] + 2*(psi[1:]*psil[1:]).sum()) - 2*(R*Rl).sum()
        II = -2*psi20*(lam**2).sum()
        III = 2*((S*nu*Rl).sum() - 2*(M*psil[1:]).sum())
        IV = 2*2*(E*(M - psil[1:])).sum()
        V = -2*(dG*(S*nu - Rl)).sum()
        tot = I + II + III + IV + V
        LQ = sum(mu[q]**2/phi[q] for q in range(1, Q + 1))
        row = [I, II, III, IV, V, tot]
        out['terms'][str(Q)] = dict(I=I/N**2, II=II/N**2, III=III/N**2, IV=IV/N**2, V=V/N**2, sum=tot/N**2,
                                    lam2_over_N=float((lam**2).sum()/N), LQ=LQ)
        print(f"{Q:>6} " + " ".join(f"{v/N**2:>9.3f}" for v in row) + f" | {(lam**2).sum()/N:>11.4f} {LQ:>7.4f} {(II+III)/N**2:>10.3f}")
        sys.stdout.flush()
    return out

if __name__ == "__main__":
    res_all = {}
    for LOGN in [int(a) for a in sys.argv[1:]] or [16]:
        N = 1 << LOGN
        Qs = sorted(set([1, 2, 3, 6, 10, 30, 100, 300, 1000, int(sqrt(N)), int(N**0.75)]))
        res_all[LOGN] = run(LOGN, Qs)
    json.dump(res_all, open("/home/claude/pruebas/check_B_lambda.json", "w"), indent=1)
