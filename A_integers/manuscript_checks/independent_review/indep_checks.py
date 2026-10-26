"""opus2 review: independent implementation (not derived from the authors' scripts).
Checks, for N=2^k given on the command line:
  * Table tab:q1 (X(1), 2Psi_1+2c_B N^2, Psi_1, Delta(N)^2/N, [X(1)-pred]/N^{3/2});
  * Table tab:SS row (SS/(N^2L^2), residue prediction with exact I_0, I_1);
  * smooth weight w=1-x: SS_w, X_w, bracket, Det_w(1), 2W_w(1) (Section 11 'Smooth weights');
  * window identity (Prop. window) at theta = 1/3 and theta = sqrt(2)-1, and Cor. permod at q = 6;
  * reflection identity (Lemma refl) with a random g.
Singular series computed from a smallest-prime-factor sieve with the full C_2 (30 digits), not truncated.
"""
import sys, math, json
import numpy as np, scipy.fft as sf, mpmath as mp

C2 = 0.66016181584686957392781211001455577843262336028473
g_e = 0.57721566490153286
a2, a1, a0 = -0.5, -2.1096386009329406, -2.6540662590673156
cB = 0.5 + g_e + math.log(math.pi/2)

def spf_sieve(n):
    spf = np.zeros(n+1, np.int64)
    for p in range(2, n+1):
        if spf[p] == 0:
            spf[p::p][spf[p::p] == 0] = p
    return spf

def setup(N):
    lim = 2*N+1
    spf = spf_sieve(lim)
    isp = (spf == np.arange(lim+1)) & (np.arange(lim+1) >= 2)
    Lam = np.zeros(N+1)
    for p in np.nonzero(isp[:N+1])[0]:
        pk = int(p)
        while pk <= N:
            Lam[pk] = math.log(p); pk *= int(p)
    # singular series via factorisation
    S = np.zeros(lim+1)
    for k in range(2, lim+1, 2):
        m, val = k, 2*C2
        while m % 2 == 0: m //= 2
        while m > 1:
            p = int(spf[m])
            val *= (p-1)/(p-2)
            while m % p == 0: m //= p
        S[k] = val
    return Lam, S

def corr(a, b, L):
    fa = sf.rfft(a, L); fb = sf.rfft(b, L)
    return sf.irfft(fa*fb, L), sf.irfft(fa*np.conj(fb), L)

def run(k):
    N = 1 << k; L = math.log(N); lim = 2*N
    Lam, S = setup(N)
    out = {}
    for name, w in (("box", np.r_[0.0, np.ones(N)]), ("1-x", np.r_[0.0, 1 - np.arange(1, N+1)/N])):
        lam = Lam*w
        F = 4*N
        Rw, ac = corr(lam, lam, F)
        Rw = Rw[:lim+1]; psi2 = ac[:N+1]              # psi2(h), h>=0
        nuw, rho_ac = corr(w, w, F)
        nuw = nuw[:lim+1]; rho = rho_ac[:N+1]
        h = np.arange(1, N+1)
        M = S[1:N+1]*rho[1:]; E = psi2[1:] - M
        dl = Rw - S[:lim+1]*nuw
        A0 = psi2[0]**2; Ep = 2*np.sum(E**2)
        SS = 2*np.sum(M**2) - np.sum((S[:lim+1]*nuw)**2)
        X = 2*2*np.sum(M*E) - 2*np.sum(S[:lim+1]*nuw*dl)
        Var = np.sum(dl**2)
        X1 = 2*(2*np.sum(rho[1:]*E) - np.sum(nuw*dl))   # q=1 component
        Det1 = 2*(-rho[0]*psi2[0] + np.sum(S[:lim+1]*nuw**2) - 2*np.sum(S[1:N+1]*rho[1:]**2))
        r = dict(A0=A0, Ep=Ep, SS=SS, X=X, Var=Var, relerr=abs(A0+Ep+SS+X-Var)/Var, X1=X1, Det1=Det1)
        if name == "box":
            psi = np.cumsum(Lam)                        # psi(m), m=0..N
            kk = np.arange(1, N)
            Psi1 = np.sum((psi[N] - psi[kk] - psi[N-kk])**2)
            DeltaN = psi[N] - N
            pred = 2*Psi1 + 2*cB*N**2
            r.update(Psi1=Psi1, DeltaN=DeltaN, X1_over_N2=X1/N**2, pred_over_N2=pred/N**2,
                     Psi1_over_N2=Psi1/N**2, D2_over_N=DeltaN**2/N, diff_over_N32=(X1-pred)/N**1.5)
            # Theorem SS prediction (exact I0, I1)
            mp.mp.dps = 30
            gt = lambda t: (t-1) if t <= N+1 else (2*N+1-t)
            I0 = mp.quad(lambda t: gt(t)**2/t, [1, N+1, 2*N+1])
            I1 = mp.quad(lambda t: gt(t)**2*mp.log(t)/t, [1, N+1, 2*N+1])
            e1 = L - 1.5; e2 = L*L/2 - 1.5*L + 1.75
            predSS = 2*N**2*(a2*e2 + a1*e1 + a0) - (a2*float(I1) + a1*float(I0))
            r.update(SS_over_N2L2=SS/(N*N*L*L), predSS_over_N2L2=predSS/(N*N*L*L), SSdiff_over_N32=(SS-predSS)/N**1.5)
            # window identity at theta = 1/3 and sqrt2-1 (full complex check), and Cor permod at q = 6
            m = np.arange(N+1)
            for th in (1/3, math.sqrt(2)-1):
                ph = np.exp(2j*np.pi*m*th)
                hh = np.arange(-(N-1), N)
                lhs1 = np.sum((N-np.abs(hh))*psi2[np.abs(hh)]*np.exp(2j*np.pi*hh*th))
                nn = np.arange(lim+1)
                lhs2 = np.sum(np.r_[0, 0, np.maximum(0, np.minimum(nn[2:]-1, 2*N-nn[2:]+1))]*Rw*np.exp(2j*np.pi*nn*th))
                pt = np.cumsum(Lam*ph)
                rhs = 2*(pt[N].imag)**2 + np.sum(np.abs(pt[N] - pt[kk] - np.conj(pt[N-kk]))**2)
                r[f"window_relerr_theta{th:.4f}"] = abs((lhs1-lhs2).real - rhs)/rhs
            q = 6; phi = 2; mu = 1
            W = 0.0
            for a in (1, 5):
                eta = np.cumsum(Lam*np.exp(2j*np.pi*a*m/q)) - mu/phi*m
                W += 2*(eta[N].imag)**2 + np.sum(np.abs(eta[N]-eta[kk]-np.conj(eta[N-kk]))**2)
            cq = lambda x: np.array([sum(math.cos(2*math.pi*a*int(t)/q) for a in (1, 5)) for t in x])
            hpos = np.arange(1, N); cqh = cq(hpos); nn = np.arange(lim+1); cqn = cq(nn)
            Xq = 2*mu**2/phi**2*(2*np.sum(cqh*(N-hpos)*E[:N-1]) - np.sum(cqn*nuw*dl))
            FM = lambda d: 2*np.sum(S[d:N:d]*(N-np.arange(d, N, d))**2)
            Fmod = lambda d: np.sum(S[d:lim+1:d]*nuw[d:lim+1:d]**2)
            Det = 2*mu**2/phi**2*(-phi*N*psi2[0] + sum(d*(1 if (q//d) in (1, 6) else -1)*(Fmod(d)-FM(d)) for d in (1, 2, 3, 6)))
            r["permod_q6_relerr"] = abs(Xq - (2/phi**2*W + Det))/abs(Xq)
            # Prop det prediction for Det(6): 2/phi * N^2 [cB - w'_3]
            wp3 = 3*1/4*math.log(3)
            r["Det6_over_N2"] = Det/N**2; r["Det6_pred_over_N2"] = 2/phi*(cB - wp3)
        else:
            r0 = 1/3; css = -0.20638145; cw = -0.61378886; cBw = 0.20962022
            r.update(SSw_norm=SS/N**2 + r0**2/2*L*L - css*L, Xw_over_N2=X/N**2,
                     bracket=(Var - Ep - r0**2/2*N*N*L*L - cw*N*N*L)/N**2,
                     Det1_over_N2=Det1/N**2, twoW1_over_N2=(X1-Det1)/N**2, Det1_pred=2*cBw)
        out[name] = {kk_: float(v) for kk_, v in r.items()}
    # reflection identity with random g
    rng = np.random.default_rng(0); gr = rng.standard_normal(2*N+1); gr[0] = 0
    nn = np.arange(2*N+1); nu = np.where(nn <= N+1, np.maximum(nn-1, 0), 2*N-nn+1)
    B = lambda j, x: np.sum(np.clip(x-nn[1:], 0, None)**j*gr[1:]*(nn[1:] <= x))
    lhs = np.sum(nu.astype(float)**2*gr); rhs = B(2, 2*N+1) - 4*N*B(1, N+1)
    out["refl_relerr"] = float(abs(lhs-rhs)/abs(lhs))
    return out

res = {}
for k in [int(a) for a in sys.argv[1:]] or [14, 16]:
    res[k] = run(k)
    print(k, json.dumps(res[k], indent=1), flush=True)
json.dump(res, open("indep_checks.json", "w"), indent=1)
