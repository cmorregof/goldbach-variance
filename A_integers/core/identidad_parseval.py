"""Descomposicion exacta de la varianza de Goldbach (version 'caja', a,b <= N) via Parseval:
   sum_n R(n)^2 = sum_h psi2(h)^2,  R(n)=sum_{a+b=n, a,b<=N} L(a)L(b),  psi2(h)=sum_{n,n+h<=N} L(n)L(n+h).
Con nu(n)=#{a,b<=N: a+b=n}, M(h)=S(h)(N-|h|) (h!=0), E(h)=psi2(h)-M(h):
   VarG := sum_n (R - S nu)^2
        = psi2(0)^2                                  (A0: diagonal h=0)
        + sum_{h!=0} E(h)^2                          (Ep: varianza de pares de primos)
        + [sum_{h!=0} M(h)^2 - sum_n S(n)^2 nu(n)^2] (SS: series singulares; aqui vive el -1/4 N^2 log^2 de GZHN)
        + 2 sum_{h!=0} M(h)E(h)                      (X1)
        - 2 sum_n S(n)nu(n)(R(n)-S(n)nu(n))          (X2)
Todo normalizado por N^2 log^2 N.
"""
import sys, json, numpy as np, scipy.fft as sf
res = {}
for LOGN in [int(a) for a in sys.argv[1:]] or [18, 20, 22, 24]:
    N = 1 << LOGN; lim = 2*N + 1
    isp = np.ones(lim, bool); isp[:2] = False
    for p in range(2, int(lim**0.5)+1):
        if isp[p]: isp[p*p::p] = False
    pr = np.nonzero(isp)[0]
    Lam = np.zeros(N + 1)
    for p in pr[pr <= N]:
        lp = np.log(p); pk = int(p)
        while pk <= N: Lam[pk] = lp; pk *= int(p)
    L = 4*N
    F = sf.rfft(Lam, L, workers=2)
    R = sf.irfft(F*F, L, workers=2)[:lim]                       # R(n), n=0..2N
    ac = sf.irfft(F*np.conj(F), L, workers=2)                    # autocorrelacion
    psi = ac[:N+1]                                               # psi2(h), h=0..N (simetrica)
    # serie singular S(k) para k = 0..2N (S(0) no se usa; impares 0)
    C2 = np.prod(1 - 1.0/(pr[1:].astype(float)-1)**2)
    S = np.zeros(lim); S[0::2] = 2*C2
    for p in pr[1:]:
        S[0::p] *= (p-1)/(p-2)                                   # solo afecta pares (impares ya 0)
    n = np.arange(lim); nu = np.where(n <= N+1, np.maximum(n-1, 0), 2*N - n + 1).astype(float)
    h = np.arange(1, N+1); M = S[1:N+1]*(N - h); E = psi[1:] - M
    A0 = psi[0]**2
    Ep = 2*(E**2).sum()
    SS = 2*(M**2).sum() - ((S*nu)**2).sum()
    X1 = 2*2*(M*E).sum()
    dG = R - S*nu
    X2 = -2*(S*nu*dG).sum()
    VarG = (dG**2).sum()
    lhs = (R**2).sum(); rhs = psi[0]**2 + 2*(psi[1:]**2).sum()
    norm = N**2*np.log(N)**2
    r = dict(logN=float(np.log(N)), parseval_relerr=float(abs(lhs-rhs)/lhs),
             VarG=VarG/norm, A0=A0/norm, Ep=Ep/norm, SS=SS/norm, X1=X1/norm, X2=X2/norm,
             suma=(A0+Ep+SS+X1+X2)/norm)
    res[LOGN] = r
    print(f"N=2^{LOGN} logN={r['logN']:.2f} parseval_err={r['parseval_relerr']:.1e} | VarG={r['VarG']:.4f} = "
          f"A0 {r['A0']:.4f} + Ep {r['Ep']:.4f} + SS {r['SS']:+.4f} + X1 {r['X1']:+.4f} + X2 {r['X2']:+.4f}"
          f"  (suma {r['suma']:.4f})", flush=True)
json.dump(res, open("identidad_parseval.json", "w"), indent=1)
