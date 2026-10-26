"""Numerical check (consistency, not a test of GRH) of Lemma ms (v3):
   S(q) := sum_{(a,q)=1} sum_{m<=N} |eta_a(m)|^2  <<  q N^2 log^2(2q),
   eta_a(y) = sum_{m<=y} Lambda(m) e(am/q) - mu(q) y/phi(q).  Also Cramer's sum_{m<=N} Delta(m)^2 / N^2 (q=1).
Run 2026-10-25, N = 2^18 and 2^20, squarefree q <= 60."""
import numpy as np, json, math, sys
def lam(N):
    isp = np.ones(N+1, bool); isp[:2] = False
    for p in range(2, int(N**0.5)+1):
        if isp[p]: isp[p*p::p] = False
    L = np.zeros(N+1)
    for p in np.nonzero(isp)[0]:
        lp = math.log(p); pk = int(p)
        while pk <= N: L[pk] = lp; pk *= int(p)
    return L
def mu_phi(q):
    n=q; ps=[]; p=2
    while p*p<=n:
        if n%p==0:
            ps.append(p); n//=p
            if n%p==0: return 0, None
        p+=1
    if n>1: ps.append(n)
    phi=q
    for p in ps: phi = phi//p*(p-1)
    return (-1)**len(ps), phi
res={}
for k in [int(a) for a in sys.argv[1:]] or [18,20]:
    N=1<<k; L=lam(N); m=np.arange(N+1)
    rows={}
    for q in range(1,61):
        mu,phi=mu_phi(q)
        if mu==0: continue
        S=0.0
        for a in range(1,q+1) if q>1 else [0]:
            if q>1 and math.gcd(a,q)!=1: continue
            ph=np.exp(2j*np.pi*((a*m)%q)/q) if q>1 else np.ones(N+1)
            eta=np.cumsum(L*ph) - mu/phi*m
            S+=float(np.sum(np.abs(eta[1:])**2))
        r1=S/(q*N**2*math.log(2*q)**2); r2=S/(phi*N**2)
        rows[q]=dict(S_over_N2=S/N**2, ratio_qlog2=r1, S_over_phiN2=r2)
        print(f"N=2^{k} q={q:2d}  S/N^2={S/N**2:9.4f}  S/(q N^2 log^2 2q)={r1:.4f}  S/(phi(q) N^2)={r2:.4f}", flush=True)
    res[k]=rows
json.dump(res,open("check_lema_ms.json","w"),indent=1)
