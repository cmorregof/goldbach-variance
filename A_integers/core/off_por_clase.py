import numpy as np, scipy.fft as sf
LOGX=24; X=1<<LOGX; lim=X+1; L=2*X
isp=np.ones(lim,bool); isp[:2]=False
for p in range(2,int(lim**0.5)+1):
    if isp[p]: isp[p*p::p]=False
pr=np.nonzero(isp)[0]
Lam=np.zeros(lim)
for p in pr:
    lp=np.log(p); pk=int(p)
    while pk<lim: Lam[pk]=lp; pk*=int(p)
def sc(a):
    f=sf.rfft(a,L,workers=2); f*=f; return sf.irfft(f,L,workers=2)[:lim:2].copy()
W=sc(Lam); D2=2*sc(Lam**2)
Nev=np.arange(0,lim,2); C2=np.prod(1-1.0/(pr[1:].astype(float)-1)**2)
S=np.full(len(Nev),2*C2)
for p in pr[1:]:
    if p>len(Nev): break
    S[0::p]*=(p-1)/(p-2)
N=Nev.astype(float); d=W-S*N
m=(N>=X/2)&(N<X); lg=np.log(N[m])
off=(d[m]**2-D2[m])/(N[m]*lg**2); dg=D2[m]/(N[m]*lg**2); Sm=S[m]
# regresion del termino off sobre S y S^2 (por N, ruidoso; usar medias por valor de S agrupado)
bins=np.quantile(Sm,np.linspace(0,1,21)); idx=np.digitize(Sm,bins[1:-1])
gS=np.array([Sm[idx==k].mean() for k in range(20)]); gO=np.array([off[idx==k].mean() for k in range(20)])
gD=np.array([dg[idx==k].mean() for k in range(20)])
for deg,name in ((1,'S'),(2,'S^2')):
    A=np.vstack([gS**deg,np.ones_like(gS)]).T; c,*_=np.linalg.lstsq(A,gO,rcond=None)
    r=gO-A@c; print(f"off ~ a*{name}+b : a={c[0]:+.3f} b={c[1]:+.3f}  rms resid={r.std():.3f}")
A=np.vstack([gS,gS**2]).T; c,*_=np.linalg.lstsq(A,gO,rcond=None); print("off ~ a S + b S^2 (sin cte):",np.round(c,3), "rms", (gO-A@c).std().round(3))
print("diag/(N log^2) vs S slope:", np.polyfit(gS,gD,1).round(3))
for cl in (0,2,4):
    mm=(Nev[m]%6)==cl
    print(f"2m%6={cl}: <S>={Sm[mm].mean():.3f}  <off>/<S>={off[mm].mean()/Sm[mm].mean():+.3f}  <off>/<S^2>={off[mm].mean()/(Sm[mm]**2).mean():+.3f}  <diag>/<S>={dg[mm].mean()/Sm[mm].mean():.3f}")
