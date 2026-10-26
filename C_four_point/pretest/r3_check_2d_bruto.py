"""Brute-force check of r3_modelo_2d.py on a tiny block: per pair (a,a'), product over primes p<=PB by residues
(+ generic factor for PB<p via the same L4 tail), p=3 factor from the ACTUAL residues of a,a',n-a,n-a'."""
import sys, numpy as np
from r3_common import sieve, singular_S, euler_sum
K=int(sys.argv[1]); lo,hi=1<<K,1<<(K+1)
primes=sieve(1<<22); S=singular_S(hi,primes)
PB=4*hi
f=lambda p,nu: np.log1p(-nu/p)-4*np.log1p(-1.0/p)
ps=[int(p) for p in primes if 5<=p<=PB]
Ltail=euler_sum(primes[primes>PB], lambda p: f(p,4), 5, -6.0)
cache={}
def S4(h,k):
    key=(h,k)
    if key in cache: return cache[key]
    if h%2: cache[key]=0.0; return 0.0
    s=Ltail
    for p in ps:
        nu=len({0,h%p,(k-h)%p,k%p}); s+=f(p,nu)
    cache[key]=8*np.exp(s); return cache[key]
tot={0:0.0,2:0.0,4:0.0}; base={0:0.0,2:0.0,4:0.0}
for n in range(lo,hi,2):
    c=n%6; acc=0.0; cnt=0
    for a in range(1,n):
        for ap in range(1,n):
            if ap==a or ap==n-a or 2*a==n or 2*ap==n: continue
            h=ap-a; k=n-2*a
            f3=(1.5**4) if (a%3 and ap%3 and (n-a)%3 and (n-ap)%3) else 0.0
            acc+=S4(h,k)*f3-S[n]**2
    tot[c]+=acc-2*n*S[n]**2; base[c]+=S[n]*n*np.log(n)**2
print("brute K=%d"%K, {c:tot[c]/base[c] for c in tot}, "total", sum(tot.values())/sum(base.values()))
