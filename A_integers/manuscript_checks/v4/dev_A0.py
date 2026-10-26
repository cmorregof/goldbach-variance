# m11: is dev = [(A0+SS) - N^2L^2/2 - (c1-2)N^2L - (c0+1)N^2]/N^2 dominated by the PNT error in A0 = psi2(0)^2?
# psi2(0) = sum_{m<=N} Lambda(m)^2 ; A0 pred = N^2L^2 - 2N^2L + N^2 ; dev_A0 = (A0 - pred)/N^2
import numpy as np, json
M = 2**24
s = np.ones(M+1, dtype=bool); s[:2] = False
for p in range(2, int(M**0.5)+1):
    if s[p]: s[p*p::p] = False
primes = np.nonzero(s)[0]
lam2 = np.zeros(M+1)
for p in primes:
    lp = np.log(p); pk = p
    while pk <= M:
        lam2[pk] = lp*lp; pk *= p
cum = np.cumsum(lam2)
dev = json.load(open('../v3_scratch/tabla_decomp_v3.json'))
out = {}
for k in range(14, 25):
    N = 2**k; L = np.log(N); psi20 = cum[N]
    A0 = psi20**2; pred = N*N*L*L - 2*N*N*L + N*N
    devA0 = (A0 - pred)/N**2
    approx = 2*L*(psi20 - N*L + N)/N
    out[k] = (devA0, approx)
    print(k, "devA0=%.4f  2L(psi2(0)-NL+N)/N=%.4f" % (devA0, approx))
json.dump({str(k): {"devA0": v[0], "approx": v[1]} for k, v in out.items()}, open('dev_A0.json', 'w'), indent=1)
