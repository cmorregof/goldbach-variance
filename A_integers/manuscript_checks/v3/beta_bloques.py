"""Dyadic block sums of beta(d) = 4 C2 prod_{p | d/2} 1/(p-2) for even squarefree d (beta=0 otherwise).
Used for the constant C_beta = sup_D sum_{D<d<=2D} beta(d) in Hypothesis H_FG (manuscript v3). Run 2026-10-25."""
import numpy as np, json
M = 1 << 23                      # d <= 2M
lim = 2*M + 1
isp = np.ones(lim, bool); isp[:2] = False
for p in range(2, int(lim**0.5)+1):
    if isp[p]: isp[p*p::p] = False
pr = np.nonzero(isp)[0]
C2 = np.prod(1 - 1.0/(pr[1:].astype(float)-1)**2)   # truncated at p<=2M (rel. err ~ 1/(2M log 2M))
# f(m) = mu^2(m) prod_{p|m} 1/(p-2) for odd m <= M
f = np.ones(M+1); f[0] = 0
f[0::2] = 0                     # m odd only
for p in pr[1:]:
    if p > M: break
    f[p::p] /= (p-2)
    f[(p*p)::(p*p)] = 0
beta = np.zeros(lim)            # beta(2m) = 4 C2 f(m)
beta[2::2][:M] = 4*C2*f[1:M+1]
cs = np.cumsum(beta)
out = {}
for j in range(0, 22):
    D = 1 << j
    s = cs[2*D] - cs[D]
    out[D] = s
    print(D, round(s, 4))
y = 1 << 22
print("sum_{d<=y} beta - 2 log y at y=2^22:", cs[y] - 2*np.log(y))
print("sup over D>=8:", max(v for D, v in out.items() if D >= 8), " sup over all D:", max(out.values()), " 2log2 =", 2*np.log(2))
json.dump({"C2_truncated": C2, "block_sums": out, "sum_le_y_minus_2logy_at_2^22": cs[y]-2*np.log(y)}, open("beta_bloques.json", "w"), indent=1)
