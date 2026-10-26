"""[M] Contraejemplo a la uniformidad literal de R3-E1: n = 2*3*5*...*p_k (primoriales), H fijo.
LHS(n,H) = sum_{0<|h|<=H}(r(n,h)-1)  frente a  RHS0 = -log^2 H/(2 S(n))  y frente al límite
Z(H) = sum_{0<|h|<=H}(S(h)-1) (S = serie singular de pares), predicho por r(n_k,h) -> S(h)."""
import json, numpy as np
from e1_comun import r_and_rho, sing, primes_upto, C2
pr = primes_upto(1 << 25)
def Sh(h):
    if h % 2: return 0.0
    s = 2 * C2; m = h
    for p in pr[1:]:
        p = int(p)
        if p * p > m: break
        if m % p == 0:
            s *= (p - 1) / (p - 2)
            while m % p == 0: m //= p
    if m > 2: s *= (m - 1) / (m - 2)
    return s
prim = [2, 6, 30, 210, 2310, 30030, 510510, 9699690]
Hs = [2, 4, 10, 30, 100]
out = []
Z = {H: 2 * sum(Sh(h) - 1 for h in range(1, H + 1)) for H in Hs}
print("Z(H) =", {H: round(v, 4) for H, v in Z.items()})
for n in prim[3:]:
    r, _ = r_and_rho(n, min(n - 1, max(Hs)), pr)
    S = sing(n, pr)
    row = dict(n=n, S=S)
    for H in Hs:
        if H > n - 1: continue
        lhs = 2 * float(np.sum(r[:H] - 1))
        row[f"H{H}"] = dict(lhs=lhs, rhs0=-np.log(H) ** 2 / (2 * S), Z=Z[H])
    out.append(row)
    print(f"n={n:9d} S(n)={S:.4f} " + " | ".join(f"H={H}: LHS={row[f'H{H}']['lhs']:+.4f} RHS0={row[f'H{H}']['rhs0']:+.4f}" for H in Hs if f'H{H}' in row))
json.dump(dict(Z=Z, rows=out), open("contraejemplo_primorial.json", "w"), indent=1)
