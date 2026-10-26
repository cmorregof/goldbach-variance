"""Árbitro 3: T2 = sum (r - rho) (agudo y Fejér) con mi r_def, dos n, varios H. Solo [M]."""
import math, numpy as np
from arb_comun import sieve, r_def, rho_def, sing
for n, P in ((30030, 1 << 17), (100000, 1 << 18), (99990, 1 << 18)):
    pr = sieve(P); assert pr[-1] >= 2 * n + 2
    hs = np.arange(1, n); r = r_def(n, hs, pr); rho = rho_def(n, hs, pr); S = sing(n, pr)
    print(f"n={n} S={S:.4f}")
    for H in (10, 100, 1000, 10000, n // 2, n - 1):
        w = 1 - hs[:H] / H; d = r[:H] - rho[:H]
        T2 = 2 * float(d.sum()); T2f = 2 * float(np.dot(w, d))
        tot = 2 * float(np.sum(r[:H] - 1)); totf = 2 * float(np.dot(w, r[:H] - 1)); F0 = -math.log(H) ** 2 / (2 * S)
        print(f"  H={H:6d}: T2 agudo={T2:+8.3f} (/log H={T2/math.log(H):+.3f})  T2 Fejér={T2f:+7.3f} | exacto agudo={tot:+9.3f} Fejér={totf:+9.3f} F0={F0:+8.3f}")
