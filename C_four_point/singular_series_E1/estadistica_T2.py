"""[M] Tamaño de T2 = sum (r - rho) (agudo y Fejér) sobre una muestra de n pares.
Lema G (GAP) predice |T2| = O(log H). Se registra max |T2|/log H por H."""
import json, numpy as np
from e1_comun import r_and_rho, primes_upto
pr = primes_upto(1 << 23)
rng = np.random.default_rng(20261025)
ns = sorted(set([int(2 * x) for x in rng.integers(50000, 1500000, 36)] +
                [30030 * 33 + 2, 30030 * 33 - 2, 510510 * 3 + 4, 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 - 6]))
Hs = [10, 100, 1000, 10000, 100000]
rows = []
for n in ns:
    r, rho = r_and_rho(n, n - 1, pr)
    rec = dict(n=n)
    for H in Hs + [n // 2, n - 1]:
        if H > n - 1: continue
        hh = np.arange(1, H + 1); w = 1 - hh / H
        d = r[:H] - rho[:H]
        rec[str(H) if H in Hs else ("n/2" if H == n // 2 else "n-1")] = dict(H=H, T2=2 * float(d.sum()), T2f=2 * float(np.dot(w, d)))
    rows.append(rec)
summ = {}
for key in [str(H) for H in Hs] + ["n/2", "n-1"]:
    vals = [(x[key]["H"], x[key]["T2"], x[key]["T2f"]) for x in rows if key in x]
    a = np.array(vals)
    summ[key] = dict(N=len(a), max_abs_T2_over_logH=float(np.max(np.abs(a[:, 1]) / np.log(a[:, 0]))),
                     mean_T2=float(a[:, 1].mean()), sd_T2=float(a[:, 1].std()),
                     max_abs_T2f=float(np.max(np.abs(a[:, 2]))), mean_T2f=float(a[:, 2].mean()))
    print(key, {k: round(v, 3) for k, v in summ[key].items()})
json.dump(dict(rows=rows, summary=summ), open("estadistica_T2.json", "w"), indent=1)
