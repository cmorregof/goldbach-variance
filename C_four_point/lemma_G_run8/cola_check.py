"""[M] Lema T de G2_prueba.md (cola puntual):  sum_{e>E} |T_e(n,h)| <= C (log 3E)^2 E^{-1} prod_{p | (n-h)(n+h), p in P'} (1 + p/(p-2)).
Se calcula la cola exacta = prod_{p in P'} (1+|t_p|) - sum_{e<=E} |T_e|  (criba multiplicativa hasta E)."""
import math, json, numpy as np
from g2_comun import primes_upto, factor, c2
pr = primes_upto(1 << 24); P5 = pr[pr >= 5]
LOGGEN = float(np.sum(np.log1p(2.0 / ((P5 - 2.0) * (P5 - 3.0)))))   # sum log(1+c2(p)) sobre p>=5
rng = np.random.default_rng(7)
rows = []
for trial in range(24):
    n = 2 * int(rng.integers(10 ** 5, 10 ** 7)); h = int(rng.integers(1, n - 1))
    pn = set(factor(n, pr)); spec = {}
    for p in set(factor(n - h, pr)) | set(factor(n + h, pr)) | set(factor(h, pr)):
        if p >= 5 and p not in pn:
            spec[p] = 0.0 if h % p == 0 else 1.0 / (p - 2)
    logtot = LOGGEN
    for p in pn:
        if p >= 5: logtot -= math.log1p(c2(p))
    for p, t in spec.items():
        logtot += math.log1p(t) - math.log1p(c2(p))
    tot = math.exp(logtot)
    for E in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6):
        lf = np.zeros(E + 1); ok = np.ones(E + 1, bool); ok[0] = False
        for p in pr[pr <= E]:
            p = int(p)
            if p < 5 or p in pn: ok[p::p] = False; continue
            if p * p <= E: ok[p * p::p * p] = False
            t = spec.get(p, c2(p))
            if t == 0.0: ok[p::p] = False; continue
            lf[p::p] += math.log(t)
        S = float(np.exp(lf[ok]).sum())        # incluye e = 1
        tail = tot - S
        Pm = 1.0
        for p, t in spec.items():
            if t > 0: Pm *= 1 + p / (p - 2)
        rows.append(dict(n=n, h=h, E=E, tail=tail, ratio=tail * E / (math.log(3 * E) ** 2 * Pm)))
    print(n, h, " ".join("E=%.0e ratio=%.3f" % (x["E"], x["ratio"]) for x in rows[-4:]))
print("max ratio", max(x["ratio"] for x in rows))
json.dump(rows, open("cola_check.json", "w"), indent=1)
