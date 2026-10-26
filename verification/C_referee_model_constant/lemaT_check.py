"""Árbitro: Lema T.  cola(E) := sum_{e>E} |T_e(n,h)|  frente a  (log 3E)^2 E^{-1} prod_{p | (n-h)(n+h), p in P'} (1 + p/(p-2)).
Cálculo exacto: sum_e |T_e| = prod_{p>=5, p∤n} (1 + |t_p|)  (producto absolutamente convergente; cola analítica para p > 2^22),
menos sum_{e<=E} |T_e| por criba multiplicativa sobre e<=E.  Casos: h genérico y h con n-h, n+h muy divisibles (peor caso)."""
import math, json, numpy as np
import arb_comun as A
pr = A.primes_upto(1 << 22); P5 = pr[pr >= 5].astype(float)
LOGGEN = float(np.sum(np.log1p(2.0 / ((P5 - 2) * (P5 - 3))))) + 2.0 / ((1 << 22) * math.log(1 << 22))
def tail_exact(n, h, E):
    pn = set(A.factor(n, pr))
    special = {}
    for p in set(A.factor(n - h, pr)) | set(A.factor(n + h, pr)) | set(A.factor(h, pr)):
        if p >= 5 and p not in pn:
            special[p] = 0.0 if h % p == 0 else 1.0 / (p - 2)
    logtot = LOGGEN - sum(math.log1p(A.c2(p)) for p in pn if p >= 5)
    for p, t in special.items(): logtot += math.log1p(t) - math.log1p(A.c2(p))
    tot = math.exp(logtot)
    la = np.zeros(E + 1); ok = np.ones(E + 1, bool); ok[0] = False
    for p in pr[pr <= E]:
        p = int(p)
        if p < 5 or p in pn: ok[p::p] = False; continue
        if p * p <= E: ok[p*p::p*p] = False
        t = special.get(p, A.c2(p))
        if t == 0: ok[p::p] = False; continue
        la[p::p] += math.log(t)
    S = float(np.exp(la[ok]).sum())
    Pm = 1.0
    for p, t in special.items():
        if t > 0: Pm *= 1 + p / (p - 2)
    return tot - S, Pm
rng = np.random.default_rng(92)
cases = []
for _ in range(10):
    n = 2 * int(rng.integers(5 * 10 ** 4, 10 ** 6)); h = int(rng.integers(1, n - 1)); cases.append((n, h, "aleatorio"))
# peores casos: n-h y n+h con muchos primos pequeños: n-h = 5*7*11*13*17 = 85085 -> h = n - 85085, n+h = 2n-85085
for n in (2 * 85085 + 2 * 1001, 4 * 85085, 10 ** 6 + 6):
    h = n - 85085
    if 0 < h < n: cases.append((n, h, "n-h=5·7·11·13·17"))
# h con n-h = 5·7·11, n+h = 13·17·19·... elegir n = (A+B)/2, h = (B-A)/2
Aa, Bb = 5 * 7 * 11 * 13, 17 * 19 * 23 * 29   # ambos impares -> A+B par
n, h = (Aa + Bb) // 2, (Bb - Aa) // 2
if n % 2 == 0: cases.append((n, h, "n∓h=5·7·11·13, 17·19·23·29"))
Aa, Bb = 5 * 7 * 11 * 13 * 17, 19 * 23 * 29 * 31
n, h = (Aa + Bb) // 2, (Bb - Aa) // 2
if n % 2 == 0: cases.append((n, h, "n∓h=5·7·11·13·17, 19·23·29·31"))
rows = []
for n, h, tag in cases:
    for E in (10 ** 3, 10 ** 4, 10 ** 5):
        tail, Pm = tail_exact(n, h, E)
        ratio = tail * E / (math.log(3 * E) ** 2 * Pm)
        rows.append(dict(n=n, h=h, tag=tag, E=E, tail=tail, Pm=Pm, ratio=ratio))
    print("%-30s n=%8d h=%8d Pm=%7.2f  " % (tag, n, h, Pm) + "  ".join("E=%.0e cola=%.2e ratio=%.4f" % (r['E'], r['tail'], r['ratio']) for r in rows[-3:]))
print("max ratio:", max(r['ratio'] for r in rows))
json.dump(rows, open("lemaT_check.json", "w"), indent=1)
