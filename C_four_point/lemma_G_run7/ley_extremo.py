"""[M] Ley del extremo con amplitud:  T2^F(n,H) ≈ -1_{3|n} c(n) Psi(H/n),
   c(n) = prod_{p>=5, p∤n} (1 - 1/(p-2)^2)   (coeficiente del hoyo de dimensión 1 en u = n∓h, ver G_prueba §9),
   Psi(t) = 2 ∫_0^t (1-s/t) ds/(1-s^2).
Datos: efecto_3n.json (8 n, 8 valores de H/n) y run6_E1/estadistica_T2.json (40 n; H = n/2, n-1, 1e4, 1e5)."""
import json, math, numpy as np
from scipy.integrate import quad
from g_comun import primes_upto, prime_factors
pr = primes_upto(1 << 22)
def Psi(t): return 2 * quad(lambda s: (1 - s / t) / (1 - s * s), 0, min(t, 1 - 1e-12), limit=400)[0]
P = pr[pr >= 5].astype(float); base = float(np.sum(np.log1p(-1 / (P - 2) ** 2))) - 1 / (P[-1] * math.log(P[-1]))
def c(n):
    s = base
    for p in prime_factors(n, pr):
        if p >= 5: s -= math.log1p(-1 / (p - 2) ** 2)
    return math.exp(s)
out = dict(c_infty=math.exp(base), A=[], B=[])
print("c(n) para n sin primos >=5 comunes: %.4f" % math.exp(base))
for row in json.load(open("efecto_3n.json")):
    n = row["n"]; pred = [-(c(n) * Psi(f)) if n % 3 == 0 else 0.0 for f in row["frac"]]
    res = [v - p for v, p in zip(row["T2f"], pred)]
    out["A"].append(dict(n=n, c=c(n), resid=res))
    print("A", n, "c=%.3f" % c(n), "resid:", " ".join("%+.3f" % x for x in res))
d = json.load(open("/home/claude/ariadna/R3/run6_E1/estadistica_T2.json"))["rows"]
for key in ["10000", "100000", "n/2", "n-1"]:
    r0, r1 = [], []
    for x in d:
        if key not in x: continue
        n = x["n"]; H = x[key]["H"]; v = x[key]["T2f"]
        r0.append(v); r1.append(v + (c(n) * Psi(H / n) if n % 3 == 0 else 0.0))
    r0, r1 = np.array(r0), np.array(r1)
    out["B"].append(dict(key=key, rms_raw=float(np.sqrt((r0**2).mean())), rms_resid=float(np.sqrt((r1**2).mean())), max_resid=float(np.abs(r1).max())))
    print("B", key, "rms bruto %.3f -> rms residuo %.3f (max %.3f)" % (out["B"][-1]["rms_raw"], out["B"][-1]["rms_resid"], out["B"][-1]["max_resid"]))
json.dump(out, open("ley_extremo.json", "w"), indent=1)
