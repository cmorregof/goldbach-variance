"""[M]+[HEURISTIC] Comparación de T2^F (3|n) con la predicción de la cola de los hoyos de dimensión 1 en h=±n:
   T2^F(n,H) ≈ -Psi(H/n),  Psi(t) = 2 ∫_0^t (1 - s/t) ds/(1 - s^2)   (Psi(t) ~ t para t pequeño; Psi(1) = 2 log 2).
Datos: efecto_3n.json."""
import json, math, numpy as np
from scipy.integrate import quad
def Psi(t): return 2 * quad(lambda s: (1 - s / t) / (1 - s * s), 0, t, limit=200)[0]
d = json.load(open("efecto_3n.json"))
out = []
for row in d:
    n = row["n"]; pred = [-Psi(min(f, 1 - 1.0 / n)) if n % 3 == 0 else 0.0 for f in row["frac"]]
    res = [v - p for v, p in zip(row["T2f"], pred)]
    out.append(dict(n=n, frac=row["frac"], T2f=row["T2f"], pred=pred, resid=res))
    print(n, "3|n" if n % 3 == 0 else "3∤n", "resid:", " ".join("%+.3f" % x for x in res))
print("Psi:", {f: round(Psi(min(f, 0.999999)), 4) for f in d[0]["frac"]}, " 2log2=", round(2 * math.log(2), 4))
r3 = np.array([o["resid"] for o in out if o["n"] % 3 == 0]); r0 = np.array([o["resid"] for o in out if o["n"] % 3])
print("3|n: rms resid by t:", np.sqrt((r3 ** 2).mean(0)).round(3), " | 3∤n rms:", np.sqrt((r0 ** 2).mean(0)).round(3))
json.dump(out, open("psi_extremo.json", "w"), indent=1)
