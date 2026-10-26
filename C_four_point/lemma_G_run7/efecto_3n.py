"""[M] El término sistemático de T2^F cuando 3|n y H ≍ n: ¿lineal en H/n?, ¿depende de log n?"""
import json, numpy as np
from g_comun import r_and_rho, primes_upto, fejer_w
pr = primes_upto(1 << 24)
fr = [1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 3/4, 1.0]
out = []
for n in (30006, 300006, 1021020, 1225206, 2999994, 30008, 300008, 2999996):
    r, rho = r_and_rho(n, n - 1, pr)
    vals = []
    for f in fr:
        H = max(2, min(n - 1, int(round(f * n))))
        vals.append(float(np.dot(fejer_w(H), r[:H] - rho[:H])))
    out.append(dict(n=n, n_mod3=n % 3, frac=fr, T2f=vals))
    print(n, "3|n" if n % 3 == 0 else "3∤n", " ".join("%+.3f" % v for v in vals), " | T2/(H/n):", " ".join("%+.2f" % (v / f) for v, f in zip(vals, fr)), flush=True)
json.dump(out, open("efecto_3n.json", "w"), indent=1)
