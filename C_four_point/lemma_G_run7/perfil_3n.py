"""[M] Perfil acumulado de T2^F en h (H = n-1 y H = n/2) para n con 3|n y 3∤n: ¿dónde se acumula el déficit?"""
import json, numpy as np
from g_comun import r_and_rho, primes_upto, fejer_w
pr = primes_upto(1 << 24)
out = []
for n in (1225206, 1625094, 1897212, 1021020, 1055530, 1285066, 1578292, 1944904):
    r, rho = r_and_rho(n, n - 1, pr)
    for H in (n - 1, n // 2, n // 8):
        w = fejer_w(H); c = np.cumsum(w * (r[:H] - rho[:H]))
        fr = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
        prof = [float(c[max(0, int(f * H) - 1)]) for f in fr]
        out.append(dict(n=n, n_mod3=n % 3, H=H, frac=fr, cum=prof))
        print(n, "3|n" if n % 3 == 0 else "3∤n", "H/n=%.3f" % (H / n), " ".join("%+.3f" % x for x in prof), flush=True)
json.dump(out, open("perfil_3n.json", "w"), indent=1)
