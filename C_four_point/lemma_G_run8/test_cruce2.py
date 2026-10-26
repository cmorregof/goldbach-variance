"""[M] Cruce de g2_comun.r_rho con run6_E1/e1_comun.r_and_rho (código del run 6, ya cruzado allí con r3_perfil_h)."""
import sys, numpy as np, json
sys.path.insert(0, "/home/claude/ariadna/R3/run6_E1")
from e1_comun import r_and_rho, primes_upto as pu6
from g2_comun import r_rho
out = []
for n in (1000, 2310, 4096, 6006, 7778, 30030, 65536):
    r6, q6 = r_and_rho(n, n - 1, pu6(1 << 22))
    r8, q8 = r_rho(n)
    m = r6 > 0
    er = float(np.max(np.abs(r8[m] / r6[m] - 1))); eq = float(np.max(np.abs(q8[m] / q6[m] - 1)))
    z = bool(np.all((r6 == 0) == (r8 == 0)))
    out.append(dict(n=n, max_rel_err_r=er, max_rel_err_rho=eq, same_zeros=z))
    print(out[-1])
json.dump(out, open("test_cruce2.json", "w"), indent=1)
