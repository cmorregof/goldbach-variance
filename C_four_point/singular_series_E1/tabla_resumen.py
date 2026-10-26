"""Tabla [M] para E1_prueba.md §9: exacto frente a la fórmula objetivo F0 = -log^2 H/(2S) y frente a las
predicciones con término secundario (Teorema 5): Fejér coef c_F = 2g-1+log(4pi)+Sigma, agudo c_A = 2g+2+log2+Sigma,
Sigma = sum_{p|n,p>2} log p/(p-1)."""
import json, numpy as np
from e1_comun import EULER, primes_upto, prime_factors
pr = primes_upto(1 << 16)
rows = json.load(open("verifica_T1.json"))
lines = ["| n | 𝔖(n) | H | exacto agudo | exacto Fejér | F0 | pred. agudo | pred. Fejér | agudo/F0 | (agudo−pred)/log H | (Fejér−pred)/log H |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for x in rows:
    n, H, S = x["n"], x["H"], x["S"]
    if n not in (1048576, 999958, 1000002, 9699690) or H < 1000: continue
    Sig = sum(np.log(p) / (p - 1) for p in prime_factors(n, pr) if p > 2)
    LH = np.log(H)
    cF = 2 * EULER - 1 + np.log(4 * np.pi) + Sig; cA = 2 * EULER + 2 + np.log(2) + Sig
    F0 = -LH ** 2 / (2 * S); pA = -(0.5 * LH ** 2 + cA * LH) / S; pF = -(0.5 * LH ** 2 + cF * LH) / S
    lines.append(f"| {n} | {S:.3f} | {H} | {x['sharp']:.2f} | {x['fej']:.2f} | {F0:.2f} | {pA:.2f} | {pF:.2f} | "
                 f"{x['sharp']/F0:.2f} | {(x['sharp']-pA)/LH:+.2f} | {(x['fej']-pF)/LH:+.3f} |")
open("tabla_resumen.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
