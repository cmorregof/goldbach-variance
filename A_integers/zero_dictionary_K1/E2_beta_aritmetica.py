"""R1-E2, corrida 4: comprobaciones aritmeticas de los pesos beta(d) usados en la prueba.
No usa primos de Goldbach ni ceros: solo la definicion de beta (Lema dbasis(a)).
beta(d) = 4*C2*prod_{p | d/2} 1/(p-2) para d par libre de cuadrados, 0 si no.
Comprueba: (1) sumas diadicas acotadas (-> 2 log 2); (2) sum_{d<=y} beta = 2 log y + O(1);
(3) sum_{d<=y} beta log^2 d  ~ (2/3) log^3 y; (4) colas sum_{d>Q} beta*(log d/phi(d)) y
sum_{d>Q} beta d^{-1/2} log^2 d frente a Q^{-1/2} log^2 Q; (5) conteo de bloques y sum_D log D.
"""
import json, math, sys
import numpy as np

X = int(sys.argv[1]) if len(sys.argv) > 1 else 4_000_000
C2 = 0.6601618158468696

prod = np.ones(X + 1)
sqf = np.ones(X + 1, dtype=bool)
phi = np.arange(X + 1, dtype=np.int64)
isp = np.ones(X + 1, dtype=bool); isp[:2] = False
for p in range(2, int(X**0.5) + 1):
    if isp[p]:
        isp[p*p::p] = False
primes = np.nonzero(isp)[0]
for p in primes:
    p = int(p)
    phi[p::p] -= phi[p::p] // p
    if p > 2:
        prod[p::p] /= (p - 2)
    if p * p <= X:
        sqf[p*p::p*p] = False
d = np.arange(X + 1)
beta = np.where((d % 2 == 0) & sqf & (d > 0), 4 * C2 * prod, 0.0)
beta[0] = 0.0

out = {"X": X}
# (1) bloques diadicos (D,2D], D=2^k
blocks = []
k = 2
while 2 ** (k + 1) <= X:
    D = 2 ** k
    blocks.append([D, float(beta[D+1:2*D+1].sum())])
    k += 1
out["bloques_diadicos"] = blocks
out["max_bloque_D>=8"] = max(b for D, b in blocks if D >= 8)
out["2log2"] = 2 * math.log(2)
# (2),(3)
cs = np.cumsum(beta)
logd = np.log(np.maximum(d, 1))
cs2 = np.cumsum(beta * logd**2)
rows = []
for y in [10**3, 10**4, 10**5, 10**6, X]:
    rows.append({"y": y, "sum_beta-2logy": float(cs[y] - 2*math.log(y)),
                 "sum_beta_log2/((2/3)log^3y)": float(cs2[y] / ((2/3)*math.log(y)**3))})
out["sumas_acumuladas"] = rows
# (4) colas del resto R_w: beta*log d/phi(d) y beta*d^{-1/2}log^2 d
with np.errstate(divide="ignore", invalid="ignore"):
    t1 = np.where(d > 2, beta * logd / np.maximum(phi, 1), 0.0)
    t2 = np.where(d > 2, beta * logd**2 / np.sqrt(np.maximum(d, 1)), 0.0)
tails = []
for Q in [10, 30, 100, 300, 1000, 3000]:
    s1 = float(t1[Q+1:].sum()); s2 = float(t2[Q+1:].sum())
    # cola analitica de t2 mas alla de X: int_X^inf 2 y^{-3/2} log^2 y dy aprox 4 X^{-1/2} log^2 X
    s2 += 4 * X**-0.5 * math.log(X)**2
    tails.append({"Q": Q, "sum_{d>Q} beta logd/phi": s1,
                  "sum_{d>Q} beta d^-1/2 log^2 d (+cola)": s2,
                  "Q^-1/2 log^2 Q": Q**-0.5 * math.log(Q)**2})
out["colas_resto"] = tails
# (5) conteo de bloques y sum_D log D sobre D=2^k en [Q0, Q1'/2], frente a (1/2+eps)^2 L^2/(2 log 2)
cnt = []
for L in [50.0, 200.0, 1000.0, 10000.0]:
    for eps in [0.05, 0.25]:
        A = 1
        lQ0 = A * math.log(L); lQ1 = (0.5 + eps) * L
        ks = [k for k in range(0, int(lQ1 / math.log(2)) + 2)
              if k*math.log(2) >= lQ0 and (k+1)*math.log(2) <= lQ1]
        sLog = sum(k * math.log(2) for k in ks)
        cnt.append({"L": L, "eps": eps, "num_bloques": len(ks),
                    "cota_bloques (1/2+eps)L/log2+1": lQ1/math.log(2) + 1,
                    "sum_logD/L^2": sLog / L**2,
                    "(1/2+eps)^2/(2log2)": (0.5+eps)**2 / (2*math.log(2))})
out["conteo_bloques"] = cnt
json.dump(out, open("/home/claude/ariadna/R1/run4_E2/E2_beta_aritmetica.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "bloques_diadicos"}, indent=1))
print("bloques diadicos (D, suma):", [(D, round(b, 4)) for D, b in blocks[-6:]])
