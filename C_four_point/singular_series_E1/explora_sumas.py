"""Exploración [M]: S(H) = sum_{0<|h|<=H}(r-1) exacto, frente a -log^2 H/(2 S(n)), -G(H), y la parte rho.
Uso: python3 explora_sumas.py  -> explora_sumas.json"""
import json, time, numpy as np
from e1_comun import r_and_rho, g_array, sing, mellin_consts, primes_upto, EULER
t0 = time.time()
ns = [2**20, 2**20 + 2, 510510, 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 // 1, 1000002, 999998, 2 * 499979]
Hs = [2, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000]
pr = primes_upto(1 << 24)
out = []
for n in ns:
    Hmax = min(n - 1, max(Hs))
    r, rho = r_and_rho(n, n - 1, pr)
    S = sing(n, pr)
    G = g_array(n, n, pr); cG = np.cumsum(G)
    E0, b1 = mellin_consts(n, pr[pr < 10**7])
    cr = np.cumsum(r - 1.0); crho = np.cumsum(rho - 1.0)
    h = np.arange(1, n)
    for H in Hs + [n - 1]:
        if H > n - 1: continue
        sharp = 2 * cr[H - 1]; sharp_rho = 2 * crho[H - 1]
        w = np.clip(1 - h[:H] / H, 0, None)
        fej = 2 * np.dot(w, r[:H] - 1); fej_rho = 2 * np.dot(w, rho[:H] - 1)
        L = np.log(H)
        rec = dict(n=n, H=H, S=S, sharp=sharp, sharp_rho=sharp_rho, T2=sharp - sharp_rho,
                   fej=fej, fej_rho=fej_rho, T2f=fej - fej_rho,
                   G=float(cG[H] - 1.0), formula=-L * L / (2 * S),
                   mellin2=-(E0) * (0.5 * L * L + (2 * EULER + b1) * L), E0S=E0 * S, b1=b1)
        out.append(rec)
        print(f"n={n:8d} H={H:7d} sharp={sharp:+10.4f} rho={sharp_rho:+10.4f} T2={sharp-sharp_rho:+8.4f} | "
              f"fej={fej:+9.4f} T2f={fej-fej_rho:+8.4f} | -G={-rec['G']:+9.4f} formula={rec['formula']:+9.4f} "
              f"mellin={rec['mellin2']:+9.4f}  (sharp-formula)/L={(sharp-rec['formula'])/L:+.3f}", flush=True)
json.dump(out, open('explora_sumas.json', 'w'), indent=1)
print("t", round(time.time() - t0, 1))
