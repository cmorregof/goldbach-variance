"""[M] Tamaño típico de T2^F sobre n (datos de run6_E1/estadistica_T2.json, 40 n entre 1e5 y 3e6)
y efecto del extremo h≈±n en H=n-1 separado por 3|n."""
import json, numpy as np
d = json.load(open("/home/claude/ariadna/R3/run6_E1/estadistica_T2.json"))["rows"]
out = {}
for key in ["10", "100", "1000", "10000", "100000", "n/2", "n-1"]:
    v = np.array([x[key]["T2f"] for x in d if key in x]); n = np.array([x["n"] for x in d if key in x])
    H = np.array([x[key]["H"] for x in d if key in x])
    m3 = n % 3 == 0
    out[key] = dict(N=len(v), rms=float(np.sqrt(np.mean(v ** 2))), median_abs=float(np.median(np.abs(v))), max_abs=float(np.abs(v).max()),
                    mean_3div=float(v[m3].mean()) if m3.any() else None, mean_3ndiv=float(v[~m3].mean()), n3=int(m3.sum()),
                    rms_times_sqrtH=float(np.sqrt(np.mean(v ** 2)) * np.sqrt(np.median(H))))
    print(key, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[key].items()})
json.dump(out, open("rms_T2.json", "w"), indent=1)

# --- RMS después de restar la ley de extremo -1_{3|n} Psi(H/n) (ver psi_extremo.py)
from scipy.integrate import quad
def Psi(t): return 2 * quad(lambda s: (1 - s / t) / (1 - s * s), 0, min(t, 1 - 1e-9), limit=200)[0]
out2 = {}
for key in ["10", "100", "1000", "10000", "100000", "n/2", "n-1"]:
    rows = [x for x in d if key in x]
    v = np.array([x[key]["T2f"] + (Psi(x[key]["H"] / x["n"]) if x["n"] % 3 == 0 else 0.0) for x in rows])
    out2[key] = dict(rms_resid=float(np.sqrt(np.mean(v ** 2))), max_abs_resid=float(np.abs(v).max()))
    print("resid", key, {k: round(x, 4) for k, x in out2[key].items()})
json.dump(dict(raw=out, resid=out2), open("rms_T2.json", "w"), indent=1)
