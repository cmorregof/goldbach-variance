"""[M] Comprobaciones de las identidades EXACTAS de E1_prueba.md (deben cuadrar a precisión de máquina):
 (I1) Fourier: alpha_q(b) >= 0, alpha_q(b) = 0 si (b,q)>1, promedio sobre (b,q)=1 = g(q)/phi(q) (Lema 2).
 (I2) Identidad de Fejér para la parte rho (Prop. 4):
      sum_{0<|h|<H} (1-|h|/H)(rho-1) = -H Gamma sum_{d>=X} kappa/d - l Gamma K(X) + Theta + 1 - f(H)/H
 (I3) Identidad aguda (Prop. 4):  sum_{0<|h|<=H}(rho-1) = -2H Gamma sum_{d>X} kappa/d - l Gamma K+(X) - 2 l Gamma W(X)
 Las colas sum_{d>=X} kappa(d)/d se obtienen como 1/Gamma - sum_{d<X} kappa(d)/d (identidad de la media, Prop. 4(ii))."""
import numpy as np, math, json
from e1_comun import r_and_rho, primes_upto, prime_factors
pr = primes_upto(1 << 24)
out = {}
# ---- (I1)
def a_p(p, n, c):
    if n % p == 0:
        u = p / (p - 1) ** 2; return u * ((c % p == 0) - 1 / p)
    u = 2 * p / (p - 2) ** 2; v = p / (p - 2) ** 2
    return u * ((c % p == 0) - 1 / p) + v * (((c - n) % p == 0) + ((c + n) % p == 0) - 2 / p)
def g_p(p, n): return 1 / (p - 1) if n % p == 0 else 2 / (p - 2)
res1 = []
for n in (1000, 3000, 2 * 7 * 11 * 13):
    for q in (3, 5, 7, 15, 21, 35, 77, 105, 143, 3 * 5 * 7 * 11, 2 * 3 * 5 * 7):
        ps = prime_factors(q, pr)
        A = np.array([np.prod([a_p(p, n, c) for p in ps]) for c in range(q)])
        alpha = np.fft.fft(A) / q          # alpha(b) = (1/q) sum_c A(c) e(-bc/q)
        cop = np.array([math.gcd(b, q) == 1 for b in range(q)])
        gq = np.prod([g_p(p, n) for p in ps]); phiq = np.prod([p - 1 for p in ps])
        res1.append(dict(n=n, q=q, min_alpha=float(alpha.real.min()), max_imag=float(np.abs(alpha.imag).max()),
                         max_noncoprime=float(np.abs(alpha[~cop]).max()) if (~cop).any() else 0.0,
                         mean_coprime=float(alpha[cop].real.mean()), g_over_phi=float(gq / phiq), A0_minus_g=float(A[0] - gq)))
print("(I1) min alpha:", min(x['min_alpha'] for x in res1), " max |Im|:", max(x['max_imag'] for x in res1),
      " max |alpha| noncoprime:", max(x['max_noncoprime'] for x in res1),
      " max |mean-g/phi|:", max(abs(x['mean_coprime'] - x['g_over_phi']) for x in res1),
      " max |A_q(0)-g(q)|:", max(abs(x['A0_minus_g']) for x in res1))
out["I1"] = res1
# ---- (I2), (I3)
def setup(n, X):
    pf = set(prime_factors(n, pr)); l = 2 if 3 in pf else 6
    lg = 0.0
    for p in pr[pr >= 3]:
        p = int(p)
        if p == 3 and 3 not in pf: continue
        lg += math.log(p * (p - 2) / (p - 1) ** 2) if p in pf else math.log(p * (p - 3) / ((p - 1) * (p - 2)))
    Gam = math.exp(lg - 2.0 / (pr[-1] * math.log(pr[-1])))
    k = np.ones(X + 2); k[0] = 0
    for p in pr[pr <= X + 1]:
        p = int(p)
        if p == 2 or (p == 3 and 3 not in pf): k[p::p] = 0; continue
        k[p::p] *= 1 / (p - 2) if p in pf else 2 / (p - 3)
        if p * p <= X + 1: k[p * p::p * p] = 0
    return pf, l, Gam, k
res2 = []
for n, H in ((1048576, 1000), (1048576, 99999), (1000002, 5000), (9699690, 30000), (999958, 777)):
    r, rho = r_and_rho(n, H, pr)
    pf, l, Gam, k = setup(n, H // 2 + 2)
    X = H / l
    h = np.arange(1, H + 1)
    lhsF = 2 * np.dot(1 - h[:H - 1] / H, rho[:H - 1] - 1)
    d = np.arange(1, len(k)); kd = k[1:]
    lt = d < X; le = d <= X
    tailF = 1 / Gam - np.sum(kd[lt] / d[lt])
    fX = lambda u: (u - np.floor(u)) * (1 - (u - np.floor(u)))
    Theta = (l * Gam / X) * np.sum(kd[lt] * d[lt] * fX(X / d[lt]))
    rhsF = -H * Gam * tailF - l * Gam * np.sum(kd[lt]) + Theta + 1 - fX(H) / H
    lhsS = 2 * np.sum(rho[:H] - 1)
    tailS = 1 / Gam - np.sum(kd[le] / d[le])
    y = X / d[le]; P = y - np.floor(y) - 0.5
    rhsS = -2 * H * Gam * tailS - l * Gam * np.sum(kd[le]) - 2 * l * Gam * np.sum(kd[le] * P)
    res2.append(dict(n=n, H=H, fejer_lhs=float(lhsF), fejer_rhs=float(rhsF), sharp_lhs=float(lhsS), sharp_rhs=float(rhsS)))
    print(f"(I2/I3) n={n} H={H}: Fejér {lhsF:+.9f} vs {rhsF:+.9f} | agudo {lhsS:+.9f} vs {rhsS:+.9f}")
out["I2_I3"] = res2
json.dump(out, open("verifica_identidades.json", "w"), indent=1)
