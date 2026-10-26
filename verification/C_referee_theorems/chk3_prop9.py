"""Árbitro 3: Lema 2(iii)-(iv) y Prop. 9(i) (identidad X_q) a precisión de máquina; cota Phi_H <= 1/(4H||theta||^2)."""
import math, numpy as np
from arb_comun import sieve, factor
pr = sieve(10 ** 5)

def a_p(p, n, c):   # r_p(c) - 1 desde la DEFINICIÓN (nu_p)
    s = {0, (-c) % p, n % p, (n - c) % p}; nup = len({0, n % p})
    return (1 - len(s) / p) / (1 - nup / p) ** 2 - 1
def rho_p_minus1(p, n, c):
    if n % p == 0 or p == 2 or (p == 3 and n % 3):
        return a_p(p, n, c)
    return (p / (p - 2) if c % p == 0 else p * (p - 3) / ((p - 1) * (p - 2))) - 1
def gp(p, n): return 1 / (p - 1) if n % p == 0 else 2 / (p - 2)

worst = 0.0
for n in (1000, 2 * 7 * 11 * 13, 30030 + 2):
    for q in (3, 5, 7, 15, 21, 35, 105, 165, 2 * 3 * 5 * 7 * 11):
        ps = factor(q, pr)
        A = np.array([np.prod([a_p(p, n, c) for p in ps]) for c in range(q)])
        Ar = np.array([np.prod([rho_p_minus1(p, n, c) for p in ps]) for c in range(q)])
        al = np.fft.fft(A) / q; alr = np.fft.fft(Ar) / q
        cop = np.array([math.gcd(b, q) == 1 for b in range(q)])
        g = np.prod([gp(p, n) for p in ps]); phi = np.prod([p - 1 for p in ps])
        # Lema 2(iii): fórmula explícita de alpha_q(b) para (b,q)=1 (producto de alpha_p en coordenadas CRT)
        def alpha_formula(b):
            v = 1.0
            for p in ps:
                qp = q // p; bp = (b * pow(qp, -1, p)) % p   # b/q = sum bp/p  mod 1
                if n % p == 0: v *= 1 / (p - 1) ** 2
                else: v *= (2 / (p - 2) ** 2) * (1 + math.cos(2 * math.pi * bp * n / p))
            return v
        af = np.array([alpha_formula(b) if cop[b] else 0.0 for b in range(q)])
        e_formula = np.max(np.abs(al - af))
        e_rho = np.max(np.abs(alr[cop] - g / phi)) + np.max(np.abs(alr[~cop])) if (~cop).any() else np.max(np.abs(alr[cop] - g / phi))
        e_mean = abs(al[cop].real.mean() - g / phi)
        # Prop 9(i): X_q por h  vs  por Fourier
        H = 40; h = np.arange(-H + 1, H); w = 1 - np.abs(h) / H
        Xq_h = float(np.sum(w * (A[h % q] - Ar[h % q])))
        j = np.arange(H); Phi = np.array([abs(np.sum(np.exp(2j * np.pi * j * b / q))) ** 2 / H for b in range(q)])
        Xq_f = float(np.sum(((al.real - g / phi) * Phi)[cop]))
        worst = max(worst, e_formula, e_rho, e_mean, abs(Xq_h - Xq_f))
        print(f"n={n:6d} q={q:5d}: |alpha-fórmula|={e_formula:.1e} |alpha^rho - g/phi|={e_rho:.1e} |media-g/phi|={e_mean:.1e} min alpha={al.real.min():+.1e} | X_q(h)={Xq_h:+.6f} X_q(Fourier)={Xq_f:+.6f}")
print("peor discrepancia:", worst)
# cota Phi_H(theta) <= 1/(4 H ||theta||^2)
H = 97; th = np.linspace(1e-4, 0.5, 20001); j = np.arange(H)
Phi = np.abs(np.exp(2j * np.pi * np.outer(th, j)).sum(1)) ** 2 / H
print("max Phi_H * 4H theta^2 (debe ser <= 1):", float(np.max(Phi * 4 * H * th ** 2)))
