"""Árbitro: Lema F.  A_e(N) := sum_{n par in (N,2N)} w(n/N) S_e(n; n-1),  S_e = sum_{0<|h|<n-1} phi_{n-1}(h) rho(n,h) T_e(n,h).
Afirmación: |A_e| <= C N' gamma(e) min(1, e/N)^sigma para todo 0<=sigma<1 (N' = sum_n w).
Pruebo: (a) e primo con e/N fijo en 3 bloques N (constancia del cociente A_e/(N' gamma));
        (b) e pequeño: cociente A_e/(N' gamma(e) e/N) (sigma=1, el borde) y A_e/(N' gamma(e) (e/N)^0.5);
        (c) e compuesto grande.
Uso: python3 lemaF_check.py K1 K2 ..."""
import sys, math, json, numpy as np
import arb_comun as A
Ks = [int(x) for x in sys.argv[1:]] or [9, 10, 11]
P = [int(p) for p in A.primes_upto(1 << 20) if p >= 5]
def nextp(x): return min(p for p in P if p >= x)
out = {}
for K in Ks:
    N = 1 << K
    E = [[5], [7], [11], [13], [5, 7], [5, 7, 11], [nextp(N // 64)], [nextp(N // 16)], [nextp(N // 4)], [nextp(N // 2)], [nextp(N)], [nextp(2 * N)], [nextp(3 * N)], [nextp(4 * N)], [nextp(8 * N)], [nextp(16 * N)]]
    q = nextp(int(math.sqrt(2 * N))); E.append(sorted({q, nextp(2 * N // q + 1)}))
    E.append(sorted({5, nextp(2 * N // 5 + 1)})); E.append(sorted({5, 7, nextp(2 * N // 35 + 1)}))
    acc = np.zeros(len(E)); W = 0.0; pmax = np.zeros(len(E))
    for n in range(N + 2, 2 * N, 2):
        w = float(A.w_smooth(np.array([n / N]))[0])
        if w == 0: continue
        r, rho = A.r_rho(n); H = n - 1; h = np.arange(1, H + 1)
        base = 2.0 * (1 - h / H) * rho
        W += w
        for i, e in enumerate(E):
            s = float(np.dot(base, A.T_e(e, n, h))); acc[i] += w * s
            pmax[i] = max(pmax[i], abs(s) / (A.gamma_e(e) * math.log(n) ** 2))
    rows = []
    for i, e in enumerate(E):
        ev = int(np.prod(e)); g = A.gamma_e(e); x = ev / N
        rows.append(dict(e=ev, primes=e, e_over_N=x, ratio0=acc[i] / (W * g), ratio_half=acc[i] / (W * g * min(1, x) ** 0.5), ratio1=acc[i] / (W * g * min(1, x)), pmax=pmax[i]))
        print("K=%d e=%8d %-16s e/N=%7.3f  A/(N'g)=%+.3e  /min^.5=%+.3e  /min^1=%+.3e  puntual/(g log^2 n) max=%.3f" % (K, ev, str(e), x, rows[-1]['ratio0'], rows[-1]['ratio_half'], rows[-1]['ratio1'], pmax[i]))
    out[K] = dict(N=N, Nprime=W, rows=rows)
json.dump(out, open("lemaF_check.json", "w"), indent=1)
