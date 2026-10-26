"""arb_Gw_directo.py -- árbitro gen2 de R1-K1.  G_w(d) calculado DIRECTAMENTE de la forma de varianza del Lema dbasis(c):
   G_w(d) = (1/2) sum_{b mod d} sum_t ( U_b(t) - U_{-b}(-t) )^2,   U_b(t) = sum_{m = b (d)} lambda(m) w_{m-t},
con lambda(m) = Lambda(m) w(m/N), w_j = 0 fuera de [1,N], w_N = 0.  Sin FFT de productos de convoluciones: U_b(t) se obtiene
por una sola convolución por clase (numpy.convolve), y se eleva al cuadrado después.  Comparación con k1_Gw_bloques.json.
Uso: python3 -I arb_Gw_directo.py
"""
import json, math
import numpy as np

def Lambda_upto(N):
    isp = np.ones(N + 1, bool); isp[:2] = False
    for p in range(2, int(N ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    Lam = np.zeros(N + 1)
    for p in np.nonzero(isp)[0]:
        q = int(p)
        while q <= N: Lam[q] = math.log(int(p)); q *= int(p)
    return Lam

def Gw_direct(N, d, wfun):
    m = np.arange(N + 1)
    w = np.zeros(N + 1); w[1:N] = wfun(m[1:N] / N)          # w_0 = w_N = 0
    lam = Lambda_upto(N) * w
    # U_b(t) = sum_m lam(m) 1_{m=b} w_{m-t}, t in [-(N-1), N-1]:  correlación de lam_b con w
    # np.correlate(a, v, 'full')[k] = sum_n a[n+k-(len(v)-1)] v[n] ...; más simple: U_b(t) = sum_j lam_b(t+j) w_j  (j = m - t)
    # => U_b = correlate(lam_b, w, 'full'), con índice t = k - N
    G = 0.0
    U = {}
    for b in range(d):
        lb = np.where(m % d == b, lam, 0.0)
        U[b] = np.correlate(lb, w, 'full')     # longitud 2N+1, índice k <-> t = k - N
    for b in range(d):
        Ub = U[b]; Umb = U[(-b) % d][::-1]      # U_{-b}(-t): invertir el índice t -> -t
        G += 0.5 * float(np.sum((Ub - Umb) ** 2))
    return G

def main():
    J = json.load(open('/home/claude/ariadna/R1/run1_K1/k1_Gw_bloques.json'))
    weights = {'1-x': lambda x: 1 - x, '(1-x)^2': lambda x: (1 - x) ** 2}
    N = 2 ** 14
    for name, wf in weights.items():
        rows = {r['d']: r for r in J['weights'][name]['per_N']['14']['rows']}
        for d in (6, 10, 30):
            G = Gw_direct(N, d, wf) / N ** 2
            print(f"{name:8s} N=2^14 d={d:3d}: G/N^2 directo = {G:.9f}   informe = {rows[d]['G']:.9f}   dif = {G - rows[d]['G']:.2e}")

if __name__ == '__main__':
    main()
