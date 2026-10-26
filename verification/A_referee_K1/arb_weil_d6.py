"""arb_weil_d6.py -- árbitro gen2 de R1-K1.  Comprobación INDEPENDIENTE con ceros reales, módulo d = 6, peso w = 1-x.

Los dos caracteres mod 6 son chi_0 (inducido por 1: ceros de zeta, q=1, par) y el impar (inducido por chi_{-3}, q=3).
  (a) Fórmula de Weil (2.1) del informe K1, en u = 0.3 y 0.7, para zeta y chi_{-3} (instancia distinta de la del informe,
      que usó chi_{-4}).  Implementación propia de cada término.
  (b) D_w(6) directamente de los ceros: (1/2) sum_chi int_0^1 sum_rho |c~_{u,chi}(rho)|^2 du, con cola por la densidad,
      frente al valor D = 0.006941 de k1_Gw_bloques.json y frente a la fórmula (2.4)-(2.5) reconstruida aquí.
  (c) La parte fuera de la diagonal O_w^T(N;6) para N = 2^14, 2^15, 2^16, frente a G_w(6)/N^2 - D_w(6) medido.
Uso: python3 -I arb_weil_d6.py   (unos minutos; T = 300)
"""
import json, math, time
import numpy as np
import mpmath as mp
from scipy.special import psi as digamma
from scipy.integrate import quad

OUT = "/home/claude/ariadna/R1/gen2_arbitro_K1/arb_weil_d6.json"
EUL = 0.57721566490153286061
mp.mp.dps = 15
T = 300.0

# ------------------------------------------------------------ Mellin cerradas, w = 1-x (derivadas a mano: ver informe del árbitro)
def mel_a(s, u):   # phi_u(x) = (1-x)(1+u-x) en [u,1] = (1+u) - (2+u)x + x^2
    return (1 + u) * (1 - u ** s) / s - (2 + u) * (1 - u ** (s + 1)) / (s + 1) + (1 - u ** (s + 2)) / (s + 2)

def mel_b(s, u):   # phi_{-u}(x) = (1-x)(1-u-x) en [0,1-u] = (1-u) - (2-u)x + x^2
    v = 1 - u
    return v ** (s + 1) / s - (2 - u) * v ** (s + 1) / (s + 1) + v ** (s + 2) / (s + 2)

def mel_c(s, u, sig):   # sig = chi(-1):  c = phi_u - sig phi_{-u}
    return mel_a(s, u) - sig * mel_b(s, u)

def c_x(x, u, sig):
    x = np.asarray(x, float)
    a = np.where((x >= u) & (x <= 1), (1 - x) * (1 + u - x), 0.0)
    b = np.where((x >= 0) & (x <= 1 - u), (1 - x) * (1 - u - x), 0.0)
    return a - sig * b

def r_(u): return (1 - u) ** 2 * (2 + u) / 6

# ------------------------------------------------------------ ceros
def zeta_zeros(T):
    zs = []; n = 1
    while True:
        g = float(mp.zetazero(n).imag)
        if g > T: break
        zs.append(g); n += 1
    return np.array(zs)

def Lchi3(s):   # L(s, chi_{-3}) = 3^{-s} [zeta(s,1/3) - zeta(s,2/3)]
    return mp.power(3, -s) * (mp.zeta(s, mp.mpf(1) / 3) - mp.zeta(s, mp.mpf(2) / 3))

def Z3(t):   # función Z real para chi_{-3} (impar, q=3, raíz unidad epsilon = 1)
    s = mp.mpc(0.5, t)
    th = mp.im(mp.loggamma((s + 1) / 2)) + t / 2 * mp.log(3 / mp.pi)
    v = mp.exp(1j * th) * Lchi3(s)
    return float(mp.re(v)), float(mp.im(v))

def chi3_zeros(T, h):
    ts = np.arange(h, T + h, h)
    vals = np.array([Z3(float(t))[0] for t in ts])
    zs = []
    for i in range(len(ts) - 1):
        if vals[i] * vals[i + 1] < 0:
            lo, hi, flo = float(ts[i]), float(ts[i + 1]), vals[i]
            for _ in range(45):
                mid = 0.5 * (lo + hi); fm = Z3(mid)[0]
                if flo * fm <= 0: hi = mid
                else: lo, flo = mid, fm
                if hi - lo < 1e-12: break
            zs.append(0.5 * (lo + hi))
    return np.array(zs), ts, vals

# ------------------------------------------------------------ cuadratura en t (densidad)
XG12, WG12 = np.polynomial.legendre.leggauss(12)
def tpanels(t0, t1, width):
    n = int(math.ceil((t1 - t0) / width)); e = np.linspace(t0, t1, n + 1)
    a, b = e[:-1], e[1:]
    t = (0.5 * (b - a)[:, None] * XG12 + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * WG12).ravel()
    return t, w

def density_integral(u, sig, q, t0, t1):
    """(1/2pi) int_{t0<|t|<t1} |c~(1/2+it)|^2 [log(q/pi) + Re psi((1/2+a+it)/2)] dt, a = (1-sig)/2."""
    a = 0 if sig == 1 else 1
    t, w = tpanels(t0, t1, 0.5)
    s = 0.5 + 1j * t
    c2 = np.abs(mel_c(s, u, sig)) ** 2
    dens = math.log(q / math.pi) + np.real(digamma((s + a) / 2))
    return 2 * np.sum(w * c2 * dens) / (2 * math.pi)

def density_tail_asym(u, sig, q, t1):
    # |c~|^2 ~ (1-u)^2 u / t^2 ; Re psi((1/2+a+it)/2) ~ log(t/2):   (1/pi) int_{t1}^inf C/t^2 [log(q/pi) + log(t/2)] dt
    C = (1 - u) ** 2 * u
    return C / (math.pi * t1) * (math.log(q / math.pi) + math.log(t1 / 2) + 1)

# ------------------------------------------------------------ términos aritméticos de Weil
def Hcheck(n, u, sig):
    c0 = float(c_x(0.0, u, sig)); top = 1.0 / n
    pts = sorted(set([0.0, top] + [p for p in (u, 1 - u, u / n, (1 - u) / n) if 0 < p < top]))
    return sum(quad(lambda y: float((c_x(y, u, sig) - c0) * c_x(n * y, u, sig)), pts[k], pts[k + 1], limit=100)[0]
               for k in range(len(pts) - 1))

def prime_powers(X):
    isp = np.ones(X + 1, bool); isp[:2] = False
    for p in range(2, int(X ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    out = []
    for p in np.nonzero(isp)[0]:
        q = int(p)
        while q <= X: out.append((q, math.log(int(p)))); q *= int(p)
    out.sort(); return out

def weil_rhs(u, sig, q, chi, PP, Xh, LpL1):
    """lado derecho de (2.1) con truncación simétrica |t| < T; devuelve dict de piezas."""
    c0 = float(c_x(0.0, u, sig)); c1 = r_(u) * (1 - sig)
    dens = density_integral(u, sig, q, 0.0, T)
    out = dict(density=dens)
    if q == 1:   # zeta: 2 H(1), H(1) = -c0 int c log x
        brk = sorted(set([0.0, u, 1 - u, 1.0]))
        clog = sum(quad(lambda x: float(c_x(x, u, sig)) * math.log(x), brk[k], brk[k + 1], limit=200)[0] for k in range(len(brk) - 1))
        out['H1term'] = -2 * c0 * clog
    else:
        out['H1term'] = 0.0
    out['odd_residue'] = 2 * c0 * c1 * LpL1 if sig == -1 else 0.0
    S = 0.0
    for n, lg in PP:
        if n > Xh: break
        cn = chi(n)
        if cn == 0: continue
        S += lg * cn * Hcheck(n, u, sig)
    # cola n > Xh: Hcheck(n) ~ K/n^2 con K = int_0^1 z c'(0+) c(z) dz; para chi_{-3} la suma alterna y la cola es despreciable
    h = 1e-7; cp0 = (float(c_x(h, u, sig)) - c0) / h
    K = quad(lambda z: z * cp0 * float(c_x(z, u, sig)), 0, 1, points=[u, 1 - u], limit=200)[0]
    tail = K / Xh if q == 1 else 0.0
    out['prime_sum'] = -2 * (S + tail)
    out['total'] = out['density'] + out['H1term'] + out['odd_residue'] + out['prime_sum']
    return out

# ------------------------------------------------------------ main
def main():
    t0 = time.time(); res = {}
    zz = zeta_zeros(T)
    Nz_exp = T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)) + 7 / 8
    print(f"zeta: {len(zz)} ceros 0<gamma<{T} (esperado {Nz_exp:.1f}); ({time.time()-t0:.0f}s)", flush=True)
    z3, ts, vals = chi3_zeros(T, 0.1)
    N3_exp = T / (2 * math.pi) * math.log(3 * T / (2 * math.pi * math.e))
    print(f"chi_-3: {len(z3)} ceros 0<gamma<{T} (esperado {N3_exp:.1f}); ({time.time()-t0:.0f}s)", flush=True)
    if abs(len(z3) - N3_exp) > 1.5:
        print("  -> recuento sospechoso; rebarrido con paso 0.02", flush=True)
        z3, ts, vals = chi3_zeros(T, 0.02)
        print(f"chi_-3: {len(z3)} ceros tras rebarrido ({time.time()-t0:.0f}s)", flush=True)
    res['zeros'] = dict(T=T, n_zeta=len(zz), n_zeta_expected=Nz_exp, n_chi3=len(z3), n_chi3_expected=N3_exp,
                        first_chi3=[float(v) for v in z3[:5]], first_zeta=[float(v) for v in zz[:3]])

    # L'/L(1, chi_{-3}) directo con mpmath
    Ld = lambda s: mp.dirichlet(s, [0, 1, -1])      # en s=1 la diferencia de Hurwitz da nan; mp.dirichlet no
    LpL3 = float(mp.re(mp.diff(Ld, 1) / Ld(1)))
    L1_3 = float(mp.re(Ld(1)))
    print(f"L(1,chi_-3) = {L1_3:.12f} (pi/(3 sqrt3) = {math.pi/(3*math.sqrt(3)):.12f});  L'/L(1,chi_-3) = {LpL3:.12f}", flush=True)
    res['LpL_chi3'] = LpL3

    # (a) Weil por carácter
    PP = prime_powers(200000)
    chi3 = lambda n: 0 if n % 3 == 0 else (1 if n % 3 == 1 else -1)
    one = lambda n: 1
    res['weil'] = []
    for u in (0.3, 0.7):
        for name, zs, sig, q, chi in (('zeta', zz, 1, 1, one), ('chi_-3', z3, -1, 3, chi3)):
            lhs = 2 * float(np.sum(np.abs(mel_c(0.5 + 1j * zs, u, sig)) ** 2))
            rhs = weil_rhs(u, sig, q, chi, PP, 200000, LpL3)
            row = dict(u=u, L=name, lhs=lhs, **rhs, diff=lhs - rhs['total'], diff_sin_extras=lhs - rhs['density'],
                       escala_trunc=(1 - u) ** 2 * u / T ** 2 * math.log(T))
            res['weil'].append(row)
            print("[Weil]", json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    # (b) D_w(6) directamente de los ceros, integrando en u.  u = e^{-s}, s in [0, 12], paneles GL.
    su, wu = tpanels(0.0, 12.0, 0.25); uu = np.exp(-su); wu = wu * uu     # du = u ds
    Dz = np.zeros(len(uu)); D3 = np.zeros(len(uu)); tailz = np.zeros(len(uu)); tail3 = np.zeros(len(uu))
    for i, u in enumerate(uu):
        Dz[i] = 2 * float(np.sum(np.abs(mel_c(0.5 + 1j * zz, u, 1)) ** 2))
        D3[i] = 2 * float(np.sum(np.abs(mel_c(0.5 + 1j * z3, u, -1)) ** 2))
        tailz[i] = density_integral(u, 1, 1, T, 20000.0) + density_tail_asym(u, 1, 1, 20000.0)
        tail3[i] = density_integral(u, -1, 3, T, 20000.0) + density_tail_asym(u, -1, 3, 20000.0)
    D_zeros = 0.5 * float(np.sum(wu * (Dz + D3)))
    D_tail = 0.5 * float(np.sum(wu * (tailz + tail3)))
    print(f"D_w(6) de ceros |gamma|<T: {D_zeros:.7f}; cola por densidad: {D_tail:.7f}; total {D_zeros+D_tail:.7f}  ({time.time()-t0:.0f}s)", flush=True)

    # fórmula (2.4)-(2.5) reconstruida (constantes exactas de arb_constantes.py)
    r0 = 1 / 3; cB = 301 / 540 + (EUL + math.log(2 * math.pi) - 8 * math.log(2)) / 9
    X2 = 17 / 288; A_odd = 11 / 30; H1 = 13 / 72
    GFG = r0 ** 2 * (math.log(6) - math.log(2) - math.log(3) / 2) - cB
    par = X2 * math.log(3) / 2
    Rodd = A_odd * 0.5 * LpL3
    Rprin = H1 / 2
    # Pr(6) = -(1/2) sum_chi int_0^1 2 sum_n Lambda(n) Re chi*(n) Hcheck_{u,chi}(n) du = -sum_n Lambda(n)[Hc^e(n) + chi_-3(n) Hc^o(n)]
    su2, wu2 = tpanels(0.0, 12.0, 0.5); uu2 = np.exp(-su2); wu2 = wu2 * uu2   # malla más gruesa para Pr (suave en u)
    def Hc_u(n, sig):
        # int_0^1 Hcheck(n,u,sig) du, con u = e^{-s}
        return float(np.sum(wu2 * np.array([Hcheck(n, u, sig) for u in uu2])))
    Pr = 0.0
    for n, lg in PP:
        if n > 400: break
        Pr -= lg * (Hc_u(n, 1) + chi3(n) * Hc_u(n, -1))
    # cola n>400: Hc^e(n) ~ Ke/n^2 ; Ke = int_0^1 du K(u)
    def Kfun(u, sig):
        c0 = float(c_x(0.0, u, sig)); h = 1e-7; cp0 = (float(c_x(h, u, sig)) - c0) / h
        return quad(lambda z: z * cp0 * float(c_x(z, u, sig)), 0, 1, points=[u, 1 - u], limit=200)[0]
    Ke = float(np.sum(wu2 * np.array([Kfun(u, 1) for u in uu2])))
    Pr_tail = -sum(lg * Ke / n ** 2 for n, lg in PP if n > 400)   # la parte chi_-3 alterna y es despreciable
    Pr += Pr_tail
    D_formula = GFG + par + Rodd + Rprin + Pr
    print(f"formula: GFG={GFG:.6f} par={par:.6f} Rodd={Rodd:.6f} Rprin={Rprin:.6f} Pr={Pr:.6f} (cola {Pr_tail:.2e})  D={D_formula:.7f}", flush=True)
    print(f"informe K1 (json): D=0.006941  par=0.032424 Rodd=0.067512 Rprin=0.090278 Pr=-0.034686", flush=True)
    res['D6'] = dict(D_zeros_T=D_zeros, D_tail=D_tail, D_zeros_total=D_zeros + D_tail, D_formula=D_formula,
                     GFG=GFG, par=par, Rodd=Rodd, Rprin=Rprin, Pr=Pr, Ke=Ke)

    # (c) fuera de la diagonal O^T(N;6) = (1/2) sum_chi int_0^1 [ |Z^T(u)|^2 - sum |c~|^2 ] du
    res['offdiag'] = {}
    J = json.load(open('/home/claude/ariadna/R1/run1_K1/k1_Gw_bloques.json'))
    for lg in (14, 15, 16):
        N = 2 ** lg
        Oz = np.zeros(len(uu)); O3 = np.zeros(len(uu))
        for i, u in enumerate(uu):
            for zs, sig, Oarr in ((zz, 1, Oz), (z3, -1, O3)):
                cv = mel_c(0.5 + 1j * zs, u, sig); cvm = mel_c(0.5 - 1j * zs, u, sig)
                Zt = np.sum(N ** (1j * zs) * cv) + np.sum(N ** (-1j * zs) * cvm)
                Oarr[i] = abs(Zt) ** 2 - 2 * float(np.sum(np.abs(cv) ** 2))
        O_T = 0.5 * float(np.sum(wu * (Oz + O3)))
        row6 = [r for r in J['weights']['1-x']['per_N'][str(lg)]['rows'] if r['d'] == 6][0]
        G6 = row6['G']
        print(f"N=2^{lg}: O^T(N;6) = {O_T:+.6f};  medido G/N^2 - D(zeros) = {G6 - (D_zeros + D_tail):+.6f};  "
              f"G/N^2 - D - O^T = {G6 - (D_zeros + D_tail) - O_T:+.6f}  (PNT del informe {J['weights']['1-x']['per_N'][str(lg)]['summary']['pnt_term']:+.6f})", flush=True)
        res['offdiag'][str(lg)] = dict(O_T=O_T, G_over_N2=G6, resid_after_O=G6 - (D_zeros + D_tail) - O_T,
                                       pnt_term=J['weights']['1-x']['per_N'][str(lg)]['summary']['pnt_term'])
    res['seconds'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), indent=1)
    print("listo", time.time() - t0)

if __name__ == '__main__':
    main()
