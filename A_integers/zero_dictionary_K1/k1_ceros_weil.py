"""k1_ceros_weil.py -- R1 / K1, corrida 1 de Opus.

Comprueba con CEROS REALES la contabilidad de la formula explicita de Weil usada en K1_informe.md, Paso 2,
para el peso w(x) = 1-x y valores fijos de u:

  para chi primitivo (q, a = 0 par / 1 impar) y c(s) = phi~_u(s) - chi(-1) phi~_{-u}(s),
  H(s) = c(s) c(1-s) (= |c(1/2+it)|^2 en la recta critica):

  sum_{|gamma|<T} |c(1/2+i gamma)|^2
     = (1/2pi) int_{-T}^{T} |c(1/2+it)|^2 [ log(q/pi) + Re psi((1/2+a+it)/2) ] dt          (densidad + Gamma)
       + [chi = 1 (zeta)]   2 H(1)                                                         (polos de Lambda_zeta en 0 y 1)
       + [chi impar]        2 c_0 c(1) Re L'/L(1,chi)                                      (polo de c en s=0)
       - 2 sum_n Lambda(n) Re chi(n) Hcheck(n),   Hcheck(n) = int_0^{1/n} (c_f(y)-c_f(0)) c_f(ny) dy
     + O(error de truncacion ~ |c(T)|^2 log T).

  Aqui c_f = phi_u - chi(-1) phi_{-u} es la funcion en x cuya transformada de Mellin es c, c_0 = c_f(0) y
  c(1) = int c_f.

Casos: zeta (par, principal, q=1) y chi_{-4} (impar, q=4).  Ademas, comprueba en el espacio t (sin ceros) las
formulas en el espacio x de las integrales con digamma (Paso 2, (2.6)-(2.8)) para cada u.

Uso: python3 -I k1_ceros_weil.py      (unos 6-8 minutos, < 300 MB)
Salida: k1_ceros_weil.json
"""
import json, math, time, sys
import numpy as np
import mpmath as mp
from scipy.special import psi as digamma
from scipy.integrate import quad

OUT = "/home/claude/ariadna/R1/run1_K1/k1_ceros_weil.json"
EUL = 0.57721566490153286061
mp.mp.dps = 20

# ------------------------------------------------------------------ peso 1-x, transformadas cerradas
def a_mel(s, u):   # Mellin de phi_u(x) = (1-x)(1+u-x) en [u,1]
    return (1 + u) * (1 - u ** s) / s - (2 + u) * (1 - u ** (s + 1)) / (s + 1) + (1 - u ** (s + 2)) / (s + 2)

def b_mel(s, u):   # Mellin de phi_{-u}(x) = (1-x)(1-u-x) en [0,1-u]
    v = 1 - u
    return v * v ** s / s - (2 - u) * v ** (s + 1) / (s + 1) + v ** (s + 2) / (s + 2)

def phi_p(x, u):
    x = np.asarray(x, float); return np.where((x >= u) & (x <= 1), (1 - x) * (1 + u - x), 0.0)

def phi_m(x, u):
    x = np.asarray(x, float); return np.where((x >= 0) & (x <= 1 - u), (1 - x) * (1 - u - x), 0.0)

def r_lin(u):
    return (1 - u) ** 2 * (2 + u) / 6

# ------------------------------------------------------------------ cuadratura en t
def tgrid(T, panel=0.5, k=12):
    xg, wg = np.polynomial.legendre.leggauss(k)
    n = int(math.ceil(T / panel)); edges = np.linspace(0, T, n + 1)
    a, b = edges[:-1], edges[1:]
    t = (0.5 * (b - a)[:, None] * xg[None, :] + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * wg[None, :]).ravel()
    return t, w

def tspace_integrals(u, T, chunk=400000):
    """(1/2pi) int_R of: (|a|^2+|b|^2) Re psi(s);  2Re(a conj b) Re beta(s);  |a|^2+|b|^2;  2 Re(a conj b);
       |a-b|^2 [Re psi(s/2)];  |a+b|^2 [Re psi((s+1)/2)];  |a-b|^2;  |a+b|^2   (s = 1/2+it), integrando en [0,T]
       y duplicando (todos los integrandos son pares en t)."""
    t, w = tgrid(T)
    acc = np.zeros(8)
    for i in range(0, len(t), chunk):
        tt = t[i:i + chunk]; ww = w[i:i + chunk]; s = 0.5 + 1j * tt
        A = a_mel(s, u); B = b_mel(s, u)
        pa = np.abs(A) ** 2 + np.abs(B) ** 2; cr = 2 * np.real(A * np.conj(B))
        ps = np.real(digamma(s)); ps0 = np.real(digamma(s / 2)); ps1 = np.real(digamma((s + 1) / 2))
        beta = 0.5 * (ps1 - ps0)                                    # Re beta(s) = (psi((s+1)/2)-psi(s/2))/2
        vals = [pa * ps, cr * beta, pa, cr, np.abs(A - B) ** 2 * ps0, np.abs(A + B) ** 2 * ps1,
                np.abs(A - B) ** 2, np.abs(A + B) ** 2]
        acc += np.array([np.sum(ww * v) for v in vals])
    acc = acc * 2 / (2 * math.pi)
    return acc

def tail_pair(u, T):
    # |a|^2 ~ (1-u)^2 u / t^2 (salto de phi_u en x=u); Re psi ~ log t:   (1/pi) int_T^inf
    C = (1 - u) ** 2 * u
    return dict(pa_psi=C * (math.log(T) + 1) / T / math.pi, pa=C / T / math.pi,
                psi0=C * (math.log(T / 2) + 1) / T / math.pi, psi1=C * (math.log(T / 2) + 1) / T / math.pi)

# ------------------------------------------------------------------ formulas en el espacio x (por u)
def xspace_per_u(u):
    fs = [lambda x: phi_p(x, u), lambda x: phi_m(x, u)]
    brk = sorted(set([0.0, u, 1 - u, 1.0]))
    def i1(f, g):
        return sum(quad(lambda x: float(f(x) * g(x)), brk[k], brk[k + 1], limit=200)[0] for k in range(len(brk) - 1))
    def corr(f, g, y):   # int f(x) g(xy) dx
        pts = sorted(set([0.0, 1.0] + [p for p in (u, 1 - u, u / y if y > 0 else 2, (1 - u) / y if y > 0 else 2) if 0 < p < 1]))
        return sum(quad(lambda x: float(f(x) * g(x * y)), pts[k], pts[k + 1], limit=200)[0] for k in range(len(pts) - 1))
    pair = 0.0
    for f in fs:
        n2 = i1(f, f)
        J = quad(lambda y: (n2 - corr(f, f, y)) / (1 - y), 0, 1, limit=200, points=[u, 1 - u])[0]
        pair += -EUL * n2 + J
    cross = quad(lambda v: (corr(fs[0], fs[1], v) + corr(fs[1], fs[0], v)) / (1 + v), 0, 1, limit=200, points=[u, 1 - u])[0]
    norm = i1(fs[0], fs[0]) + i1(fs[1], fs[1]); crossnorm = 2 * i1(fs[0], fs[1])
    return dict(pair_psi=pair, cross_beta=cross, norm=norm, crossnorm=crossnorm)

# ------------------------------------------------------------------ ceros
def zeta_zeros(T):
    zs = []; n = 1
    while True:
        g = float(mp.zetazero(n).imag)
        if g > T: return zs, g
        zs.append(g); n += 1

def Zchi4(t):
    s = mp.mpc(0.5, t)
    th = mp.im(mp.loggamma((s + 1) / 2)) + t / 2 * mp.log(4 / mp.pi)
    v = mp.exp(1j * th) * mp.dirichlet(s, [0, 1, 0, -1])
    return float(mp.re(v)), float(mp.im(v))

def chi4_zeros(T, h=0.1):
    ts = np.arange(0.05, T + h, h); vals = []; maximag = 0.0
    for t in ts:
        re, im = Zchi4(float(t)); vals.append(re); maximag = max(maximag, abs(im) / (abs(re) + 1e-30) if abs(re) > 1e-8 else 0)
    vals = np.array(vals); zs = []
    for i in range(len(ts) - 1):
        if vals[i] == 0 or vals[i] * vals[i + 1] < 0:
            lo, hi, flo = float(ts[i]), float(ts[i + 1]), vals[i]
            for _ in range(40):
                mid = 0.5 * (lo + hi); fm = Zchi4(mid)[0]
                if flo * fm <= 0: hi = mid
                else: lo, flo = mid, fm
                if hi - lo < 1e-11: break
            zs.append(0.5 * (lo + hi))
    return zs, float(np.min(np.abs(np.diff(vals)))), maximag

# ------------------------------------------------------------------ terminos aritmeticos
def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]

def prime_powers(X):
    P = sieve(X); out = []
    for p in P:
        q = int(p)
        while q <= X: out.append((q, math.log(p))); q *= int(p)
    out.sort(); return out

def Hcheck(n, cf, u):
    c0 = float(cf(np.array([0.0]))[0])
    pts = sorted(set([0.0, 1.0 / n] + [p for p in (u, 1 - u, u / n, (1 - u) / n) if 0 < p < 1.0 / n]))
    return sum(quad(lambda y: float((cf(y) - c0) * cf(n * y)), pts[k], pts[k + 1], limit=100)[0] for k in range(len(pts) - 1))

def arith_terms(u, sign, chi, PP, Xh):
    """sign = chi(-1); chi: funcion n -> chi(n). Devuelve c0, c(1), H(1), -2 sum Lambda Re chi Hcheck."""
    cf = lambda x: phi_p(x, u) - sign * phi_m(x, u)
    c0 = float(cf(np.array([0.0]))[0])
    c1 = r_lin(u) * (1 - sign)
    brk = sorted(set([0.0, u, 1 - u, 1.0]))
    cprime1 = sum(quad(lambda x: float(cf(x)) * math.log(x) if x > 0 else 0.0, brk[k], brk[k + 1], limit=200)[0] for k in range(len(brk) - 1))
    H1 = -c0 * cprime1
    S = 0.0
    for n, lg in PP:
        if n > Xh: break
        cn = chi(n)
        if cn == 0: continue
        S += lg * cn * Hcheck(n, cf, u)
    # cola: Hcheck(n) ~ K/n^2 con K = int_0^1 z eps'(0) c_f(z) dz; eps'(0) = derivada de c_f en 0+
    h = 1e-7; epsp = float((cf(np.array([h]))[0] - c0) / h)
    K = quad(lambda z: z * epsp * float(cf(z)), 0, 1, points=[u, 1 - u], limit=200)[0]
    tail = K / Xh if sign == 1 and chi(3) == 1 else 0.0   # solo para chi=1 (zeta); para chi_-4 la cola se cancela en media
    return dict(c0=c0, c1=c1, H1=H1, prime_sum=-2 * (S + tail), tail_used=-2 * tail)

def main():
    t0 = time.time(); res = dict()
    us = [0.3, 0.7]
    # ---------- (1) espacio t frente a espacio x, por u
    T = 1.0e5
    res['tspace_vs_xspace'] = []
    for u in us:
        It = tspace_integrals(u, T); tl = tail_pair(u, T); xs = xspace_per_u(u)
        row = dict(u=u, T=T,
                   pair_psi_t=It[0] + tl['pa_psi'], pair_psi_x=xs['pair_psi'],
                   cross_beta_t=It[1], cross_beta_x=xs['cross_beta'],
                   norm_t=It[2] + tl['pa'], norm_x=xs['norm'], crossnorm_t=It[3], crossnorm_x=xs['crossnorm'])
        res['tspace_vs_xspace'].append(row)
        print("[t vs x]", json.dumps({k: (round(v, 9) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    # ---------- (2) ceros
    PP = prime_powers(200000)
    Tz = 600.0
    zz, gnext = zeta_zeros(Tz)
    print(f"zeta: {len(zz)} ceros con gamma<{Tz}; siguiente {gnext:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    T4 = 400.0
    z4, mindiff, maximag = chi4_zeros(T4)
    # numero esperado de ceros 0<gamma<T de L(s,chi) primitivo: (T/2pi) log(qT/(2 pi e)) + O(log qT)
    exp4 = T4 / (2 * math.pi) * math.log(4 * T4 / (2 * math.pi * math.e))
    expz = Tz / (2 * math.pi) * math.log(Tz / (2 * math.pi * math.e)) + 7 / 8
    print(f"chi_-4: {len(z4)} ceros con 0<gamma<{T4} (esperado ~{exp4:.1f}); max|Im|/|Re| de Lambda en la recta {maximag:.1e} ({time.time()-t0:.0f}s)", flush=True)
    Lp = mp.diff(lambda s: mp.dirichlet(s, [0, 1, 0, -1]), 1); L1 = mp.dirichlet(1, [0, 1, 0, -1])
    LpL = float(Lp / L1)
    chi4 = lambda n: 0 if n % 2 == 0 else (1 if n % 4 == 1 else -1)
    one = lambda n: 1
    res['weil'] = []
    for u in us:
        for name, zs, Tc, sign, q, chi in (('zeta', zz, Tz, 1, 1, one), ('chi_-4', z4, T4, -1, 4, chi4)):
            g = np.array(zs); s = 0.5 + 1j * g
            cvals = a_mel(s, u) - sign * b_mel(s, u)
            lhs = 2 * float(np.sum(np.abs(cvals) ** 2))       # +-gamma
            It = tspace_integrals(u, Tc)
            dens = math.log(q / math.pi) * (It[6] if sign == 1 else It[7]) + (It[4] if sign == 1 else It[5])
            ar = arith_terms(u, sign, chi, PP, 200000)
            extra = 0.0
            if name == 'zeta': extra += 2 * ar['H1']
            if sign == -1: extra += 2 * ar['c0'] * ar['c1'] * LpL
            extra += ar['prime_sum']
            rhs = dens + extra
            row = dict(u=u, L=name, nzeros_pos=len(zs), T=Tc, lhs_sum_zeros=lhs, rhs_density=dens,
                       rhs_H1=2 * ar['H1'] if name == 'zeta' else 0.0,
                       rhs_residue_odd=2 * ar['c0'] * ar['c1'] * LpL if sign == -1 else 0.0,
                       rhs_prime_sum=ar['prime_sum'], rhs_total=rhs, diff=lhs - rhs,
                       diff_without_extras=lhs - dens, trunc_scale=(1 - u) ** 2 * u / Tc ** 2 * math.log(Tc))
            res['weil'].append(row)
            print("[Weil]", json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    res['meta'] = dict(zeta_expected=expz, chi4_expected=exp4, LpL_chi4=LpL, chi4_maximag=maximag, seconds=time.time() - t0)
    json.dump(res, open(OUT, 'w'), indent=1)
    print("listo", time.time() - t0)

if __name__ == '__main__':
    main()
