"""k1_constantes.py -- R1 / K1, corrida 1 de Opus.  Contraste (i): la diagonal D_w(d) sin primos.

Para cada peso w calcula, por cuadratura de Gauss-Legendre (sin primos y sin ceros):
  r0, kappa_w, h_w, gamma_w y c_B^w = r0^2(gamma+log 2pi) - r0 kappa_w + h_w - gamma_w/2   (manuscrito v5, Prop. Xw(b));
  las piezas de la diagonal (K1_informe.md, Paso 2):
     T_pair  = int_0^1 dy/(1-y) int_0^1 w(x) [ r0 w(x) - w(xy) r(x(1-y)) ] dx
     T_cross = int_0^1 dv/(1+v) int_0^1 w(x) w(xv) g(x(1+v)) dx
     X2      = int_0^1 w(x)^2 g(2x) dx            ( = I_1 - r0^2 = r0^2 - I_0 )
     A_odd   = 4 w(0) int_0^1 w(u) r(u) du         (coeficiente del residuo de los caracteres impares)
     H1int   = 2 w(0) int_0^1 w(u) [ int (phi_u - phi_{-u})(x) log x dx ] du   (polos de zeta; caracter principal)
  y la constante diagonal  C_diag = -r0^2(gamma + log 2pi) + T_pair + T_cross,  que debe ser -c_B^w;
  ademas las identidades exactas T_cross = gamma_w/2 y T_pair = r0 kappa_w - h_w (Paso 2, Lema 2).
Pesos: 1-x, (1-x)^2, bump = exp(-x^2/(1-x^2)) (el de ronda3 / Obs. HFG), expw = exp(-x/(1-x)) (Teorema SSw).
Uso: python3 -I k1_constantes.py     (< 1 min)
Salida: k1_constantes.json
"""
import json, math
import numpy as np

OUT = "/home/claude/ariadna/R1/run1_K1/k1_constantes.json"
EUL = 0.57721566490153286061

def make_w(name):
    if name == '1-x':     f = lambda x: 1.0 - x
    elif name == '(1-x)^2': f = lambda x: (1.0 - x) ** 2
    elif name == 'bump':  f = lambda x: np.exp(-x * x / np.maximum(1.0 - x * x, 1e-300))
    elif name == 'expw':  f = lambda x: np.exp(-x / np.maximum(1.0 - x, 1e-300))
    else: raise ValueError(name)
    def W(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x); m = (x >= 0) & (x < 1)
        out[m] = f(x[m]); return out
    return W

GL = {}
def gl(n):
    if n not in GL: GL[n] = np.polynomial.legendre.leggauss(n)
    return GL[n]

def integ(f, a, b, n=200):
    """int_a^b f, vectorizado sobre arrays a, b (broadcast)."""
    xg, wg = gl(n)
    a = np.asarray(a, float); b = np.asarray(b, float)
    x = 0.5 * (b - a)[..., None] * xg + 0.5 * (a + b)[..., None]
    return np.sum(0.5 * (b - a)[..., None] * wg * f(x), axis=-1)

def run(name, n=240):
    W = make_w(name)
    r = lambda tau: integ(lambda x: W(x) * W(x + np.asarray(tau)[..., None]), 0 * np.asarray(tau), 1 - np.asarray(tau), n)
    def g(z):
        z = np.asarray(z, float)
        lo = np.maximum(0, z - 1); hi = np.minimum(1, z)
        return integ(lambda a: W(a) * W(z[..., None] - a), lo, hi, n)
    r0 = float(integ(W2 := (lambda x: W(x) ** 2), 0.0, 1.0, 400))
    # kappa_w = int w^2 log x : sustitucion x = t^2 para suavizar el log
    kap = float(integ(lambda t: W(t * t) ** 2 * np.log(np.maximum(t * t, 1e-300)) * 2 * t, 0.0, 1.0, 400))
    # h_w = int_0^1 (r^2 - r0^2)/u du ;  gamma_w = int_0^2 g^2/u du
    xg, wg = gl(n)
    uu = 0.5 * xg + 0.5; wu = 0.5 * wg
    hw = float(np.sum(wu * (r(uu) ** 2 - r0 ** 2) / uu))
    gam = float(np.sum(wu * g(uu) ** 2 / uu) + np.sum(wu * g(1 + uu) ** 2 / (1 + uu)))
    cBw = r0 ** 2 * (EUL + math.log(2 * math.pi)) - r0 * kap + hw - gam / 2
    # T_pair: tensor (y, x)
    yy = uu; wy = wu
    X = uu[None, :] * np.ones_like(yy)[:, None]; Y = yy[:, None] * np.ones_like(uu)[None, :]
    inner = W(X) * (r0 * W(X) - W(X * Y) * r(X * (1 - Y)))
    Tpair = float(np.sum(wy[:, None] * wu[None, :] * inner / (1 - Y)))
    # T_cross: por v, partir x en x* = 1/(1+v) (g'' salta en z=1)
    Tc = 0.0
    for v, wv in zip(yy, wy):
        xs = 1.0 / (1.0 + v)
        val = integ(lambda x: W(x) * W(x * v) * g(x * (1 + v)), np.array([0.0, xs]), np.array([xs, 1.0]), n).sum()
        Tc += wv * val / (1 + v)
    X2 = float(integ(lambda x: W(x) ** 2 * g(2 * x), np.array([0.0, 0.5]), np.array([0.5, 1.0]), n).sum())
    w0 = float(W(np.array([0.0]))[0])
    A_odd = 4 * w0 * float(np.sum(wu * W(uu) * r(uu)))
    # H1int
    def cprime(u):
        A = integ(lambda x: W(x) * W(x - u) * np.log(x), u, 1.0, n)
        B = integ(lambda t: W((1 - u) * t * t) * W((1 - u) * t * t + u) * np.log(np.maximum((1 - u) * t * t, 1e-300)) * 2 * (1 - u) * t, 0.0, 1.0, n)
        return float(A - B)
    H1int = 2 * w0 * float(sum(wi * float(W(np.array([ui]))[0]) * cprime(ui) for ui, wi in zip(uu, wu)))
    Cdiag = -r0 ** 2 * (EUL + math.log(2 * math.pi)) + Tpair + Tc
    out = dict(weight=name, nodes=n, r0=r0, kappa=kap, h_w=hw, gamma_w=gam, cBw=cBw,
               T_pair=Tpair, T_cross=Tc, X2=X2, A_odd=A_odd, H1int=H1int, w0=w0,
               id_cross=Tc - gam / 2, id_pair=Tpair - (r0 * kap - hw),
               C_diag=Cdiag, C_diag_plus_cBw=Cdiag + cBw)
    return out

def main():
    res = []
    for name in ('1-x', '(1-x)^2', 'bump', 'expw'):
        rows = [run(name, n) for n in (160, 240)]
        o = rows[-1]; o['C_diag_plus_cBw_n160'] = rows[0]['C_diag_plus_cBw']
        res.append(o)
        print(json.dumps({k: (round(v, 10) if isinstance(v, float) else v) for k, v in o.items()}), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

if __name__ == '__main__':
    main()
