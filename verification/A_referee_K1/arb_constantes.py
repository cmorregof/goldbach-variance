"""arb_constantes.py -- árbitro gen2 de R1-K1.  Recomputación INDEPENDIENTE de las constantes de (A2).

(1) w = 1-x, todo exacto con sympy: r0, kappa_w, h_w, gamma_w, c_B^w, y las dos piezas de la diagonal
    T_pair  = int_0^1 dy/(1-y) int_0^1 w(x)[r0 w(x) - w(xy) r(x(1-y))] dx
    T_cross = int_0^1 dv/(1+v) int_0^1 w(x) w(xv) g(x(1+v)) dx
    y X2, A_odd, H1 de la fórmula (2.5) del informe K1.  Se comprueba T_cross = gamma_w/2, T_pair = r0 kappa_w - h_w
    y C_diag = -r0^2(gamma+log 2pi) + T_pair + T_cross = -c_B^w, todo en forma cerrada.
(2) w = (1-x)^2: lo mismo por cuadratura adaptativa (scipy), sin reutilizar nada de k1_constantes.py.
Uso: python3 -I arb_constantes.py
"""
import math
import sympy as sp
from scipy.integrate import quad

EUL = 0.57721566490153286061

# ---------------------------------------------------------------- (1) exacto, w = 1 - x
x, y, v, u, a, b, z, tau = sp.symbols('x y v u a b z tau', positive=True)
w = lambda t: 1 - t
r0 = sp.integrate(w(x)**2, (x, 0, 1))                                  # 1/3
r = sp.simplify(sp.integrate(w(x)*w(x+u), (x, 0, 1-u)))                # (1-u)^2 (2+u)/6
g1 = sp.simplify(sp.integrate(w(x)*w(z-x), (x, 0, z)))                 # g en [0,1]
g2 = sp.simplify(sp.integrate(w(x)*w(z-x), (x, z-1, 1)))               # g en [1,2]
kappa = sp.integrate(w(x)**2*sp.log(x), (x, 0, 1))                     # -11/18
h_w = sp.integrate(sp.expand((r**2 - r0**2)/u), (u, 0, 1))             # -5/24
gamma_w = sp.integrate(sp.expand(g1**2/z), (z, 0, 1)) + sp.integrate(sp.expand(g2**2/z), (z, 1, 2))
gamma_w = sp.nsimplify(sp.simplify(gamma_w))
cB = r0**2*(sp.EulerGamma + sp.log(2*sp.pi)) - r0*kappa + h_w - gamma_w/2

# T_cross: en variables (a,b) = (x, xv), es (1/2) iint_{[0,1]^2} w(a)w(b) g(a+b)/(a+b) ; lo hago en las variables
# originales para no reutilizar el cambio de variables del informe:  int_0^1 dv/(1+v) int_0^1 w(x) w(xv) g(x(1+v)) dx,
# partiendo x en x < 1/(1+v) (g = g1) y x > 1/(1+v) (g = g2).
inner_c = (sp.integrate(sp.expand(w(x)*w(x*v)*g1.subs(z, x*(1+v))), (x, 0, 1/(1+v)))
           + sp.integrate(sp.expand(w(x)*w(x*v)*g2.subs(z, x*(1+v))), (x, 1/(1+v), 1)))
inner_c = sp.simplify(inner_c)
T_cross = sp.simplify(sp.integrate(sp.apart(sp.simplify(inner_c/(1+v)), v), (v, 0, 1)))

# T_pair: int_0^1 dy/(1-y) int_0^1 w(x)[r0 w(x) - w(xy) r(x(1-y))] dx   (absolutamente convergente)
inner_p = sp.simplify(sp.integrate(sp.expand(w(x)*(r0*w(x) - w(x*y)*r.subs(u, x*(1-y)))), (x, 0, 1)))
T_pair = sp.simplify(sp.integrate(sp.cancel(inner_p/(1-y)), (y, 0, 1)))

C_diag = -r0**2*(sp.EulerGamma + sp.log(2*sp.pi)) + T_pair + T_cross

# piezas del resto (2.5)
X2 = sp.integrate(sp.expand(w(x)**2*g1.subs(z, 2*x)), (x, 0, sp.Rational(1, 2))) + sp.integrate(sp.expand(w(x)**2*g2.subs(z, 2*x)), (x, sp.Rational(1, 2), 1))
A_odd = 4*w(0)*sp.integrate(sp.expand(w(u)*r), (u, 0, 1))
# H1 = 2 w(0) int_0^1 w(u) [ int_u^1 w(x)w(x-u) log x dx - int_0^{1-u} w(x) w(x+u) log x dx ] du
cp = sp.integrate(w(x)*w(x-u)*sp.log(x), (x, u, 1)) - sp.integrate(w(x)*w(x+u)*sp.log(x), (x, 0, 1-u))
H1 = 2*w(0)*sp.integrate(sp.expand(w(u)*sp.simplify(cp)), (u, 0, 1))

print("=== w = 1-x, exacto ===")
for name, val in (('r0', r0), ('kappa_w', kappa), ('h_w', h_w), ('gamma_w', gamma_w), ('c_B^w', cB),
                  ('T_pair', T_pair), ('T_cross', T_cross), ('C_diag', C_diag),
                  ('T_cross - gamma_w/2', sp.simplify(T_cross - gamma_w/2)),
                  ('T_pair - (r0 kappa - h_w)', sp.simplify(T_pair - (r0*kappa - h_w))),
                  ('C_diag + c_B^w', sp.simplify(C_diag + cB)),
                  ('X2', X2), ('A_odd', A_odd), ('H1', sp.simplify(H1))):
    print(f"{name:28s} = {sp.nsimplify(val) if val.is_number else val}   = {sp.N(val, 15)}")

# ---------------------------------------------------------------- (2) numérico, w = (1-x)^2
def W(t):
    return (1 - t)**2 if 0 <= t <= 1 else 0.0

def rr(t):
    return quad(lambda s: W(s)*W(s+t), 0, 1-t)[0] if t < 1 else 0.0

def gg(t):
    lo, hi = max(0.0, t-1), min(1.0, t)
    return quad(lambda s: W(s)*W(t-s), lo, hi)[0] if hi > lo else 0.0

r0n = quad(lambda t: W(t)**2, 0, 1)[0]
kapn = quad(lambda t: W(t)**2*math.log(t), 0, 1)[0]
hwn = quad(lambda t: (rr(t)**2 - r0n**2)/t, 0, 1)[0]
gamn = quad(lambda t: gg(t)**2/t, 0, 1)[0] + quad(lambda t: gg(t)**2/t, 1, 2)[0]
cBn = r0n**2*(EUL + math.log(2*math.pi)) - r0n*kapn + hwn - gamn/2
Tc = quad(lambda vv: quad(lambda xx: W(xx)*W(xx*vv)*gg(xx*(1+vv)), 0, 1, points=[1/(1+vv)])[0]/(1+vv), 0, 1)[0]
Tp = quad(lambda yy: quad(lambda xx: W(xx)*(r0n*W(xx) - W(xx*yy)*rr(xx*(1-yy))), 0, 1)[0]/(1-yy), 0, 1)[0]
Cd = -r0n**2*(EUL + math.log(2*math.pi)) + Tp + Tc
print("\n=== w = (1-x)^2, cuadratura independiente ===")
print(f"r0={r0n:.10f} kappa={kapn:.10f} h_w={hwn:.10f} gamma_w={gamn:.10f} c_B^w={cBn:.10f}")
print(f"T_pair={Tp:.10f}  r0 kappa - h_w={r0n*kapn-hwn:.10f}  dif={Tp-(r0n*kapn-hwn):.2e}")
print(f"T_cross={Tc:.10f} gamma_w/2={gamn/2:.10f}  dif={Tc-gamn/2:.2e}")
print(f"C_diag={Cd:.10f}  C_diag + c_B^w = {Cd+cBn:.2e}")
