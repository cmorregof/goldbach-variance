# Independent check of c_B^w = r0^2(gamma+log 2pi) - r0*kappa_w + h_w - gamma_w/2 for w(x)=1-x
# and of 2c_B^w; also the identity int_0^1 r w = int_0^1 g w (Euler-Maclaurin cancellation, referee m1)
from mpmath import mp, mpf, quad, log, euler, pi
mp.dps = 30
w  = lambda x: (1-x) if 0<=x<=1 else mpf(0)
r  = lambda u: quad(lambda x: w(x)*w(x+u), [0, 1-u])
g  = lambda u: quad(lambda y: w(u-y)*w(y), [max(0,u-1), min(1,u)])
r0 = quad(lambda x: w(x)**2, [0,1])
kap = quad(lambda x: w(x)**2*log(x), [0,1])
# closed form r(u) = (1-u)^2(2+u)/6
rc = lambda u: (1-u)**2*(2+u)/6
h  = quad(lambda u: (rc(u)**2 - r0**2)/u, [0,1])
# g for w=1-x: on [0,1] g(u)=u - u^2 + u^3/6 ; on [1,2] g(u)=(2-u)^3/6
gc = lambda u: (u - u**2 + u**3/6) if u<=1 else (2-u)**3/6
gam = quad(lambda u: gc(u)**2/u, [0,1,2])
cBw = r0**2*(euler+log(2*pi)) - r0*kap + h - gam/2
print("r0", r0, "kappa", kap, "(-11/18 =", mpf(-11)/18, ")")
print("h_w", h, "(-5/24 =", mpf(-5)/24, ")")
print("gamma_w", gam)
print("check g closed form at 0.3,1.4:", g(mpf('0.3')), gc(mpf('0.3')), g(mpf('1.4')), gc(mpf('1.4')))
print("c_B^w =", cBw)
print("2 c_B^w =", 2*cBw)
Irw = quad(lambda u: rc(u)*w(u), [0,1]); Igw = quad(lambda u: gc(u)*w(u), [0,1])
print("int r w =", Irw, " int g w =", Igw)
# second weight (1-x)^2 for the cancellation identity (generic check)
w2 = lambda x: (1-x)**2 if 0<=x<=1 else mpf(0)
r2 = lambda u: quad(lambda x: w2(x)*w2(x+u), [0,1-u])
g2 = lambda u: quad(lambda y: w2(u-y)*w2(y), [0,u])
print("(1-x)^2: int r w =", quad(lambda u: r2(u)*w2(u), [0,1]), " int g w =", quad(lambda u: g2(u)*w2(u), [0,1]))
