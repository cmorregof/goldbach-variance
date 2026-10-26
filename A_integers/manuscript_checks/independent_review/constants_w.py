"""Independent check of the smooth-weight constants of Thm SSw / Prop Xw."""
import mpmath as mp
mp.mp.dps = 20
g = mp.euler; S1 = mp.mpf('0.533821625093524696548430443962')
a1 = -mp.log(4*mp.pi)/2 - g - S1/2
weights = {'1-x': lambda x: 1-x, '(1-x)^2': lambda x: (1-x)**2,
           'exp(-x/(1-x))': lambda x: mp.e**(-x/(1-x)) if x < 1 else mp.mpf(0)}
for name, w in weights.items():
    r = lambda u: mp.quad(lambda x: w(x)*w(x+u), [0, 1-u]) if u < 1 else mp.mpf(0)
    def gconv(u):
        lo, hi = max(0, u-1), min(1, u)
        return mp.quad(lambda y: w(u-y)*w(y), [lo, hi]) if hi > lo else mp.mpf(0)
    r0 = r(0)
    hw = mp.quad(lambda u: (r(u)**2 - r0**2)/u, [0, 0.5, 1])
    gw = mp.quad(lambda u: gconv(u)**2/u, [0, 1, 2])
    kw = mp.quad(lambda x: w(x)**2*mp.log(x), [0, 1])
    css = 2*r0**2*a1 - hw + gw/2
    cw = 2*r0*kw + css
    cBw = r0**2*(g + mp.log(2*mp.pi)) - r0*kw + hw - gw/2
    # Plancherel check of N^3 cancellation: 2 int r^2 = int g^2
    pl = 2*mp.quad(lambda u: r(u)**2, [0, 1]) - mp.quad(lambda u: gconv(u)**2, [0, 1, 2])
    print(f"{name:14s} r0={mp.nstr(r0,8)} h={mp.nstr(hw,9)} gamma={mp.nstr(gw,9)} kappa={mp.nstr(kw,9)} "
          f"c_ss={mp.nstr(css,9)} c_w={mp.nstr(cw,9)} c_B^w={mp.nstr(cBw,9)} 2c_B^w={mp.nstr(2*cBw,7)} plancherel={mp.nstr(pl,3)}")
