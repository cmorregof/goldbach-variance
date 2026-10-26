"""Independent recomputation of the constants of Theorem SS (opus2 review).
Route 1: closed forms with prime sums evaluated through the prime zeta function
         P(s)=sum_n mu(n)/n log zeta(ns) (exact to working precision).
Route 2: a0 and a1 read off directly as Laurent coefficients of
         D(s)=2^{-s} zeta(s) zeta(s+1)^2 Jhat(s) by a Cauchy integral on |s|=r,
         with Jhat from a truncated Euler product (independent of the beta_2 algebra).
"""
import mpmath as mp
mp.mp.dps = 30

def mobius(n):
    m, r, p = n, 1, 2
    while p*p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0: return 0
            r = -r
        p += 1
    if m > 1: r = -r
    return r

def Pd(s, k):
    """k-th derivative of prime zeta P(s) = sum_p p^{-s}, s>=2 real, via mu-series."""
    tot = mp.mpf(0)
    for n in range(1, 200):
        mu = mobius(n)
        if mu == 0: continue
        if n*s > 200: break
        f = lambda t: mp.log(mp.zeta(t))
        tot += mu * mp.mpf(n)**(k-1) * mp.diff(f, n*s, k)
    return tot

# S1' = sum_{all p} log p/(p-1)^2 = sum_{k>=2} (k-1) sum_p log p p^{-k} = sum (k-1)(-P'(k))
S1all = mp.nsum(lambda k: (k-1)*(-Pd(int(k), 1)), [2, 90]) if False else sum((k-1)*(-Pd(k,1)) for k in range(2, 95))
S1 = S1all - mp.log(2)
# K = prod_p (1+(p-1)^-3): log K = sum_p log(1+(p-1)^{-3}); expand in x=1/p
x = mp.mpf(0)
def series_coeffs(f, n):
    return mp.taylor(f, 0, n)
cK = series_coeffs(lambda x: mp.log(1 + x**3/(1-x)**3), 90)
logK = sum(cK[k]*Pd(k, 0) for k in range(3, 91))
K = mp.e**logK
# (log Jhat)''(0) = -4 log^2 2 + sum_{p>2} log^2 p * F(p),  F as series in x=1/p
cF = series_coeffs(lambda x: x*((2-3*x)*(1-2*x)**2 - 2*(1-x)**2)/(1-x)**4, 90)
sumF_all = sum(cF[k]*Pd(k, 2) for k in range(2, 91))
F2 = lambda p: (2*p-3)*(p-2)**2/mp.mpf(p-1)**4 - 2*mp.mpf(p)/(p-1)**2
sumF = sumF_all - mp.log(2)**2*F2(2)
logJ2 = -4*mp.log(2)**2 + sumF

g = mp.euler; g1 = mp.stieltjes(1)
z0 = mp.zeta(0); z1 = mp.zeta(0, derivative=1); z2 = mp.zeta(0, derivative=2)/2
b1 = mp.log(2) + S1
b2 = logJ2/2 + b1**2/2
a2 = z0
a1 = z1 + z0*(2*g + b1)
a0 = z2 + z1*(2*g + b1) + z0*(g**2 - 2*g1 + 2*g*b1 + b2)
A = 4*mp.log(2) - 2
B = 2*mp.log(2)**2 - 6*mp.log(2) + 3
Bint = mp.quad(lambda u: min(u, 2-u)**2*mp.log(u)/u, [0, 1, 2])
c1 = 2*a1 + 2*mp.log(2) + mp.mpf(1)/2
c1cf = mp.mpf(1)/2 - 2*g - mp.log(mp.pi) - S1
c0 = 2*a0 - (1 + 4*mp.log(2))*a1 - mp.mpf(7)/4 + B/2
print("K       ", K)
print("S1      ", S1)
print("a1      ", a1, " closed form", -mp.log(4*mp.pi)/2 - g - S1/2)
print("a0      ", a0)
print("B, int  ", B, Bint)
print("c1      ", c1, c1cf)
print("c0      ", c0)
print("P2 coeffs: L", 2*(a1+mp.mpf(3)/4), " const", 2*a0 - 3*a1 - mp.mpf(7)/4)
print("G  coeffs: L", -(2*mp.log(2)-1), " const", a1*A - B/2)
print("c1-2, c0+1", c1-2, c0+1)
print("c_B       ", mp.mpf(1)/2 + g + mp.log(mp.pi/2))

# Route 2: Cauchy integral for Laurent coefficients of D(s) with truncated Euler product.
import numpy as np
P = 3_000_000
sieve = np.ones(P+1, bool); sieve[:2] = False
for p in range(2, int(P**0.5)+1):
    if sieve[p]: sieve[p*p::p] = False
pr = np.nonzero(sieve)[0][1:].astype(np.float64)  # odd primes
h = (2*pr-3)/(pr-2)**2
def logJhat(s):
    s = complex(s)
    ps = np.exp(-s*np.log(pr))
    jp = (1 + h*ps)*(1 - ps/pr)**2
    j0 = (1 + h)*(1 - 1/pr)**2
    val = np.sum(np.log(jp) - np.log(j0))
    val += 2*np.log(1 - 2**(-1-s)) - 2*np.log(0.5)
    return val
def D(s):
    s = mp.mpc(s)
    return mp.power(2, -s)*mp.zeta(s)*mp.zeta(s+1)**2*mp.e**mp.mpc(logJhat(complex(s)))
r = 0.05; M = 256
co = {}
for n in (-2, -1, 0):
    acc = mp.mpc(0)
    for j in range(M):
        t = 2*mp.pi*j/M; s = r*mp.e**(1j*t)
        acc += D(s)*s**(-n)
    co[n] = acc/M
print("Route 2 (P=3e6, r=0.05): a2", mp.nstr(co[-2].real, 10), " a1", mp.nstr(co[-1].real, 10), " a0", mp.nstr(co[0].real, 10))

# --- fixes appended after first run ---
# (1) K: the x-series of log(1+x^3/(1-x)^3) has radius 1/sqrt(3); at x=1/2 it converges too slowly,
#     so treat p=2 exactly and use P(k)-2^{-k} for the odd primes.
logK2 = mp.log(2) + sum(cK[k]*(Pd(k, 0) - mp.mpf(2)**(-k)) for k in range(3, 91))
print("K (p=2 exact)", mp.e**logK2)
# (2) Route 2 with a PNT tail correction for p>P: tail of log Jhat(s) ~ s*t1 + s^2*t2
t1 = mp.quad(lambda t: mp.log(t)/(t-1)**2/mp.log(t), [P, mp.inf])           # sum_{p>P} log p/(p-1)^2
t2 = mp.quad(lambda t: (mp.log(t)**2*F2(t))/mp.log(t), [P, mp.inf])/2        # (1/2) sum log^2 p F(p)
a1c = co[-1].real + a2*t1
a0c = co[0].real + a1c*t1 + a2*(t2 + t1**2/2)
print("Route 2 + tail: a1", mp.nstr(a1c, 10), " a0", mp.nstr(a0c, 10))
