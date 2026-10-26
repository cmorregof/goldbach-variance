"""Cruce de mi r, rho con g2_comun (run8) y con e1_comun (run6)."""
import sys, numpy as np
sys.path.insert(0, '/home/claude/ariadna/R3/run8_G2'); sys.path.insert(0, '/home/claude/ariadna/R3/run6_E1')
import arb_comun as A
import g2_comun as G
for n in (210, 1000, 2310, 4096, 6006, 10000):
    r1, q1 = A.r_rho(n); r2, q2 = G.r_rho(n)
    m = r2 > 0
    print(n, "rel err r:", float(np.max(np.abs(r1[m] - r2[m]) / r2[m])), " rho:", float(np.max(np.abs(q1[m] - q2[m]) / q2[m])), " zeros match:", bool(np.all((r1 == 0) == (r2 == 0))))
    # tau = r/rho vs prod (1+t_p) sobre p<=2n (control de t_p)
    h = np.arange(1, n)
    tau = np.ones(n - 1)
    for p in A.primes_upto(2 * n + 2):
        p = int(p)
        if p >= 5: tau *= 1 + A.t_p(p, n, h)
    tr, tq = A.tail_sums(2 * n + 2)
    tau *= np.exp(tr - tq)
    print("   max |r - rho*tau|/r:", float(np.max(np.abs(r1[m] - q1[m] * tau[m]) / r1[m])))
