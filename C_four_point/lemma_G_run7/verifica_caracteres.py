"""[M] (C1) Identidad de caracteres (Prop. 1):  T_e(n,h) = c2(e) * sum_{chi in X_e} chi(n) conj(chi(h)),
X_e = caracteres primitivos mod e, pares en cada primo.  Se compara para todo (n,h) mod e.
(C2) Decaimiento de S_e = sum_h w rho T_e para e pequeño (Prop. 3: trivial zero L(0,chi)=0)."""
import math, json, itertools, numpy as np
from g_comun import r_and_rho, primes_upto, fejer_w, prime_factors
pr = primes_upto(1 << 24)

def prim_root(p):
    fac = prime_factors(p - 1, pr)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in fac): return g

def ind_table(p):
    g = prim_root(p); ind = np.full(p, -1); x = 1
    for k in range(p - 1): ind[x] = k; x = x * g % p
    return ind

def c2(p): return 2.0 / ((p - 2) * (p - 3))

def t_p(p, n, h):   # vectorizado en h (array), n escalar
    h = np.asarray(h); out = np.full(h.shape, -c2(p))
    out[(h - n) % p == 0] = 1.0 / (p - 2); out[(h + n) % p == 0] = 1.0 / (p - 2)
    out[h % p == 0] = 0.0
    if n % p == 0: out[:] = 0.0
    return out

res = {}
# (C1)
err = 0.0
for ps in ((5,), (7,), (13,), (5, 7), (5, 7, 11), (13, 17)):
    e = int(np.prod(ps)); inds = {p: ind_table(p) for p in ps}
    # characters: for each p choose even nonzero k mod (p-1): chi_p(x) = exp(2 pi i k ind(x)/(p-1))
    choices = [[k for k in range(2, p - 1, 2)] for p in ps]
    nn, hh = np.meshgrid(np.arange(e), np.arange(e), indexing="ij")
    lhs = np.ones((e, e))
    for p in ps: lhs = lhs * np.array([t_p(p, int(n), hh[n]) for n in range(e)])
    rhs = np.zeros((e, e), complex)
    for ks in itertools.product(*choices):
        cn = np.ones(e, complex); ch = np.ones(e, complex)
        for p, k in zip(ps, ks):
            ip = inds[p]; x = np.arange(e) % p
            val = np.where(ip[x] >= 0, np.exp(2j * math.pi * k * ip[x] / (p - 1)), 0)
            cn *= val; ch *= val
        rhs += np.outer(cn, np.conj(ch))
    rhs *= np.prod([c2(p) for p in ps])
    err = max(err, float(np.abs(lhs - rhs).max()))
    print("C1 e=%d  max|T_e - c2 sum chi(n)conj chi(h)| = %.2e   (#X_e=%d)" % (e, np.abs(lhs - rhs).max(), len(list(itertools.product(*choices)))))
res["C1_max_err"] = err
# (C2)
rows = []
for n in (1055530, 1684178, 1048576, 1021020):
    r, rho = r_and_rho(n, 100000, pr)
    pf = set(prime_factors(n, pr))
    es = []
    for e in range(5, 400):
        f = prime_factors(e, pr)
        if any(p < 5 or p in pf for p in f) or any(e % (p * p) == 0 for p in f): continue
        es.append((e, f))
    for H in (100, 1000, 10000, 100000):
        h = np.arange(1, H + 1); base = fejer_w(H) * rho[:H]
        S = []
        for e, f in es:
            T = np.ones(H)
            for p in f: T *= t_p(p, n, h)
            S.append(float(np.dot(base, T)))
        S = np.array(S); emax = [e for e, _ in es]
        small = np.array(emax) <= 40
        rows.append(dict(n=n, H=H, sum_S_e_le40=float(S[small].sum()), sumabs_S_e_le40=float(np.abs(S[small]).sum()),
                         sum_S_e_le400=float(S.sum()), sumabs_le400=float(np.abs(S).sum()),
                         S5=float(S[0]), max_abs=float(np.abs(S).max())))
        print("C2 n=%d H=%6d  sum_{e<=40} S_e=%+.2e (sum|.|=%.2e)  sum_{e<400} S_e=%+.2e (sum|.|=%.2e)  S_5=%+.2e" %
              (n, H, rows[-1]["sum_S_e_le40"], rows[-1]["sumabs_S_e_le40"], rows[-1]["sum_S_e_le400"], rows[-1]["sumabs_le400"], rows[-1]["S5"]), flush=True)
res["C2"] = rows
json.dump(res, open("verifica_caracteres.json", "w"), indent=1)
