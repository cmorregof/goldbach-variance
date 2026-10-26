"""R2, corrida 5 (T1): verificaciones numericas de la prueba de R2-E1 / R2-E2.

Partes (argv[1]):
  ss      : SS/q^{2n} = -sum_{deg[D,D']<=n} w_D w_D'  (formula exacta del Lema 2.4) y
            A0/q^{2n} = (sum_{e|n} e^2 pi(e))^2/q^{2n}, contra la auditoria (jsonl).
  e2 q    : constantes beta_q, gamma_q de R2-E2 y residuo SS/q^{2n} + (n^2/2+beta n+gamma), n<=60.
  lemaB q : prueba directa del Lema B (grado impar d=3, f escindido) clase por clase:
            para cada clase {chi*rho_Norm : rho bueno} de caracteres genericos,
            promedio retorcido (1/#cl) sum_rho rho(-1) F(theta)  con
            F = |tr theta|^2 - 1 (Lambda = adjunta) y F = |tr theta^n|^2 - min(n,2), n=2,3.
            Salida: max y media sobre clases de sqrt(q)*|promedio|.
Solo usa q primo impar en lemaB (logaritmo discreto en F_q).
"""
import sys, json, math
import numpy as np
from fractions import Fraction

AUD = "/home/claude/ariadna/auditorias/R2_numerica_2026-10-25.jsonl"


def mobius(k):
    res, m, p = 1, k, 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            res = -res
        p += 1
    if m > 1:
        res = -res
    return res


def npirr(q, d):
    return sum(mobius(e) * q ** (d // e) for e in range(1, d + 1) if d % e == 0) // d


def logCq(q, emax=200):
    s = 0.0
    for e in range(1, emax):
        if q ** e > 1e150:
            break
        s += npirr(q, e) * math.log1p(-1.0 / (q ** e - 1.0) ** 2)
    return s


def lcm_sum(q, n):
    """sum_{m<=n} c_m, c_m = sum_{deg[D,D']=m} w_D w_D' / C_q^2: coeficientes de
    prod_e (1 + a_e u^e)^{pi(e)}, a_e = (2|P|-3)/(|P|-2)^2, |P| = q^e (racional exacto)."""
    import mpmath as mp
    mp.mp.dps = 60
    poly = [mp.mpf(0)] * (n + 1)
    poly[0] = mp.mpf(1)
    for e in range(1, n + 1):
        Pq = mp.mpf(q) ** e
        a = (2 * Pq - 3) / (Pq - 2) ** 2
        pi = npirr(q, e)
        # (1 + a u^e)^pi = sum_k binom(pi,k) a^k u^{ek}
        fac = [mp.binomial(pi, k) * a ** k for k in range(n // e + 1)]
        new = [mp.mpf(0)] * (n + 1)
        for m0 in range(n + 1):
            if poly[m0] == 0:
                continue
            for k, c in enumerate(fac):
                if m0 + e * k > n:
                    break
                new[m0 + e * k] += poly[m0] * c
        poly = new
    return float(mp.fsum(poly))


def part_ss():
    rows = [json.loads(l) for l in open(AUD)]
    out = []
    for r in rows:
        q, n = r["q"], r["n"]
        C2 = math.exp(2 * logCq(q))
        SS = -C2 * lcm_sum(q, n)
        psi0 = sum(e * e * npirr(q, e) for e in range(1, n + 1) if n % e == 0)
        A0 = (psi0 / q ** n) ** 2
        out.append(dict(q=q, n=n, SS_formula=SS, SS_aud=r["SS"], relerr_SS=abs(SS - r["SS"]) / abs(r["SS"]),
                        A0_formula=A0, A0_aud=r["A0"], relerr_A0=abs(A0 - r["A0"]) / r["A0"],
                        SS_plus_binom=SS + math.comb(n + 2, 2), q_times=q * (SS + math.comb(n + 2, 2))))
    return out


def part_e2(q, nmax=60):
    # log G(u) = sum_P log((1 + a_P u^deg)(1 - u^deg/|P|)^2); gamma1 = (log G)'(1), gamma2 = G''(1)/G(1)
    g1 = 0.0
    l2 = 0.0  # (log G)''(1)
    for e in range(1, 400):
        Pq = float(q) ** e
        if Pq > 1e140:
            break
        a = (2 * Pq - 3) / (Pq - 2) ** 2
        pi = npirr(q, e)
        # h(x) = log(1+a x) + 2 log(1 - x/Pq), x = u^e ; d/du = e u^{e-1} h'(x); d2/du2 = e(e-1)u^{e-2}h' + e^2 u^{2e-2} h''
        h1 = a / (1 + a) - 2.0 / (Pq - 1)
        h2 = -a * a / (1 + a) ** 2 - 2.0 / (Pq - 1) ** 2
        g1 += pi * e * h1
        l2 += pi * (e * (e - 1) * h1 + e * e * h2)
    g1_closed = -sum(npirr(q, e) * e / (float(q) ** e - 1) ** 2 for e in range(1, 400) if e * math.log10(q) < 140)
    g2 = l2 + g1 * g1
    beta = 1.5 - g1
    gamma = 1.0 - g1 + g2 / 2.0
    C2 = math.exp(2 * logCq(q))
    res = []
    for n in range(1, nmax + 1):
        ss = -C2 * lcm_sum(q, n)
        res.append((n, ss, ss + (0.5 * n * n + beta * n + gamma)))
    return dict(q=q, gamma1=g1, gamma1_closed=g1_closed, gamma2=g2, beta_q=beta, gamma_q=gamma,
                residuals=[(n, r) for n, s, r in res if n in (5, 10, 15, 20, 30, 40, 50, 60)])


# ---------------- Lema B: d = 3, f = (X-al1)(X-al2)(X-al3) ----------------
def primroot(q):
    fac = [p for p in range(2, q) if (q - 1) % p == 0 and all(p % r for r in range(2, int(p ** .5) + 1))]
    for g in range(2, q):
        if all(pow(g, (q - 1) // p, q) != 1 for p in fac):
            return g


def irreducible_flags(q, n):
    """lam[idx] = Lambda(T^n + sum c_j T^j), idx = sum c_j q^j  (q primo, criba por productos)."""
    sys.path.insert(0, "/home/claude/ariadna/auditorias")
    import importlib.util
    spec = importlib.util.spec_from_file_location("aud", "/home/claude/ariadna/auditorias/R2_ffgold_2026-10-25.py")
    aud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aud)
    irr = aud.irreducibles(q, n)
    lam = np.zeros(q ** n)
    lam[aud.index_of(irr[n][:, :n], q)] = n
    for d in range(1, n):
        if n % d == 0:
            for P in irr[d]:
                f = aud.polypow(P, n // d, q)
                lam[aud.index_of(f[None, :n], q)[0]] = d
    return lam


def part_lemaB(q, roots=(0, 1, 3), ns=(1, 2, 3)):
    g = primroot(q)
    lg = np.full(q, -1, dtype=np.int64)
    x = 1
    for k in range(q - 1):
        lg[x] = k
        x = x * g % q
    m = q - 1
    al = np.array(roots, dtype=np.int64)
    out = dict(q=q, roots=list(roots))
    # caracteres a in (Z/m)^3 ; chi_a(-1) = (-1)^{a1+a2+a3}
    A = np.indices((m, m, m)).reshape(3, -1).T
    s = A.sum(1) % m
    totram = (A != 0).all(1) & (s != 0)
    generic = totram & (A[:, 0] != A[:, 1]) & (A[:, 0] != A[:, 2]) & (A[:, 1] != A[:, 2]) & \
        ((3 * A) % m != s[:, None]).any(1)
    for n in ns:
        lam = irreducible_flags(q, n) if n > 1 else np.ones(q)
        # todos los monicos de grado n: coeficientes c_0..c_{n-1}
        idx = np.arange(q ** n, dtype=np.int64)
        C = np.empty((q ** n, n), dtype=np.int64)
        t = idx.copy()
        for j in range(n):
            C[:, j] = t % q
            t //= q
        vals = []
        for a in al:
            v = np.ones(q ** n, dtype=np.int64) * pow(int(a), n, q)
            for j in range(n):
                v = (v + C[:, j] * pow(int(a), j, q)) % q
            vals.append(v)
        vals = np.stack(vals, 1)
        ok = (vals != 0).all(1) & (lam > 0)
        L = lg[vals[ok]]
        H = np.zeros((m, m, m))
        np.add.at(H, (L[:, 0], L[:, 1], L[:, 2]), lam[ok])
        Psi = np.fft.fftn(H).reshape(-1)  # Psi(n, chi_{-a}); conjunto simetrico
        trn = -Psi / q ** (n / 2.0)  # tr theta^n (para chi_{-a})
        # indice de -a
        negidx = ((-A) % m) @ np.array([m * m, m, 1])
        trn = trn[negidx]  # ahora tr theta_{chi_a}^n
        F2 = np.abs(trn) ** 2
        fields = {"abs_tr_n_sq_minus_main": F2 - min(n, 2)}
        if n == 1:
            fields = {"adjunta(|tr|^2-1)": F2 - 1.0}
        # clases: a ~ a + r(1,1,1); representante con a1 = 0?  usamos r = -a1 => a1' = 0 no generico.
        # agrupamos por (a2-a1, a3-a1) mod m
        key = ((A[:, 1] - A[:, 0]) % m) * m + ((A[:, 2] - A[:, 0]) % m)
        sgn = np.where(A.sum(1) % 2 == 0, 1.0, -1.0)  # chi(-1)
        res_n = {}
        for nm, F in fields.items():
            sel = generic
            k = key[sel]
            # dentro de una clase, chi*rho_r(-1) = chi(-1)(-1)^{3r}: el signo relativo es (-1)^r = (-1)^{a1'-a1}
            num_tw = np.bincount(k, weights=(sgn * F)[sel], minlength=m * m)
            num_un = np.bincount(k, weights=F[sel], minlength=m * m)
            cnt = np.bincount(k, minlength=m * m)
            nz = cnt > 0
            tw = np.abs(num_tw[nz] / cnt[nz])
            un = np.abs(num_un[nz] / cnt[nz])
            res_n[nm] = dict(classes=int(nz.sum()), min_class=int(cnt[nz].min()),
                             max_sqrtq_tw=float(np.sqrt(q) * tw.max()), mean_sqrtq_tw=float(np.sqrt(q) * tw.mean()),
                             max_sqrtq_un=float(np.sqrt(q) * un.max()), mean_sqrtq_un=float(np.sqrt(q) * un.mean()),
                             global_tw_avg=float((sgn * F)[generic].mean()),
                             global_mean_abs_tr_sq=float(F2[totram].mean()))
        out[f"n={n}"] = res_n
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "ss":
        for r in part_ss():
            print(json.dumps(r))
    elif what == "e2":
        print(json.dumps(part_e2(int(sys.argv[2]))))
    elif what == "lemaB":
        q = int(sys.argv[2])
        roots = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0, 1, 3)
        ns = tuple(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1, 2, 3)
        print(json.dumps(part_lemaB(q, roots, ns)))
