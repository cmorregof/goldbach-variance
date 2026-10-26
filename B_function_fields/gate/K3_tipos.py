"""K3 (run 3, gate R2): contabilidad determinista de X por tipos de factorizacion.

Calcula, para q impar (cualquier potencia de primo: solo entra el numero q) y n fijo,
la prediccion "determinista + medias de matrices aleatorias" de
    X_pp/q^{2n},  X_G/q^{2n},  SS/q^{2n}
usando la expansion exacta  S_q(x) = sum_{D | x, D libre de cuadrados} w_D,
w_D = C_q / prod_{P|D}(|P|-2), y la descomposicion por caracteres de
    V(D) = sum_{h in G, D|h} psi2(h)  = sum_r Psi(D,r)^2,
    W(D) = sum_{g, D|g} R(g)          = sum_r Psi(D,r) Psi(D,-r).
Hipotesis usadas en la parte aleatoria (solo para V', W'):
  (KR)  media de |Psi(n,chi)|^2 sobre chi primitivos impares mod D*: q^n min(n, deg D* - 1)
  (Hev) media sobre primitivos pares: 1 + q^n min(n, deg D* - 2)
  (Htw) las subfamilias impares con chi(-1)=+1 y chi(-1)=-1 tienen la misma media (KR).
Todo es exacto (racional en q) salvo C_q, que se evalua con mpmath a 80 digitos.
Salida: JSON por linea.
"""
import sys, json, itertools
from math import comb
import mpmath as mp

mp.mp.dps = 80


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


def C_q(q, emax=None):
    # C_q = prod_P (1 - 1/(|P|-1)^2) ; log = sum_e pi(e) log1p(-1/(q^e-1)^2)
    s = mp.mpf(0)
    e = 1
    while True:
        t = mp.mpf(npirr(q, e)) * mp.log1p(-1 / (mp.mpf(q) ** e - 1) ** 2)
        s += t
        if abs(t) < mp.mpf(10) ** (-85) and e > 3:
            break
        e += 1
    return mp.e ** s


def partitions_types(d, emax):
    """multisets of prime degrees summing to d: dict e->k (k primes of degree e)."""
    out = []

    def rec(rem, e, cur):
        if rem == 0:
            out.append(dict(cur))
            return
        if e > rem:
            return
        # choose k primes of degree e
        for k in range(0, rem // e + 1):
            if k:
                cur[e] = k
            elif e in cur:
                del cur[e]
            rec(rem - k * e, e + 1, cur)
        if e in cur:
            del cur[e]

    rec(d, 1, {})
    return out


def type_count(q, ty):
    c = 1
    for e, k in ty.items():
        c *= comb(npirr(q, e), k)
    return c


def rmt_parts(q, n, degs):
    """degs: list of prime degrees (with repetition) of squarefree D.
    returns (Phi, Phi2, a, b, bprime, Vp_rmt*Phi, Wp_rmt*Phi) as exact ints/Fractions-like (mp)."""
    N = mp.mpf(q) ** n
    Phi = 1
    Phi2 = 1
    a = 0
    b = 0
    for e in degs:
        Phi *= q ** e - 1
        Phi2 *= q ** e - 2
        if n % e == 0:
            a += e
            b += e * e
    bprime = (degs[0] ** 2) if (len(degs) == 1 and n % degs[0] == 0) else 0
    V = mp.mpf(0)
    W = mp.mpf(0)
    k = len(degs)
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            dS = sum(degs[i] for i in S)
            PhiS = 1
            for i in S:
                PhiS *= q ** degs[i] - 2
            sgn = (-1) ** r
            Nev = mp.mpf(PhiS + sgn * (q - 2)) / (q - 1)
            Nmin = mp.mpf(PhiS - sgn) / 2
            Nodd = PhiS - Nev
            Nplus = Nodd - Nmin
            m_odd = N * min(n, dS - 1)
            m_even = (1 + N * min(n, dS - 2)) if dS >= 2 else mp.mpf(0)
            V += Nodd * m_odd + Nev * m_even
            W += (Nplus - Nmin) * m_odd + Nev * m_even
    return Phi, Phi2, a, b, bprime, V, W


def series_S(q, n):
    """S_m = sum_{x monic deg m} S_q(x)^2 / C_q^2  for m=0..n, exact rationals via mp."""
    # prod_e (1 + (r_e^2-1) u^e)^{pi(e)} / (1 - q u)
    poly = [mp.mpf(0)] * (n + 1)
    poly[0] = mp.mpf(1)
    for e in range(1, n + 1):
        pe = npirr(q, e)
        r2m1 = (mp.mpf(q ** e - 1) / (q ** e - 2)) ** 2 - 1
        fac = [mp.mpf(0)] * (n + 1)
        for kk in range(0, n // e + 1):
            fac[kk * e] = comb(pe, kk) * r2m1 ** kk
        new = [mp.mpf(0)] * (n + 1)
        for i in range(n + 1):
            if poly[i] == 0:
                continue
            for j in range(0, n + 1 - i):
                if fac[j] != 0:
                    new[i + j] += poly[i] * fac[j]
        poly = new
    S = [mp.mpf(0)] * (n + 1)
    for m in range(n + 1):
        S[m] = sum(poly[i] * mp.mpf(q) ** (m - i) for i in range(m + 1))
    return S


def predict(q, n, detail=False):
    N = mp.mpf(q) ** n
    C = C_q(q)
    psi0 = sum(e * e * npirr(q, e) for e in range(1, n + 1) if n % e == 0)
    S = series_S(q, n)
    sum_s2_h = C ** 2 * (q - 1) * sum(S[m] for m in range(n))
    sum_s2_g = C ** 2 * S[n]
    # D-sums by type
    pp = {k: mp.mpf(0) for k in ("main", "a_lin", "rest_main", "b", "Vp", "psi0")}
    gg = {k: mp.mpf(0) for k in ("main", "b", "Wp")}
    per_d = []
    for d in range(0, n + 1):
        acc = dict(main=mp.mpf(0), Vp=mp.mpf(0), Wp=mp.mpf(0), wsum=mp.mpf(0), cnt=0)
        for ty in partitions_types(d, n):
            degs = sorted(sum(([e] * k for e, k in ty.items()), []))
            cnt = type_count(q, ty)
            if cnt == 0:
                continue
            if d == 0:
                Phi, Phi2, a, b, bp, V, W = 1, 1, 0, 0, 0, mp.mpf(0), mp.mpf(0)
            else:
                Phi, Phi2, a, b, bp, V, W = rmt_parts(q, n, degs)
            w = C / Phi2
            main = (N - a) ** 2 / Phi
            acc["cnt"] += cnt
            acc["wsum"] += cnt * w
            acc["main"] += cnt * w * main
            acc["Vp"] += cnt * w * V / Phi
            acc["Wp"] += cnt * w * W / Phi
            if d <= n - 1:
                pp["main"] += cnt * w * main
                pp["b"] += cnt * w * b
                pp["Vp"] += cnt * w * V / Phi
                pp["psi0"] += cnt * w * psi0
            gg["main"] += cnt * w * main
            gg["b"] += cnt * w * bp
            gg["Wp"] += cnt * w * W / Phi
        per_d.append(dict(d=d, wsum=float(acc["wsum"]), Vp_over_N=float(acc["Vp"] / N),
                          Wp_over_N=float(acc["Wp"] / N)))
    Xpp = 2 * ((pp["main"] + pp["b"] + pp["Vp"] - pp["psi0"]) / N - sum_s2_h)
    XG = -2 * ((gg["main"] + gg["b"] + gg["Wp"]) / N - sum_s2_g)
    SS = sum_s2_h - sum_s2_g
    # deterministic pieces (no character part): what X would be with V'=W'=0
    Xpp_noV = 2 * ((pp["main"] + pp["b"] - pp["psi0"]) / N - sum_s2_h)
    XG_noW = -2 * ((gg["main"] + gg["b"]) / N - sum_s2_g)
    out = dict(q=q, n=n, Xpp_det=float(Xpp), XG_det=float(XG), X_det=float(Xpp + XG), SS=float(SS),
               A0=float(mp.mpf(psi0) ** 2 / N ** 2),
               Xpp_sinV=float(Xpp_noV), XG_sinW=float(XG_noW), C_q=float(C),
               ppVp_over_N=float(pp["Vp"] / N), ggWp_over_N=float(gg["Wp"] / N))
    if detail:
        out["per_d"] = per_d
    return out


if __name__ == "__main__":
    n = int(sys.argv[1])
    qs = [int(x) for x in sys.argv[2:]]
    for q in qs:
        print(json.dumps(predict(q, n, detail=True)))
        sys.stdout.flush()
