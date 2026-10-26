#!/usr/bin/env python3
"""Arbitro 2 (Fable). Fuerza bruta independiente de A0, E_p, SS, X_pp, X_G, Var_G
en F_q[T], q primo, con el marco de la auditoria (§0):
  M_n monicos de grado n, G = F_q[T]_{<n}, Lambda(P^k)=deg P,
  psi_2(h) = sum_f Lambda(f)Lambda(f+h), R(g) = sum_f Lambda(f)Lambda(g-f) con g in 2T^n+G,
  S_q(x) = C_q prod_{P|x} (|P|-1)/(|P|-2).
Codigo escrito desde cero (no reutiliza R2_ffgold ni K3_*). Uso: python3 arb2_bruteforce.py q n
"""
import sys, json, math
import numpy as np


def mobius(m):
    mu, p = 1, 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            mu = -mu
        p += 1
    if m > 1:
        mu = -mu
    return mu


def Cq(q, emax=60):
    s = 0.0
    for e in range(1, emax + 1):
        pi = sum(mobius(e // d) * q ** d for d in range(1, e + 1) if e % d == 0) // e
        s += pi * math.log1p(-1.0 / (q ** e - 1) ** 2)
    return math.exp(s)


# ---- polinomios: tupla de coeficientes (c0,c1,...,c_deg) con c_deg != 0 (salvo 0) ----
def pmul(a, b, q):
    if not a or not b:
        return ()
    c = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                c[i + j] = (c[i + j] + ai * bj) % q
    return tuple(c)


def all_polys(q, deg):
    """todos los polinomios de grado exactamente deg (coef principal != 0)."""
    import itertools
    for low in itertools.product(range(q), repeat=deg):
        for lc in range(1, q):
            yield low + (lc,)


def monics(q, deg):
    import itertools
    if deg == 0:
        yield (1,)
        return
    for low in itertools.product(range(q), repeat=deg):
        yield low + (1,)


def irreducibles(q, nmax):
    """monicos irreducibles de grado <= nmax por criba."""
    irr = {e: [] for e in range(1, nmax + 1)}
    reducible = set()
    for e in range(1, nmax + 1):
        for f in monics(q, e):
            if f not in reducible:
                irr[e].append(f)
        # marcar productos de grado <= nmax que involucran grado e
        for a in range(1, e + 1):
            for P in irr[a]:
                for b in range(a, nmax - a + 1):
                    if b > e:
                        break
                    for Q in irr[b]:
                        reducible.add(pmul(P, Q, q))
        # productos con mas de dos factores quedan cubiertos porque los reducibles
        # de grado d con factor de grado a tienen cofactor monico de grado d-a, y
        # marcamos P*cofactor para todo monico cofactor via la pasada siguiente:
    # la criba anterior solo marca productos de irreducibles; completar con todos los monicos
    reducible = set()
    for a in range(1, nmax):
        for P in irr[a]:
            for b in range(1, nmax - a + 1):
                for m in monics(q, b):
                    reducible.add(pmul(P, m, q))
    irr2 = {e: [f for f in monics(q, e) if f not in reducible] for e in range(1, nmax + 1)}
    return irr2


def idx_low(f, q, n):
    """indice en (Z/q)^n de la parte baja (grado < n) de f, interpretada como h in G."""
    i = 0
    for k in range(n - 1, -1, -1):
        i = i * q + (f[k] if k < len(f) else 0)
    return i


def main(q, n):
    N = q ** n
    cq = Cq(q)
    irr = irreducibles(q, n)
    pi = {e: len(irr[e]) for e in irr}
    # control: sum e*pi(e) sobre e|n = q^n
    assert sum(e * pi[e] for e in irr if n % e == 0) == N, pi

    # Lambda sobre M_n como array en (Z/q)^n (indice de la parte baja)
    Lam = np.zeros(N, dtype=np.float64)
    for e in irr:
        if n % e:
            continue
        k = n // e
        for P in irr[e]:
            f = (1,)
            for _ in range(k):
                f = pmul(f, P, q)
            assert len(f) == n + 1 and f[n] == 1
            Lam[idx_low(f, q, n)] += e
    assert abs(Lam.sum() - N) < 1e-9
    shape = (q,) * n
    L = Lam.reshape(shape)
    F = np.fft.fftn(L)
    psi2 = np.real(np.fft.ifftn(np.abs(F) ** 2)).reshape(-1)  # psi2[h] = sum_f Lam(f) Lam(f+h)
    R = np.real(np.fft.ifftn(F * F)).reshape(-1)             # R[h]  = sum_f Lam(f) Lam(h-f)
    # orientacion del indice: autocorrelacion por ifft(|F|^2) da c[h]=sum_f a(f) a(f+h) o a(f-h);
    # psi2 es par en h (h -> -h) asi que no importa.

    # serie singular en G\{0} (grado < n) y en 2T^n + G
    S_G = np.full(N, cq)
    S_g = np.full(N, cq)
    for e in irr:
        w = (q ** e - 1.0) / (q ** e - 2.0)
        for P in irr[e]:
            # multiplos en G: P*m con deg m <= n-1-e, cualquier m != 0
            for dm in range(0, n - e):
                for m in all_polys(q, dm) if dm > 0 else [(c,) for c in range(1, q)]:
                    S_G[idx_low(pmul(P, m, q), q, n)] *= w
            # multiplos de grado n con coef principal 2: m de grado n-e con coef principal 2
            dm = n - e
            import itertools
            for low in itertools.product(range(q), repeat=dm):
                m = low + (2 % q,)
                x = pmul(P, m, q)
                assert len(x) == n + 1 and x[n] == 2 % q
                S_g[idx_low(x, q, n)] *= w
    S_G[0] = 0.0  # h=0 excluido

    psi0 = psi2[0]
    A0 = psi0 ** 2
    mask = np.arange(N) != 0
    E = psi2 - S_G * N
    delta = R - S_g * N
    Ep = np.sum(E[mask] ** 2)
    VarG = np.sum(delta ** 2)
    SS = N * N * (np.sum(S_G[mask] ** 2) - np.sum(S_g ** 2))
    Xpp = 2 * N * np.sum(S_G[mask] * E[mask])
    XG = -2 * N * np.sum(S_g * delta)
    X = VarG - Ep - A0 - SS
    parseval = np.sum(R ** 2) - np.sum(psi2 ** 2)
    out = dict(q=q, n=n, N=N, Cq=cq, A0=A0 / N ** 2, Ep=Ep / N ** 2, SS=SS / N ** 2,
               Xpp=Xpp / N ** 2, XG=XG / N ** 2, X=X / N ** 2, VarG=VarG / N ** 2,
               VarG_minus_Ep=(VarG - Ep) / N ** 2,
               identity_relerr=abs(X - Xpp - XG) / max(abs(X), 1e-300),
               parseval_relerr=abs(parseval) / np.sum(psi2 ** 2),
               target_binom=math.comb(n - 1, 2), pi=pi)
    print(json.dumps(out))
    return out


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
