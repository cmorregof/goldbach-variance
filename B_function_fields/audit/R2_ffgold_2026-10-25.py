"""Exact Parseval decomposition VarG = A0 + Ep + SS + X for the function-field box.

Setting (q an odd prime, n fixed):
  A = monic polynomials of degree n over F_q, written f = T^n + u, deg u < n.
  lam(u) = Lambda(T^n + u).
  psi2(h) = sum_u lam(u) lam(u+h),  h in G = F_q[T]_{<n}   (prime pairs)
  R(v)    = sum_u lam(u) lam(v-u),  target g = 2T^n + v       (Goldbach)
  M(h) = S(h) q^n, MG(g) = S(g) q^n, S = singular series
  S(x) = C_q prod_{P|x} (|P|-1)/(|P|-2),  C_q = prod_P (1 - 1/(|P|-1)^2).
"""
import sys, math, json
import numpy as np


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


def all_polys(q, m):
    """coefficient array (q^m, m) of all polys of degree < m, row i <-> index i = sum c_j q^j"""
    idx = np.arange(q ** m, dtype=np.int64)
    out = np.empty((q ** m, m), dtype=np.int64)
    for j in range(m):
        out[:, j] = idx % q
        idx //= q
    return out


def index_of(coeffs, q):
    """coeffs (N, m) -> index sum c_j q^j"""
    m = coeffs.shape[1]
    w = q ** np.arange(m, dtype=np.int64)
    return coeffs @ w


def mul(Pc, Mc, q):
    """Pc: (d+1,) coefficients of P (low->high); Mc: (N, k) coefficient rows. returns (N, d+k)"""
    d1 = len(Pc)
    N, k = Mc.shape
    out = np.zeros((N, d1 + k - 1), dtype=np.int64)
    for i, c in enumerate(Pc):
        if c:
            out[:, i:i + k] += c * Mc
    return out % q


def irreducibles(q, nmax):
    """dict d -> array (pi(d), d+1) of monic irreducibles of degree d (coeffs low->high)."""
    irr = {}
    for m in range(1, nmax + 1):
        isp = np.ones(q ** m, dtype=bool)
        for d in range(1, m // 2 + 1):
            k = m - d
            G = all_polys(q, k)
            Mc = np.concatenate([G, np.ones((q ** k, 1), dtype=np.int64)], axis=1)  # monic deg k
            for P in irr[d]:
                prod = mul(P, Mc, q)  # (q^k, m+1), monic
                isp[index_of(prod[:, :m], q)] = False
        U = all_polys(q, m)[isp]
        irr[m] = np.concatenate([U, np.ones((U.shape[0], 1), dtype=np.int64)], axis=1)
        assert irr[m].shape[0] == npirr(q, m), (m, irr[m].shape[0], npirr(q, m))
    return irr


def polypow(P, k, q):
    r = np.array([1], dtype=np.int64)
    for _ in range(k):
        r = np.convolve(r, P) % q
    return r


def run(q, n, Dmax=80):
    irr = irreducibles(q, n)
    Qn = q ** n
    lam = np.zeros(Qn)
    lam[index_of(irr[n][:, :n], q)] = n
    for d in range(1, n):
        if n % d == 0:
            for P in irr[d]:
                f = polypow(P, n // d, q)
                lam[index_of(f[None, :n], q)[0]] = d
    # constant C_q
    logC = sum(npirr(q, d) * math.log1p(-1.0 / float(q ** d - 1) ** 2) for d in range(1, Dmax) if q ** d < 10 ** 150)
    # singular series on h in G (deg<n), and on g = 2T^n + v
    logS_h = np.zeros(Qn)
    logS_g = np.zeros(Qn)
    for d in range(1, n + 1):
        fac = math.log((q ** d - 1) / (q ** d - 2))
        if d <= n - 1:
            Mh = all_polys(q, n - d)  # all m with deg < n-d
            for P in irr[d]:
                pr = mul(P, Mh, q)  # (q^{n-d}, n)
                logS_h[index_of(pr[:, :n], q)] += fac
        k = n - d
        Mg = np.concatenate([all_polys(q, k), 2 * np.ones((q ** k, 1), dtype=np.int64)], axis=1)
        for P in irr[d]:
            pr = mul(P, Mg, q)  # (q^k, n+1), lc 2
            logS_g[index_of(pr[:, :n], q)] += fac
    logS_h[0] = 0.0  # h = 0 excluded
    Sh = np.exp(logC + logS_h)
    Sg = np.exp(logC + logS_g)
    # FFT over (Z/q)^n
    shape = (q,) * n
    # index = sum c_j q^j ; C-order reshape puts c_{n-1} on axis 0: consistent for +,-
    L = lam.reshape(shape)
    F = np.fft.fftn(L)
    psi2 = np.rint(np.real(np.fft.ifftn(np.abs(F) ** 2))).reshape(-1)
    R = np.rint(np.real(np.fft.ifftn(F * F))).reshape(-1)
    # psi2(h) = sum_u lam(u) lam(u+h)?  ifft(|F|^2)[h] = sum_u lam(u+h) lam(u). yes.
    A0 = psi2[0] ** 2
    Mh = Sh * Qn
    Eh = psi2 - Mh
    Eh[0] = 0.0
    Mh[0] = 0.0
    Ep = float(np.sum(Eh ** 2))
    MG = Sg * Qn
    dG = R - MG
    VarG = float(np.sum(dG ** 2))
    SS = float(np.sum(Mh ** 2) - np.sum(MG ** 2))
    X = float(2 * np.sum(Mh * Eh) - 2 * np.sum(MG * dG))
    Xpp = float(2 * np.sum(Mh * Eh)); XG = float(-2 * np.sum(MG * dG))
    sc = float(Qn) ** 2
    out = dict(q=q, n=n, sum_lam=float(lam.sum()), A0=A0 / sc, Ep=Ep / sc, SS=SS / sc, X=X / sc,
               Xpp=Xpp / sc, XG=XG / sc,
               VarG=VarG / sc, VarG_minus_Ep=(VarG - Ep) / sc,
               identity_relerr=abs(VarG - (A0 + Ep + SS + X)) / VarG,
               # alternative: no singular series (main term q^n flat)
               Ep_flat=float(np.sum((psi2[1:] - Qn) ** 2)) / sc,
               VarG_flat=float(np.sum((R - Qn) ** 2)) / sc,
               meanS_h=float(Sh[1:].mean()), meanS_g=float(Sg.mean()))
    return out


if __name__ == "__main__":
    q, n = int(sys.argv[1]), int(sys.argv[2])
    print(json.dumps(run(q, n)))
