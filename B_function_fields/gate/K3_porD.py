"""K3 (run 3, gate R2): verificacion exacta por modulo D.

Para q primo impar y n fijo calcula, para TODO D monico libre de cuadrados con deg D <= n,
    V(D) = sum_{h in G, D|h} psi2(h)          (deg D <= n-1)
    W(D) = sum_{g in 2T^n+G, D|g} R(g)        (deg D <= n)
y sus partes de caracteres no triviales
    V'(D) = V(D) - (q^n - a_D)^2/Phi(D) - b_D        = (1/Phi) sum_{chi != chi0} |Psi(n,chi)|^2
    W'(D) = W(D) - (q^n - a_D)^2/Phi(D) - b'_D       = (1/Phi) sum_{chi != chi0} chi(-1) |Psi(n,chi)|^2
Comprueba las identidades exactas
    sum_{h!=0} S(h) psi2(h) = sum_{deg D<=n-1} w_D (V(D) - psi2(0)),
    sum_g S(g) R(g)         = sum_{deg D<=n}   w_D W(D),
y compara, grado a grado, con las medias RMT (KR, Htw) del modulo K3_tipos.
Usa codigo propio (no importa el script de la auditoria).
"""
import sys, json, math, time
import numpy as np

sys.dont_write_bytecode = True
import importlib.util
_spec = importlib.util.spec_from_file_location("K3t", __file__.replace("K3_porD.py", "K3_tipos.py"))
K3t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K3t)


def npirr(q, d):
    return K3t.npirr(q, d)


def all_polys(q, m):
    idx = np.arange(q ** m, dtype=np.int64)
    out = np.empty((q ** m, m), dtype=np.int64)
    for j in range(m):
        out[:, j] = idx % q
        idx //= q
    return out


def index_of(coeffs, q):
    m = coeffs.shape[1]
    w = q ** np.arange(m, dtype=np.int64)
    return coeffs @ w


def mulbatch(A, B, q):
    """A: (a, da) coeff rows, B: (b, db) coeff rows -> (a*b, da+db-1) products mod q (row-major a-major)."""
    a, da = A.shape
    b, db = B.shape
    out = np.zeros((a, b, da + db - 1), dtype=np.int64)
    for t in range(da):
        out[:, :, t:t + db] += A[:, t, None, None] * B[None, :, :]
    return (out % q).reshape(a * b, da + db - 1)


def chunked_products(A, B, q, maxrows=1_500_000):
    """yield (row_slice_of_A, products) in chunks."""
    b = B.shape[0]
    step = max(1, maxrows // max(1, b))
    for s in range(0, A.shape[0], step):
        yield s, mulbatch(A[s:s + step], B, q)


def irreducibles(q, nmax):
    irr = {}
    for m in range(1, nmax + 1):
        isp = np.ones(q ** m, dtype=bool)
        for d in range(1, m // 2 + 1):
            Mc = np.concatenate([all_polys(q, m - d), np.ones((q ** (m - d), 1), dtype=np.int64)], axis=1)
            for s, pr in chunked_products(irr[d], Mc, q):
                isp[index_of(pr[:, :m], q)] = False
        U = all_polys(q, m)[isp]
        irr[m] = np.concatenate([U, np.ones((U.shape[0], 1), dtype=np.int64)], axis=1)
        assert irr[m].shape[0] == npirr(q, m)
    return irr


def sieve_degree(q, n, d, irr):
    """For monic D of degree d (indexed by lower d coeffs): squarefree flag, Phi, Phi2, a, b, type code."""
    M = q ** d
    sqf = np.ones(M, dtype=bool)
    Phi = np.ones(M)
    Phi2 = np.ones(M)
    a = np.zeros(M)
    b = np.zeros(M)
    code = np.zeros(M, dtype=np.int64)
    base = n + 2
    if d == 0:
        return sqf, Phi, Phi2, a, b, code
    for e in range(1, d + 1):
        k = d - e
        Mc = np.concatenate([all_polys(q, k), np.ones((q ** k, 1), dtype=np.int64)], axis=1) if k > 0 else np.ones((1, 1), dtype=np.int64)
        cnt = np.zeros(M, dtype=np.int64)
        for s, pr in chunked_products(irr[e], Mc, q):
            idx = index_of(pr[:, :d], q)
            cnt += np.bincount(idx, minlength=M)
        Phi *= (float(q ** e - 1)) ** cnt
        Phi2 *= (float(q ** e - 2)) ** cnt
        if n % e == 0:
            a += e * cnt
            b += e * e * cnt
        code += cnt * base ** (e - 1)
        if 2 * e <= d:
            k2 = d - 2 * e
            P2 = np.array([np.convolve(P, P) % q for P in irr[e]])
            Mc2 = np.concatenate([all_polys(q, k2), np.ones((q ** k2, 1), dtype=np.int64)], axis=1) if k2 > 0 else np.ones((1, 1), dtype=np.int64)
            for s, pr in chunked_products(P2, Mc2, q):
                sqf[index_of(pr[:, :d], q)] = False
    return sqf, Phi, Phi2, a, b, code


def decode(code, n):
    base = n + 2
    degs = []
    e = 1
    c = int(code)
    while c:
        k = c % base
        degs += [e] * k
        c //= base
        e += 1
    return degs


def run(q, n):
    t0 = time.time()
    N = float(q ** n)
    irr = irreducibles(q, n)
    # Lambda on monic deg n
    lam = np.zeros(q ** n)
    lam[index_of(irr[n][:, :n], q)] = n
    for dd in range(1, n):
        if n % dd == 0:
            for P in irr[dd]:
                f = np.array([1], dtype=np.int64)
                for _ in range(n // dd):
                    f = np.convolve(f, P) % q
                lam[index_of(f[None, :n], q)[0]] = dd
    assert abs(lam.sum() - N) < 0.5
    F = np.fft.fftn(lam.reshape((q,) * n))
    psi2 = np.rint(np.real(np.fft.ifftn(np.abs(F) ** 2))).reshape(-1)
    R = np.rint(np.real(np.fft.ifftn(F * F))).reshape(-1)
    del F
    psi0 = psi2[0]
    Cq = float(K3t.C_q(q))
    # singular series arrays via D-expansion (also gives independent check of S_q)
    S_h = np.zeros(q ** n)
    S_g = np.zeros(q ** n)
    per_d = []
    tot = dict(sum_w_V=0.0, sum_w_W=0.0)
    rmt_cache = {}
    for d in range(0, n + 1):
        sqf, Phi, Phi2, a, b, code = sieve_degree(q, n, d, irr)
        ids = np.nonzero(sqf)[0]
        Dc = np.concatenate([all_polys(q, d)[ids], np.ones((len(ids), 1), dtype=np.int64)], axis=1) if d > 0 else np.ones((1, 1), dtype=np.int64)
        w = Cq / Phi2[ids]
        main = (N - a[ids]) ** 2 / Phi[ids]
        isprime = (code[ids] == (n + 2) ** (d - 1)) if d > 0 else np.zeros(1, dtype=bool)
        bprime = np.where(isprime & (n % max(d, 1) == 0) & (d > 0), float(d * d), 0.0)
        # W(D): g = D*(2T^{n-d} + m'), m' deg < n-d
        k = n - d
        Mg = np.concatenate([all_polys(q, k), 2 * np.ones((q ** k, 1), dtype=np.int64)], axis=1) if k > 0 else 2 * np.ones((1, 1), dtype=np.int64)
        Wv = np.zeros(len(ids))
        for s, pr in chunked_products(Dc, Mg, q):
            gi = index_of(pr[:, :n], q)
            nr = pr.shape[0] // Mg.shape[0]
            Wv[s:s + nr] = R[gi].reshape(nr, Mg.shape[0]).sum(1)
            S_g += np.bincount(gi, weights=np.repeat(w[s:s + nr], Mg.shape[0]), minlength=S_g.size)
        Wp = Wv - main - bprime
        rec = dict(d=d, nD=int(len(ids)))
        if d <= n - 1:
            Mh = all_polys(q, k)  # all m with deg < n-d (incl. 0)
            Vv = np.zeros(len(ids))
            for s, pr in chunked_products(Dc, Mh, q):
                hi = index_of(pr[:, :n], q)
                nr = pr.shape[0] // Mh.shape[0]
                Vv[s:s + nr] = psi2[hi].reshape(nr, Mh.shape[0]).sum(1)
                S_h += np.bincount(hi, weights=np.repeat(w[s:s + nr], Mh.shape[0]), minlength=S_h.size)
            Vp = Vv - main - b[ids]
            tot["sum_w_V"] += float(np.sum(w * (Vv - psi0)))
            rec.update(Vp_mean_over_N=float(Vp.mean() / N), Vp_w_over_N=float(np.sum(w * Vp) / N),
                       chi_minus_mean_over_N=float(((Vp - Wp) / 2).mean() / N),
                       chi_plus_mean_over_N=float(((Vp + Wp) / 2).mean() / N))
        tot["sum_w_W"] += float(np.sum(w * Wv))
        rec.update(Wp_mean_over_N=float(Wp.mean() / N), Wp_w_over_N=float(np.sum(w * Wp) / N),
                   Wp_absmax_over_N=float(np.abs(Wp).max() / N))
        # RMT predictions per type
        Vr = np.zeros(len(ids))
        Wr = np.zeros(len(ids))
        if d > 0:
            for c in np.unique(code[ids]):
                c = int(c)
                if c not in rmt_cache:
                    degs = decode(c, n)
                    Ph, Ph2, aa, bb, bp, V_, W_ = K3t.rmt_parts(q, n, degs)
                    rmt_cache[c] = (float(V_ / Ph), float(W_ / Ph))
                sel = code[ids] == c
                Vr[sel], Wr[sel] = rmt_cache[c]
        if d <= n - 1:
            rec.update(Vp_rmt_mean_over_N=float(Vr.mean() / N), Vp_rmt_w_over_N=float(np.sum(w * Vr) / N))
        rec.update(Wp_rmt_mean_over_N=float(Wr.mean() / N), Wp_rmt_w_over_N=float(np.sum(w * Wr) / N))
        per_d.append(rec)
    # identity checks
    S_h[0] = 0.0
    lhs_h = float(np.sum(S_h * psi2) - S_h[0] * psi2[0])
    lhs_g = float(np.sum(S_g * R))
    # S_q check vs Euler product definition: mean values
    Mh_ = S_h * N
    Mh_[0] = 0
    Eh = psi2 - Mh_
    Eh[0] = 0
    Xpp = float(2 * np.sum(Mh_ * Eh)) / N ** 2
    MG = S_g * N
    XG = float(-2 * np.sum(MG * (R - MG))) / N ** 2
    SS = float(np.sum(Mh_ ** 2) - np.sum(MG ** 2)) / N ** 2
    out = dict(q=q, n=n, C_q=Cq,
               ident_pp_relerr=abs(lhs_h - tot["sum_w_V"]) / abs(lhs_h),
               ident_G_relerr=abs(lhs_g - tot["sum_w_W"]) / abs(lhs_g),
               Xpp=Xpp, XG=XG, SS=SS, meanS_h=float(S_h[1:].mean()), meanS_g=float(S_g.mean()),
               per_d=per_d, secs=round(time.time() - t0, 1))
    return out


if __name__ == "__main__":
    q, n = int(sys.argv[1]), int(sys.argv[2])
    print(json.dumps(run(q, n)))
