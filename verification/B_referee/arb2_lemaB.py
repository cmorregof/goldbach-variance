#!/usr/bin/env python3
"""Arbitro 2 (Fable). Test independiente del Lema B / Prop. W.

Para f = producto de factores irreducibles distintos (grados e_i), B^x = prod F_{q^{e_i}}^x.
Indexo caracteres por (a_1,...,a_r), a_i in Z/(q^{e_i}-1). Calculo L_f(u,chi) = sum_{deg h<=d-1,
(h,f)=1} chi(h) u^{deg h} por DFT de arrays de conteo, obtengo las raices inversas alpha_j,
y Psi(n,chi) = -sum_j alpha_j^n (Newton). Clases de Katz: chi ~ chi rho_Norm, con
rho_Norm indexado por el corrimiento diagonal j*(q^{e_i}-1)/(q-1) en cada factor;
rho_j(-1) = (-1)^j. Para cada clase mido
   tw(C) = |(1/#C) sum_{chi' in C} chi'(-1) F(theta_chi')|,  F = |tr theta^n|^2 - min(n,d-1),
y comparo con la cota del Lema B. Tambien V'(D), W'(D) y las dos mitades chi(-1)=+-1.
Uso: python3 arb2_lemaB.py q "e1,e2,..." nmax
"""
import sys, json, itertools, math
import numpy as np


def primitive_root(q):
    for g in range(2, q):
        seen, x = set(), 1
        for _ in range(q - 1):
            x = x * g % q
            seen.add(x)
        if len(seen) == q - 1:
            return g
    raise ValueError


class Fq2:
    """F_{q^2} = F_q[s]/(s^2 - r), r no residuo. Elementos (x,y) = x + y s."""
    def __init__(self, q):
        self.q = q
        res = {x * x % q for x in range(1, q)}
        self.r = next(r for r in range(2, q) if r not in res)

    def mul(self, a, b):
        q, r = self.q, self.r
        return ((a[0] * b[0] + r * a[1] * b[1]) % q, (a[0] * b[1] + a[1] * b[0]) % q)

    def generator_and_log(self):
        q = self.q
        order = q * q - 1
        for gx in range(q):
            for gy in range(1, q):
                g = (gx, gy)
                x, log = (1, 0), {}
                ok = True
                for k in range(order):
                    if x in log:
                        ok = False
                        break
                    log[x] = k
                    x = self.mul(x, g)
                if ok and len(log) == order:
                    return g, log
        raise ValueError


def build_factor(q, e, g1=None):
    """Devuelve (orden, funcion eval->log, log de -1, paso diagonal, generador de F_q que usa)."""
    if e == 1:
        g = g1
        log = {}
        x = 1
        for k in range(q - 1):
            log[x] = k
            x = x * g % q
        return dict(e=1, order=q - 1, log=log, step=1, g1=g)
    if e == 2:
        F = Fq2(q)
        g2, log2 = F.generator_and_log()
        # N(g2) = g2^{q+1} debe ser el generador g1 de F_q^x que se usa en los factores lineales
        n_g2 = (1, 0)
        for _ in range(q + 1):
            n_g2 = F.mul(n_g2, g2)
        assert n_g2[1] == 0
        return dict(e=2, order=q * q - 1, log=log2, step=q + 1, g1=n_g2[0], F=F)
    raise NotImplementedError


def main(q, degs, nmax):
    d = sum(degs)
    # construir factores; el generador de F_q^x lo fija el factor cuadratico si lo hay
    quad = [build_factor(q, 2) for e in degs if e == 2]
    assert len(quad) <= 1, "a lo sumo un factor cuadratico en este script"
    g1 = quad[0]['g1'] if quad else primitive_root(q)
    lin_roots = list(range(len([e for e in degs if e == 1])))  # raices 0,1,2,...
    factors = []
    for e in degs:
        if e == 1:
            factors.append(build_factor(q, 1, g1))
        else:
            factors.append(quad.pop())
    orders = [F['order'] for F in factors]

    def evals(h):
        """h = tupla de coeficientes (c0..cm). devuelve indices log en cada factor o None si no coprimo."""
        out = []
        li = 0
        for F in factors:
            if F['e'] == 1:
                a = lin_roots[li]; li += 1
                v = 0
                for c in reversed(h):
                    v = (v * a + c) % q
                if v == 0:
                    return None
                out.append(F['log'][v])
            else:
                Fx = F['F']
                v = (0, 0)
                s = (0, 1)
                for c in reversed(h):
                    v = Fx.mul(v, s)
                    v = ((v[0] + c) % q, v[1])
                if v == (0, 0):
                    return None
                out.append(F['log'][v])
        return tuple(out)

    # arrays de conteo para grados 1..d-1
    A = [np.zeros(orders) for _ in range(d)]
    A[0][tuple([0] * len(orders))] = 1.0
    for m in range(1, d):
        for low in itertools.product(range(q), repeat=m):
            h = low + (1,)
            ix = evals(h)
            if ix is not None:
                A[m][ix] += 1.0
    # c_m(chi) para chi = indice k: usamos fftn (chi_k <-> exp(-2 pi i k.x/orden)); simetrico bajo k->-k
    C = [np.fft.fftn(A[m]) for m in range(d)]
    # paridad chi(-1): -1 = g1^{(q-1)/2}; en el factor i, log(-1) = step_i*(q-1)/2 ; chi_k(-1) = exp(-2pi i k_i log_i(-1)/order_i)
    grids = np.meshgrid(*[np.arange(o) for o in orders], indexing='ij')
    sgn_exp = sum(grids[i] * (factors[i]['step'] * (q - 1) // 2) % orders[i] / orders[i] for i in range(len(orders)))
    chi_m1 = np.round(np.real(np.exp(-2j * np.pi * sgn_exp))).astype(int)
    assert set(np.unique(chi_m1)) <= {-1, 1}
    # primitivo: todas las componentes != 0 ; impar (KR): no trivial en F_q^x <=> sum_i k_i*step_i != 0 mod (q-1)
    prim = np.ones(orders, dtype=bool)
    for i in range(len(orders)):
        prim &= grids[i] != 0
    # restriccion a F_q^x: c = g^t tiene log_i(c) = step_i * t, asi que chi_k(c) = exp(-2 pi i t sum_i k_i/(q-1))
    restr = sum(grids[i] for i in range(len(orders))) % (q - 1)
    odd = restr != 0
    TR = prim & odd
    even_prim = prim & ~odd
    nontriv = np.ones(orders, dtype=bool)
    nontriv[tuple([0] * len(orders))] = False

    # raices inversas: L(u) = sum c_m u^m = prod (1 - alpha_j u). Para chi != chi_0 grado <= d-1.
    coef = np.stack([C[m].reshape(-1) for m in range(d)], axis=1)  # (Nchi, d)
    Nchi = coef.shape[0]
    # Newton: e_k = (-1)^k c_k ; p_1 = e_1 ; p_n = sum_{k=1}^{n-1} (-1)^{k-1} e_k p_{n-k} + (-1)^{n-1} n e_n
    e = np.array([(-1) ** k for k in range(d)])[None, :] * coef
    P = {}
    for n in range(1, nmax + 1):
        s = np.zeros(Nchi, dtype=complex)
        for k in range(1, min(n - 1, d - 1) + 1):
            s += (-1) ** (k - 1) * e[:, k] * P[n - k]
        if n <= d - 1:
            s += (-1) ** (n - 1) * n * e[:, n]
        P[n] = s
    Psi = {n: -P[n] for n in P}  # Psi(n,chi) = -sum alpha_j^n

    # control de Weil para TR: |alpha_j| = sqrt q  <=>  |c_{d-1}| = q^{(d-1)/2}
    cd = np.abs(coef[:, d - 1]).reshape(orders)
    weil_ok = np.allclose(cd[TR], q ** ((d - 1) / 2), rtol=1e-8)

    Phi = float(np.prod([o for o in orders]))
    out = dict(q=q, degs=degs, d=d, Phi=Phi, nTR=int(TR.sum()), nEvenPrim=int(even_prim.sum()), weil_ok=bool(weil_ok))
    # comprobacion directa Psi(n,chi) para q pequeno (fuerza bruta sobre monicos de grado n con Lambda)
    if q <= 7 and nmax >= 2:
        n = 2
        # Lambda en grado 2: irreducibles (2) y cuadrados de lineales (1)
        tot = np.zeros(orders, dtype=complex)
        for low in itertools.product(range(q), repeat=2):
            h = low + (1,)
            # factor? h = T^2 + s T + t: reducible iff discriminante residuo
            s_, t_ = low[1], low[0]
            disc = (s_ * s_ - 4 * t_) % q
            if disc == 0:
                lam = 1.0
            elif pow(disc, (q - 1) // 2, q) == 1:
                lam = 0.0
            else:
                lam = 2.0
            ix = evals(h)
            if ix is None or lam == 0:
                continue
            ph = np.zeros(orders)
            ph[ix] = lam
            tot += np.fft.fftn(ph)
        out['brute_Psi2_check'] = float(np.max(np.abs(tot.reshape(-1) - Psi[2])))

    # V', W' y mitades, en unidades de q^n
    res = {}
    for n in range(1, nmax + 1):
        ps2 = np.abs(Psi[n]) ** 2 / q ** n
        ps2 = ps2.reshape(orders)
        Vp = ps2[nontriv].sum() / Phi
        Wp = (chi_m1 * ps2)[nontriv].sum() / Phi
        plus = ps2[nontriv & (chi_m1 == 1)].sum() / Phi
        minus = ps2[nontriv & (chi_m1 == -1)].sum() / Phi
        trTR = ps2[TR].mean()
        twTR = (chi_m1 * ps2)[TR].mean()
        res[n] = dict(Vp=Vp, Wp=Wp, Wp_times_qm1=Wp * (q - 1), half_plus=plus, half_minus=minus,
                      meanTR_abs_tr2=trTR, twistedTR=twTR, sqrtq_twistedTR=twTR * math.sqrt(q))
    out['VW'] = res

    # clases de Katz entre los TR genericos (genericos: componentes distintas en la extension comun;
    # aqui solo filtro 'TR' y dejo que la clase contenga <= 2d malos, como en la Prop. W del autor).
    # Representante canonico: restar j tal que la primera componente lineal sea 0 ... usamos orbitas del
    # corrimiento diagonal (k_i + j*step_i).
    steps = np.array([F['step'] for F in factors])
    idx_all = np.array(list(np.ndindex(*orders)))
    seen = np.zeros(Nchi, dtype=bool)
    lin_index = np.ravel_multi_index
    cls_stats = {n: [] for n in range(1, nmax + 1)}
    ncls = 0
    min_class = 10 ** 9
    for flat in range(Nchi):
        if seen[flat] or not TR.reshape(-1)[flat]:
            continue
        k = idx_all[flat]
        members = []
        for j in range(q - 1):
            kk = tuple((k + j * steps) % np.array(orders))
            fl = lin_index(kk, orders)
            if TR.reshape(-1)[fl]:
                members.append((fl, j))
            seen[fl] = True
        ncls += 1
        min_class = min(min_class, len(members))
        fls = np.array([m[0] for m in members])
        sg = chi_m1.reshape(-1)[fls]
        for n in range(1, nmax + 1):
            Fv = np.abs(Psi[n][fls]) ** 2 / q ** n - min(n, d - 1)
            tw = abs(np.sum(sg * Fv)) / len(fls)
            un = abs(np.sum(Fv)) / len(fls)
            cls_stats[n].append((tw, un))
    out['classes'] = ncls
    out['min_class'] = min_class
    cs = {}
    for n in range(1, nmax + 1):
        arr = np.array(cls_stats[n])
        a, b = n, n
        rk_bound = (a + b) * d * (d - 1) ** (a + b - 1)
        # cota del Lema B para el promedio por clase de sum_Lambda m_Lambda Tr Lambda:
        # <= [ (q-1)/sqrt q * rk_bound + 2 d * dim ] * sum|m_Lambda| / #Good ; uso sum|m|dim <= (d-1)^{2n}
        bound_sqrtq = ((d - 1) ** (2 * n)) * (rk_bound + 2 * d * math.sqrt(q) / (q - 1))
        cs[n] = dict(max_sqrtq_tw=float(arr[:, 0].max() * math.sqrt(q)), mean_sqrtq_tw=float(arr[:, 0].mean() * math.sqrt(q)),
                     max_sqrtq_un=float(arr[:, 1].max() * math.sqrt(q)), mean_sqrtq_un=float(arr[:, 1].mean() * math.sqrt(q)),
                     lemmaB_bound_sqrtq_times_mean=float(bound_sqrtq))
    out['class_stats'] = cs
    print(json.dumps(out, default=float))
    return out


if __name__ == '__main__':
    q = int(sys.argv[1]); degs = [int(x) for x in sys.argv[2].split(',')]; nmax = int(sys.argv[3])
    main(q, degs, nmax)
