#!/usr/bin/env python3
"""Arbitro 2 (Fable). Comprobacion independiente de SS (Lema 4.2), de la Prop. 4.4
y de las constantes de R2-E2 (Prop. 4.5), con mpmath a 50 digitos.

Rutas independientes del script del autor:
  (a) SS/N^2 = -sum_{deg[D,D']<=n} w_D w_D'  calculado como coeficiente de
      C_q^2 prod_P (1+a_P u^{deg P}) / (1-u), con el producto sobre P truncado en grado n
      (exacto, porque solo entran P de grado <= n).
  (b) El asintotico -(n^2/2 + beta_q n + gamma_q) con beta_q, gamma_q calculados por
      sumas sobre primos (gamma1 = -sum deg P/(|P|-1)^2; gamma2 por la formula de la prueba).
  (c) Comparacion con los SS crudos de la auditoria (R2_numerica_2026-10-25.jsonl).
"""
import json, sys
from mpmath import mp, mpf

mp.dps = 50


def _mobius(m):
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


def prime_counts(q, nmax):
    """pi_q(e) para e<=nmax (Gauss / Moebius)."""
    return {e: sum(_mobius(e // d) * q ** d for d in range(1, e + 1) if e % d == 0) // e
            for e in range(1, nmax + 1)}


def poly_mul_trunc(a, b, n):
    c = [mpf(0)] * (n + 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if i + j > n:
                break
            c[i + j] += ai * bj
    return c


def coeffs_SS(q, n):
    """Devuelve lista c[m] = sum_{deg[D,D']=m} w_D w_D' / C_q^2 para m<=n (producto de Euler truncado)."""
    pc = prime_counts(q, n)
    poly = [mpf(1)] + [mpf(0)] * n
    for e in range(1, n + 1):
        P = mpf(q) ** e
        aP = (2 * P - 3) / (P - 2) ** 2
        # (1 + aP u^e)^{pi(e)} truncado
        fac = [mpf(1)] + [mpf(0)] * n
        base = [mpf(0)] * (n + 1)
        base[0] = mpf(1)
        base[e] = aP
        k = pc[e]
        # exponenciacion binaria truncada
        while k:
            if k & 1:
                fac = poly_mul_trunc(fac, base, n)
            k >>= 1
            if k:
                base = poly_mul_trunc(base, base, n)
        poly = poly_mul_trunc(poly, fac, n)
    return poly


def Cq(q, emax=60):
    # C_q = prod_P (1 - 1/(|P|-1)^2); cola geometrica despreciable a 50 digitos con emax grande
    pc = prime_counts(q, emax)
    s = mpf(0)
    for e in range(1, emax + 1):
        P = mpf(q) ** e
        s += pc[e] * mp.log(1 - 1 / (P - 1) ** 2)
    return mp.e ** s


def SS_over_N2(q, n):
    c = coeffs_SS(q, n)
    return -Cq(q) ** 2 * sum(c[: n + 1])


def e2_constants(q, emax=80):
    pc = prime_counts(q, emax)
    g1 = mpf(0)
    s2 = mpf(0)
    for e in range(1, emax + 1):
        P = mpf(q) ** e
        hp = -1 / (P - 1) ** 2
        hpp = -((2 * P - 3) ** 2 + 2 * (P - 1) ** 2) / (P - 1) ** 4
        g1 += pc[e] * e * hp
        s2 += pc[e] * (e * (e - 1) * hp + e * e * hpp)
    g2 = g1 ** 2 + s2
    beta = mpf(3) / 2 - g1
    gamma = 1 - g1 + g2 / 2
    return g1, g2, beta, gamma


def main():
    out = {}
    # (c) comparacion con la auditoria
    aud = [json.loads(l) for l in open('/home/claude/ariadna/auditorias/R2_numerica_2026-10-25.jsonl')]
    print("== SS: formula exacta (Lema 4.2, ruta propia) vs auditoria ==")
    maxrel = 0
    for d in aud:
        q, n = d['q'], d['n']
        v = SS_over_N2(q, n)
        rel = abs((v - mpf(d['SS'])) / v)
        maxrel = max(maxrel, rel)
        print(f"q={q:4d} n={n:2d}  SS/N^2 propio={mp.nstr(v, 14):>20}  auditoria={d['SS']:.12f}  rel={mp.nstr(rel, 3)}")
    print("max rel err:", mp.nstr(maxrel, 3))
    out['ss_max_relerr_vs_audit'] = float(maxrel)

    # (B) q grande, n fijo: q*(SS/N^2 + binom(n+2,2)) -> -(n-2) segun la compuerta
    print("\n== Prop 4.4: q*(SS/N^2 + binom(n+2,2)) para q grande ==")
    from math import comb
    for n in range(2, 7):
        row = []
        for q in (10007, 100003, 1000003):
            v = SS_over_N2(q, n)
            row.append(float(q * (v + comb(n + 2, 2))))
        print(f"n={n}: {row}")
        out[f'qSS_n{n}'] = row

    # (R2-E2) constantes y residuo
    print("\n== R2-E2: beta_q, gamma_q y residuo SS/N^2 + (n^2/2 + beta n + gamma) ==")
    for q in (3, 5, 7, 11, 101):
        g1, g2, beta, gamma = e2_constants(q)
        res = {}
        for n in (10, 20, 30, 40):
            if q > 7 and n > 20:
                continue
            v = SS_over_N2(q, n)
            res[n] = float(v + (mpf(n) ** 2 / 2 + beta * n + gamma))
        print(f"q={q}: beta_q={mp.nstr(beta, 10)} gamma_q={mp.nstr(gamma, 10)} residuos={res}")
        out[f'e2_q{q}'] = {'beta': float(beta), 'gamma': float(gamma), 'res': res}
        # tasa geometrica
        if q == 3:
            r = [abs(res[n]) for n in sorted(res) if n <= 30]
            print("   cociente de residuos por 10 pasos (esperado 3^5=15.6):", [r[i] / r[i + 1] for i in range(len(r) - 1)])
    json.dump(out, open('/home/claude/ariadna/R2/arbitro2_fable/arb2_ss_e2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
