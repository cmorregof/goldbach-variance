"""k1_Gw_bloques.py -- R1 / K1, corrida 1 de Opus.  Contraste (ii).

Mide G_w(d) (Lema dbasis(c) del manuscrito v5) para d par libre de cuadrados, 6 <= d <= DMAX, N = 2^14, 2^15, 2^16,
y lo compara con la prediccion diagonal N^2 D_w(d) de K1_informe.md (Paso 2, formula (2.12)):

  D_w(d) = r0^2 (log d - sum_{p|d} log p/(p-1)) - c_B^w                       [= G_w^FG(d)/N^2]
           + X2 * [d = 2p] * log p/(p-1)                                       (paridad del conductor, d = 2p)
           + A_odd * avg_odd(d)                                                (residuo de los impares, GRH: O(d^-1/2 log^2 d))
           + H1int / phi(d)                                                    (polos de zeta)
           + Pr(d)                                                             (suma de primos de Weil, n = +-1 mod d y p^k, p|d)

  avg_odd(d) = (1/phi(d)) sum_{chi impar mod d} Re L'/L(1, chi*), por ortogonalidad (primos hasta XP; la cola de la
  diferencia de dos progresiones es O(XP^{-1/2}) en la practica).

Las constantes r0, c_B^w, X2, A_odd, H1int se toman de k1_constantes.json (cuadratura sin primos).

Salida: k1_Gw_bloques.json (por d y por bloque diadico); stdout.
Uso: python3 -I k1_Gw_bloques.py [DMAX]     (DMAX por defecto 1024; unos 3-5 minutos, < 1.5 GB)
"""
import sys, json, math, time
import numpy as np, scipy.fft as sf

OUT = "/home/claude/ariadna/R1/run1_K1/k1_Gw_bloques.json"
CST = {o['weight']: o for o in json.load(open("/home/claude/ariadna/R1/run1_K1/k1_constantes.json"))}
DMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DMAIN = 330
XP = 10 ** 8
C2 = 0.660161815846869573927812110015

def sieve(n):
    isp = np.ones(n + 1, bool); isp[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if isp[p]: isp[p * p::p] = False
    return np.nonzero(isp)[0]

def pfac(n):
    out = []; m = n; p = 2
    while p * p <= m:
        if m % p == 0:
            out.append(p)
            while m % p == 0: m //= p
        p += 1
    if m > 1: out.append(m)
    return out

def squarefree(n):
    m = n; p = 2
    while p * p <= m:
        if m % (p * p) == 0: return False
        if m % p == 0: m //= p
        p += 1
    return True

def phi(n):
    r = n
    for p in pfac(n): r = r // p * (p - 1)
    return r

# ------------------------------------------------------------------ pesos
def W(name, x):
    x = np.asarray(x, float); out = np.zeros_like(x); m = (x >= 0) & (x < 1)
    xm = x[m]
    if name == '1-x': out[m] = 1 - xm
    elif name == '(1-x)^2': out[m] = (1 - xm) ** 2
    elif name == 'bump': out[m] = np.exp(-xm * xm / (1 - xm * xm))
    elif name == 'expw': out[m] = np.exp(-xm / (1 - xm))
    return out

def dW0(name):
    return {'1-x': -1.0, '(1-x)^2': -2.0, 'bump': 0.0, 'expw': -1.0}[name]

# ------------------------------------------------------------------ Hcheck^{e,o}(n) = int_0^1 du int_0^{1/n} (c_f(y)-c_f(0)) c_f(ny) dy
XG, WG = np.polynomial.legendre.leggauss(24)
def gl_panels(edges):
    a = np.asarray(edges[:-1]); b = np.asarray(edges[1:])
    x = (0.5 * (b - a)[:, None] * XG + 0.5 * (a + b)[:, None]).ravel()
    w = (0.5 * (b - a)[:, None] * WG).ravel()
    return x, w

def cf(name, sig, x, u):
    return W(name, x) * W(name, x - u) - sig * W(name, x) * W(name, x + u)

def Hcheck(name, sig, n, nz=64):
    # y = z/n:  (1/n) int_0^1 dz int_0^1 du [c_f(z/n)-c_f(0)] c_f(z)
    zx, zw = gl_panels(np.linspace(0, 1, 9))
    tot = 0.0
    for z, wz in zip(zx, zw):
        br = sorted(set([0.0, 1.0] + [b for b in (z, z / n, 1 - z, 1 - z / n) if 0 < b < 1]))
        ux, uw = gl_panels(br)
        c0 = -sig * W(name, np.zeros_like(ux)) * W(name, ux)
        val = (cf(name, sig, np.full_like(ux, z / n), ux) - c0) * cf(name, sig, np.full_like(ux, z), ux)
        tot += wz * np.sum(uw * val)
    return tot / n

def Kasym(name, sig):
    # Hcheck(n) ~ K/n^2,  K = int_0^1 du c_f'(0+) int_0^1 z c_f(z) dz,  c_f'(0+) = -sig (w'(0) w(u) + w(0) w'(u))
    ux, uw = gl_panels(np.linspace(0, 1, 17)); h = 1e-6
    dwu = (W(name, ux + h) - W(name, ux - h)) / (2 * h)
    cp = -sig * (dW0(name) * W(name, ux) + W(name, np.zeros_like(ux)) * dwu)
    zx, zw = gl_panels(np.linspace(0, 1, 17))
    inner = np.array([np.sum(zw * zx * cf(name, sig, zx, u)) for u in ux])
    return float(np.sum(uw * cp * inner))

# ------------------------------------------------------------------ G_w(d)
def Gw_all(Lam, N, name, ds):
    m = np.arange(N + 1); w = W(name, m / N); w[0] = 0.0; w[N] = 0.0
    lam = Lam[:N + 1] * w
    Lf = 1
    while Lf < 2 * N + 2: Lf *= 2
    FL = sf.rfft(lam, Lf); Fw = sf.rfft(w, Lf)
    R = sf.irfft(FL * FL, Lf)[:2 * N + 1]; psi2 = sf.irfft(FL * np.conj(FL), Lf)[:N + 1]
    nu = sf.irfft(Fw * Fw, Lf)[:2 * N + 1]; rho = sf.irfft(Fw * np.conj(Fw), Lf)[:N + 1]
    aG = rho * psi2; bG = nu * R
    base = rho[0] * psi2[0]
    G = {d: float(base + 2 * aG[d::d].sum() - bG[d::d].sum()) for d in ds}
    return G, float(rho[0]), float(psi2[0])

def main():
    t0 = time.time()
    ds = [d for d in range(6, DMAX + 1, 2) if squarefree(d)]
    # ---------------- avg_odd(d) por ortogonalidad, primos hasta XP
    P = sieve(XP); print(f"criba hasta {XP:.0e}: {len(P)} primos ({time.time()-t0:.0f}s)", flush=True)
    lp_over_p = np.log(P) / P
    # potencias p^k, k>=2, hasta XP
    pp = [];
    for p in P[P <= int(XP ** 0.5) + 1]:
        q = int(p) * int(p)
        while q <= XP: pp.append((q, math.log(p))); q *= int(p)
    ppn = np.array([a for a, _ in pp], np.int64); ppl = np.array([b for _, b in pp]) / ppn
    avg_odd = {}
    for d in ds:
        r = P % d; rq = ppn % d
        cop = np.ones(len(P), bool)
        for p in pfac(d): cop &= (P != p)
        s1 = lp_over_p[(r == 1) & cop].sum() + ppl[rq == 1].sum()
        sm = lp_over_p[(r == d - 1) & cop].sum() + ppl[rq == d - 1].sum()
        val = -0.5 * (s1 - sm)
        for p in pfac(d):
            mm = d // p
            if mm < 3: continue
            k = 1; pk = p; acc = 0.0
            while pk < 10 ** 15:
                if pk % mm == 1: acc += math.log(p) / pk
                if pk % mm == mm - 1: acc -= math.log(p) / pk
                k += 1; pk *= p
            val -= acc / (2 * (p - 1))
        avg_odd[d] = val
    print(f"avg_odd listo ({time.time()-t0:.0f}s)", flush=True)
    # potencias de primos pequenas para la suma de Weil Pr(d)
    NH = 3000
    PPs = []
    for p in P[P <= NH]:
        q = int(p)
        while q <= NH: PPs.append((q, math.log(p))); q *= int(p)
    PPs.sort()
    PPs_big = [(int(p), math.log(p)) for p in P[(P > NH) & (P <= 2 * 10 ** 6)]]
    weights = ('1-x', '(1-x)^2', 'bump', 'expw')
    res = dict(meta=dict(DMAX=DMAX, DMAIN=DMAIN, XP=XP, NH=NH), weights={})
    # Lambda hasta 2^16
    Nmax = 1 << 16
    Lam = np.zeros(Nmax + 1)
    for p in P[P <= Nmax]:
        q = int(p)
        while q <= Nmax: Lam[q] = math.log(p); q *= int(p)
    for name in weights:
        c = CST[name]; r0 = c['r0']; cBw = c['cBw']
        # tabla de Hcheck^{e,o}
        He = {n: Hcheck(name, +1, n) for n, _ in PPs}; Ho = {n: Hcheck(name, -1, n) for n, _ in PPs}
        Ke, Ko = Kasym(name, +1), Kasym(name, -1)
        Hfun = lambda n, sig: (He if sig == 1 else Ho).get(n, (Ke if sig == 1 else Ko) / n ** 2)
        PPn = np.array([n for n, _ in PPs + PPs_big], np.int64); PPl = np.array([l for _, l in PPs + PPs_big])
        HE = np.array([Hfun(int(n), 1) for n in PPn]); HO = np.array([Hfun(int(n), -1) for n in PPn])
        print(f"[{name}] Hcheck listo; K_e={Ke:.4f} K_o={Ko:.4f}; He(2)={He[2]:.5f} Ho(2)={Ho[2]:.5f} ({time.time()-t0:.0f}s)", flush=True)
        dpred = {}
        for d in ds:
            fs = pfac(d); ph = phi(d)
            lbar = math.log(d) - sum(math.log(p) / (p - 1) for p in fs)
            GFG = r0 ** 2 * lbar - cBw
            odd = [p for p in fs if p > 2]
            par = c['X2'] * math.log(odd[0]) / (odd[0] - 1) if len(odd) == 1 else 0.0
            Rodd = c['A_odd'] * avg_odd[d]
            Rprin = c['H1int'] / ph
            # suma de primos de Weil: -sum_{(n,d)=1} Lam(n)[(1_{n=1}+1_{n=-1}) He + (1_{n=1}-1_{n=-1}) Ho] - terminos p|d
            rr = PPn % d
            cop = np.ones(len(PPn), bool)
            for p in fs: cop &= (PPn % p != 0)
            m1 = cop & (rr == 1); mm1 = cop & (rr == d - 1)
            Pr = -float(np.sum(PPl[m1] * (HE[m1] + HO[m1])) + np.sum(PPl[mm1] * (HE[mm1] - HO[mm1])))
            for p in fs:
                mm = d // p; pk = p
                while pk < 10 ** 12:
                    lg = math.log(p)
                    if mm <= 2:
                        Pr -= 2 * lg * Hfun(pk, 1) / (p - 1)          # un solo caracter (par) con chi_p trivial
                    else:
                        e1 = (pk % mm == 1) + (pk % mm == mm - 1); o1 = (pk % mm == 1) - (pk % mm == mm - 1)
                        Pr -= lg * (e1 * Hfun(pk, 1) + o1 * Hfun(pk, -1)) / (p - 1)
                    pk *= p
            dpred[d] = dict(GFG=GFG, par=par, Rodd=Rodd, Rprin=Rprin, Pr=Pr, D=GFG + par + Rodd + Rprin + Pr)
        out = dict(per_N={})
        for lg in (14, 15, 16):
            N = 1 << lg; N2 = float(N) ** 2
            G, rho0, psi20 = Gw_all(Lam, N, name, ds)
            pnt = rho0 * (psi20 - N * (r0 * math.log(N) + c['kappa'])) / N2
            rows = []
            for d in ds:
                g = G[d] / N2; dp = dpred[d]
                rows.append(dict(d=d, phi=phi(d), G=g, D=dp['D'], GFG=dp['GFG'], dev_D=g - dp['D'], dev_FG=g - dp['GFG'],
                                 par=dp['par'], Rodd=dp['Rodd'], Rprin=dp['Rprin'], Pr=dp['Pr'],
                                 alpha=math.log(N) / math.log(d)))
            beta = {d: 4 * C2 * np.prod([1.0 / (p - 2) for p in pfac(d) if p > 2]) for d in ds}
            blocks = []
            for D in (4, 8, 16, 32, 64, 128, 256, 512):
                for lo, hi, tag in ((D, 2 * D, 'full'),):
                    sel = [r for r in rows if lo < r['d'] <= hi]
                    if not sel: continue
                    for lim, tg in ((DMAIN, 'main'), (DMAX, 'ext')):
                        s2 = [r for r in sel if r['d'] <= lim]
                        if not s2 or (tg == 'ext' and hi <= DMAIN): continue
                        dv = np.array([r['dev_D'] for r in s2]); df = np.array([r['dev_FG'] for r in s2])
                        bt = np.array([beta[r['d']] for r in s2])
                        blocks.append(dict(D=D, range=tg, n=len(s2), dmin=s2[0]['d'], dmax=s2[-1]['d'],
                                           mean_dev_D=float(dv.mean()), sd_dev_D=float(dv.std(ddof=1)) if len(dv) > 1 else float('nan'),
                                           se_dev_D=float(dv.std(ddof=1) / math.sqrt(len(dv))) if len(dv) > 1 else float('nan'),
                                           mean_dev_FG=float(df.mean()), beta_sum_dev_D=float((bt * dv).sum()),
                                           beta_sum_dev_FG=float((bt * df).sum()), beta_sum=float(bt.sum()),
                                           alpha_range=[float(s2[-1]['alpha']), float(s2[0]['alpha'])]))
            allm = np.array([r['dev_D'] for r in rows if r['d'] <= DMAIN]); allf = np.array([r['dev_FG'] for r in rows if r['d'] <= DMAIN])
            summ = dict(N=N, pnt_term=pnt, n_main=len(allm), mean_dev_D_main=float(allm.mean()), se_main=float(allm.std(ddof=1) / math.sqrt(len(allm))),
                        sd_main=float(allm.std(ddof=1)), mean_dev_FG_main=float(allf.mean()),
                        mean_dev_D_main_minus_pnt=float(allm.mean() - pnt))
            out['per_N'][str(lg)] = dict(summary=summ, blocks=blocks, rows=rows)
            print(f"[{name}] N=2^{lg}: termino PNT {pnt:+.5f}; d<= {DMAIN}: media(G/N^2 - D) = {summ['mean_dev_D_main']:+.5f} +- {summ['se_main']:.5f} "
                  f"(sd {summ['sd_main']:.4f}, n={len(allm)}); media(G/N^2 - G_FG/N^2) = {summ['mean_dev_FG_main']:+.5f}; media - PNT = {summ['mean_dev_D_main_minus_pnt']:+.5f}", flush=True)
            for b in blocks:
                print(f"      bloque ({b['D']},{2*b['D']}] {b['range']:4s} n={b['n']:3d}  media(G-D)={b['mean_dev_D']:+.5f} +- {b['se_dev_D']:.5f}  sd={b['sd_dev_D']:.4f}  "
                      f"media(G-GFG)={b['mean_dev_FG']:+.5f}  sum beta(G-D)={b['beta_sum_dev_D']:+.5f}  alpha in [{b['alpha_range'][0]:.2f},{b['alpha_range'][1]:.2f}]", flush=True)
        res['weights'][name] = out
    res['meta']['seconds'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), indent=1)
    print("listo", time.time() - t0)

if __name__ == '__main__':
    main()
