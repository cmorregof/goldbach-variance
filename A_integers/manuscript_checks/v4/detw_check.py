"""Independent check (v4 referee) of Prop. Xw(b) / Lemma dbasis(d) and of Det_w(1) -> 2 c_B^w.

Everything is computed from the definitions of the paper:
  w_m = w(m/N), 1<=m<=N (w_N = 0), rho_w(h) = sum_m w_m w_{m+h}, nu_w(n) = sum_{a+b=n} w_a w_b,
  F_M(d)   = sum_{h != 0, d|h} S(h) rho_w(h)^2,   F_mod(d) = sum_{d|n} S(n) nu_w(n)^2,
  Det_w(1) = 2[ -rho_w(0) psi_2^w(0) + F_mod(1) - F_M(1) ],  psi_2^w(0) = sum Lambda(m)^2 w_m^2.
Constants r0, h_w, gamma_w, kappa_w, int r^2, int g^2, int r w, int g w by mpmath quadrature.
Usage: python3 detw_check.py WEIGHT kmin kmax   (WEIGHT in lin, sq, exp)
"""
import sys, json, math, time
import numpy as np
import mpmath as mp

mp.mp.dps = 25
C2 = 0.66016181584686957
EUL = float(mp.euler); L2PI = math.log(2 * math.pi)

def wfun(name):
    if name == 'lin':
        return (lambda x: 1 - x), (lambda x: np.where(x < 1, 1 - x, 0.0))
    if name == 'sq':
        return (lambda x: (1 - x) ** 2), (lambda x: np.where(x < 1, (1 - x) ** 2, 0.0))
    if name == 'exp':
        f = lambda x: mp.e ** (-x / (1 - x)) if x < 1 else mp.mpf(0)
        def fn(x):
            out = np.zeros_like(x)
            m = x < 1
            out[m] = np.exp(-x[m] / (1 - x[m]))
            return out
        return f, fn
    raise ValueError(name)

def constants(name):
    w, _ = wfun(name)
    W = lambda x: w(x) if 0 <= x <= 1 else mp.mpf(0)
    r = lambda u: mp.quad(lambda x: W(x) * W(x + u), [0, 1 - u]) if u < 1 else mp.mpf(0)
    def g(u):
        lo, hi = max(0, u - 1), min(1, u)
        return mp.quad(lambda y: W(u - y) * W(y), [lo, hi]) if hi > lo else mp.mpf(0)
    r0 = r(0)
    c = dict(
        r0=r0,
        h=mp.quad(lambda u: (r(u) ** 2 - r0 ** 2) / u, [0, 0.5, 1]),
        gam=mp.quad(lambda u: g(u) ** 2 / u, [0, 1, 2]),
        kap=mp.quad(lambda x: W(x) ** 2 * mp.log(x), [0, 1]),
        Ir2=mp.quad(lambda u: r(u) ** 2, [0, 0.5, 1]),
        Ig2=mp.quad(lambda u: g(u) ** 2, [0, 0.5, 1, 1.5, 2]),
        Irw=mp.quad(lambda u: r(u) * W(u), [0, 0.5, 1]),
        Igw=mp.quad(lambda u: g(u) * W(u), [0, 0.5, 1]),
        w0=W(mp.mpf(0)),
    )
    c['cBw'] = c['r0'] ** 2 * (mp.euler + mp.log(2 * mp.pi)) - c['r0'] * c['kap'] + c['h'] - c['gam'] / 2
    return {k: float(v) for k, v in c.items()}

def sieve(M):
    isp = np.ones(M + 1, bool); isp[:2] = False
    for p in range(2, int(M ** 0.5) + 1):
        if isp[p]:
            isp[p * p::p] = False
    return np.nonzero(isp)[0]

def kernels_fft(wm, N):
    # wm[m] for m=0..N, wm[0]=0 ; rho(h) h=0..N-1 ; nu(n) n=0..2N
    size = 1 << int(math.ceil(math.log2(2 * N + 2)))
    F = np.fft.rfft(wm, size)
    nu = np.fft.irfft(F * F, size)[:2 * N + 1]
    ac = np.fft.irfft(F * np.conj(F), size)[:N]
    return ac, nu

def kernels_lin_exact(N):
    # w_m = (N-m)/N ; N^2 rho(h) = sum_{j=0}^{n-1} j(j+h), n=N-h
    h = np.arange(N, dtype=np.float64); n = N - h
    S1 = (n - 1) * n / 2; S2 = (n - 1) * n * (2 * n - 1) / 6
    rho = (S2 + h * S1) / N ** 2
    # N^2 nu(n) = sum_{a=lo}^{hi} (N-a)(N-n+a); lo=max(1,n-N), hi=min(N,n-1)
    nn = np.arange(2 * N + 1, dtype=np.float64)
    lo = np.maximum(1, nn - N); hi = np.minimum(N, nn - 1)
    cnt = np.maximum(hi - lo + 1, 0)
    # (N-a)(N-n+a) = -a^2 + a n + N(N-n)  ... expand: N^2 - N n + a n - a^2 ... check: (N-a)(N-n+a)=N^2-Nn+Na-aN+an-a^2 = N^2-Nn+an-a^2
    def P1(x): return x * (x + 1) / 2
    def P2(x): return x * (x + 1) * (2 * x + 1) / 6
    s1 = P1(hi) - P1(lo - 1); s2 = P2(hi) - P2(lo - 1)
    nu = np.where(cnt > 0, (N * N - N * nn) * cnt + nn * s1 - s2, 0.0) / N ** 2
    return rho, nu

def main():
    name = sys.argv[1]; kmin, kmax = int(sys.argv[2]), int(sys.argv[3])
    c = constants(name)
    print('constants', name, json.dumps({k: round(v, 10) for k, v in c.items()}))
    _, wnp = wfun(name)
    Nmax = 1 << kmax
    t0 = time.time()
    P = sieve(2 * Nmax)
    S = np.zeros(2 * Nmax + 1); S[0::2] = 2 * C2
    for p in P[1:]:
        S[p::p] *= (p - 1) / (p - 2)
    S[0] = 0.0
    Lam = np.zeros(Nmax + 1)
    for p in P[P <= Nmax]:
        lp = math.log(p); q = p
        while q <= Nmax:
            Lam[q] = lp; q *= p
    print('sieve %.1fs' % (time.time() - t0), flush=True)
    dlist = [1, 2, 3, 6, 15, 30, 105, 210, 1155, 2310, 30030]
    out = []
    for k in range(kmin, kmax + 1):
        N = 1 << k; L = math.log(N)
        m = np.arange(N + 1, dtype=np.float64)
        wm = wnp(m / N); wm[0] = 0.0; wm[N] = 0.0
        rho, nu = kernels_fft(wm, N)
        fftdiff = None
        if name == 'lin':
            rho_e, nu_e = kernels_lin_exact(N)
            fftdiff = (float(np.max(np.abs(rho - rho_e))), float(np.max(np.abs(nu - nu_e))))
            rho, nu = rho_e, nu_e
        rho0 = float(np.sum(wm ** 2))
        psi2w0 = float(np.sum(Lam[:N + 1] ** 2 * wm ** 2))
        Sn = S[:2 * N + 1]
        row = dict(k=k, N=N, fft_vs_exact=fftdiff)
        for d in dlist:
            d0 = d // 2 if d % 2 == 0 else d
            ps = [p for p in P[P <= d0] if d0 % p == 0 and p > 2]
            phid0 = 1.0
            for p in ps: phid0 *= (p - 1)
            sump = sum(math.log(p) / (p - 1) for p in ps)
            hh = np.arange(d, N, d)
            FM = 2 * float(np.sum(Sn[hh] * rho[hh] ** 2))
            nn = np.arange(d, 2 * N + 1, d)
            Fmod = float(np.sum(Sn[nn] * nu[nn] ** 2))
            cd1 = EUL + L2PI + sump                     # 1 + c_d
            pred_diff = c['r0'] ** 2 * N ** 2 * math.log(N / d0) + N ** 2 * (c['r0'] ** 2 * cd1 + c['h'] - c['gam'] / 2)
            FM_naive = 2 * N ** 3 / phid0 * c['Ir2'] - c['r0'] ** 2 * N ** 2 * L - c['h'] * N ** 2 \
                       - c['r0'] ** 2 * N ** 2 * (cd1 - math.log(d0))
            Fmod_naive = N ** 3 / phid0 * c['Ig2'] - 0.5 * c['gam'] * N ** 2
            corrM = -2 * N ** 2 * c['w0'] * c['Irw'] / phid0
            corrmod = -2 * N ** 2 * c['w0'] * c['Igw'] / phid0
            e = (Fmod - FM) - pred_diff
            row['d%d' % d] = dict(
                err_diff_over_N2=e / N ** 2,
                err_diff_over_sqrtd_N32=e / (math.sqrt(d) * N ** 1.5),
                FM_minus_naive_over_N2=(FM - FM_naive) / N ** 2,
                corrM_over_N2=corrM / N ** 2,
                Fmod_minus_naive_over_N2=(Fmod - Fmod_naive) / N ** 2,
                corrmod_over_N2=corrmod / N ** 2,
                FM_resid_after_corr_over_N2=(FM - FM_naive - corrM) / N ** 2,
                Fmod_resid_after_corr_over_N2=(Fmod - Fmod_naive - corrmod) / N ** 2,
            )
        # Det_w(1) from the definition
        hh = np.arange(1, N); nn = np.arange(1, 2 * N + 1)
        FM1 = 2 * float(np.sum(Sn[hh] * rho[hh] ** 2)); Fmod1 = float(np.sum(Sn[nn] * nu[nn] ** 2))
        Det1 = 2 * (-rho0 * psi2w0 + Fmod1 - FM1)
        pnt = 2 * rho0 * (N * (c['r0'] * L + c['kap']) - psi2w0)   # PNT part of Det_w(1)
        row.update(rho0_minus_Nr0=rho0 - N * c['r0'], psi2w0_dev_over_sqrtN=(psi2w0 - N * (c['r0'] * L + c['kap'])) / math.sqrt(N),
                   Det1_over_N2=Det1 / N ** 2, twocBw=2 * c['cBw'], Det1_minus_2cBw=Det1 / N ** 2 - 2 * c['cBw'],
                   Det1_det_part_minus_2cBw=(Det1 - pnt) / N ** 2 - 2 * c['cBw'], Det1_pnt_part_over_N2=pnt / N ** 2,
                   Det1_minus_2cBw_times_sqrtN=(Det1 / N ** 2 - 2 * c['cBw']) * math.sqrt(N))
        out.append(row)
        print(json.dumps({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in row.items() if not kk.startswith('d')}), flush=True)
    json.dump(dict(weight=name, constants=c, rows=out), open('detw_%s_%d_%d.json' % (name, kmin, kmax), 'w'), indent=1)

if __name__ == '__main__':
    main()
