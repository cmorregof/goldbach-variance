"""check_halves.py -- sizes of the two halves of the prime part per q-block:  P = P_pair - P_gold,
P_pair(Q) = sum_{Q<q<=2Q} 2mu^2/phi^2 sum_{d|q} d mu(q/d) Gpair(d),  Gpair(d)=sum_{d|h} rho psi2 (h over all integers),
P_gold(Q) likewise with Ggold(d)=sum_{d|n} nu R.  (the deterministic part D pairs with P_pair - P_gold).
Also the 'main-term' comparison: for each half, subtract its own Hardy-Littlewood prediction  (rho*S*rho  resp. nu*S*nu).
Usage: python3 -I check_halves.py"""
import sys, os, math, numpy as np, scipy.fft as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bloques as cb

def halves(ar, N, wname, D=1 << 14):
    wf = cb.WEIGHTS[wname]; m = np.arange(N + 1); w = np.zeros(N + 1); w[1:] = wf(m[1:] / N)
    lam = ar.Lam[:N + 1] * w; Lf = cb.flen(N)
    FL = sf.rfft(lam, Lf, workers=2); Fw = sf.rfft(w, Lf, workers=2)
    R = sf.irfft(FL * FL, Lf, workers=2)[:2 * N + 1]; psi2 = sf.irfft(FL * np.conj(FL), Lf, workers=2)[:N + 1].copy()
    nu = sf.irfft(Fw * Fw, Lf, workers=2)[:2 * N + 1]; rho = sf.irfft(Fw * np.conj(Fw), Lf, workers=2)[:N + 1].copy()
    S = ar.S[:2 * N + 1]
    Gp = np.zeros(D + 1); Gg = np.zeros(D + 1); Mp = np.zeros(D + 1); Mg = np.zeros(D + 1)
    a = rho[1:] * psi2[1:]; b = nu * R; am = rho[1:] ** 2 * S[1:N + 1]; bm = nu ** 2 * S
    for d in range(1, D + 1):
        Gp[d] = rho[0] * psi2[0] + 2 * a[d - 1::d].sum(); Gg[d] = b[0::d].sum()
        Mp[d] = rho[0] * psi2[0] + 2 * am[d - 1::d].sum(); Mg[d] = bm[0::d].sum()
    mu, phi, wsum = cb.sieve_arith(D)
    def blocks(F):
        A = np.zeros(D + 1)
        for d in range(1, D + 1):
            mm = D // d; A[d:d * mm + 1:d] += d * F[d] * mu[1:mm + 1]
        sq = (mu != 0); sq[0] = False; ph = np.maximum(phi, 1).astype(float)
        Xq = np.where(sq, 2.0 / ph ** 2 * A, 0.0); out = []; Q = 1
        while 2 * Q <= D:
            lo, hi = (1, 2) if Q == 1 else (Q + 1, 2 * Q); out.append((Q if Q > 1 else 0, Xq[lo:hi + 1].sum() / float(N) ** 2)); Q = 2 * Q if Q > 1 else 2
        return out
    return blocks(Gp), blocks(Gg), blocks(Mp), blocks(Mg)

if __name__ == "__main__":
    ar = cb.Arith(1 << 20)
    for N, wn in ((1 << 18, '1-x'), (1 << 20, '1-x'), (1 << 20, '(1-x)^2')):
        bp, bg, mp, mg = halves(ar, N, wn)
        print(f"w={wn} N=2^{int(math.log2(N))}: per q-block, /N^2:  Ppair, Pgold, Ppair-Pgold | model parts: Mpair, Mgold (H-L prediction), and residuals Ppair-Mpair, Pgold-Mgold")
        for (q, a), (_, b), (_, c), (_, d) in zip(bp, bg, mp, mg):
            print(f"   Q={q:5d}  Ppair={a:9.4f} Pgold={b:9.4f} diff={a-b:+9.4f} | Mpair={c:9.4f} Mgold={d:9.4f} | resid pair {a-c:+8.4f} gold {b-d:+8.4f}")
