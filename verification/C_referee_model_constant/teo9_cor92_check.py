"""Árbitro: Teorema 9 y Corolario 9.2, con código propio.
 - media_w T2^F(n,n-1) = sum_n w T2 / sum_n w   (Teor. 9 afirma O(log N loglog N); run8 midió ≈ -0.391)
 - Phi(N) = sum_n w S(n)^2 M1(n) / sum_n w S(n) n log^2 n,  con M1 = sum_{0<|h|<n} (n-1-|h| - [h par, |h|<=n-2]) (r-1)
 - descomposición: parte T1 (rho) y parte T2 (r-rho) de Phi, y comparación con -1/2 - c_F/L (E1 Teor. 7') ."""
import sys, math, json, numpy as np
import arb_comun as A
Ks = [int(x) for x in sys.argv[1:]] or [9, 10]
out = {}
for K in Ks:
    N = 1 << K
    num1 = num2 = numM = den = W = 0.0; T2s = []; ws = []
    for n in range(N + 2, 2 * N, 2):
        w = float(A.w_smooth(np.array([n / N]))[0])
        if w == 0: continue
        r, rho = A.r_rho(n); H = n - 1; h = np.arange(1, H + 1)
        fej = 1 - h / H
        T1 = 2 * float(np.dot(fej, rho - 1)); T2 = 2 * float(np.dot(fej, r - rho))
        wt = (n - 1 - h) - ((h % 2 == 0) & (h <= n - 2))
        M1 = 2 * float(np.dot(wt, r - 1))
        S = A.sing(n)
        num1 += w * S ** 2 * (n - 1) * T1; num2 += w * S ** 2 * (n - 1) * T2; numM += w * S ** 2 * M1
        den += w * S * n * math.log(n) ** 2; W += w; T2s.append(T2); ws.append(w)
    T2s = np.array(T2s); ws = np.array(ws)
    L = math.log(1.5 * N)
    res = dict(N=N, mean_w_T2=float(np.sum(ws * T2s) / W), rms_T2=float(np.sqrt(np.mean(T2s ** 2))), Phi=numM / den, Phi_T1=num1 / den, Phi_T2=num2 / den,
               Phi_rest=(numM - num1 - num2) / den, pred_T1=-0.5 - 3.094 / L, L=L, logN_loglogN=math.log(N) * math.log(math.log(N)))
    print(json.dumps(res, indent=1)); out[K] = res
json.dump(out, open("teo9_cor92_check.json", "w"), indent=1)
