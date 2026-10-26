"""E1_resumen.py -- tabulates E1_bloques_<k>.json: L_w(Y)/N^2, L_w(Y)*Y^2/N^3, fitted exponents, top blocks."""
import json, math, numpy as np
ks = list(range(14, 23)); out = {}
D = {k: json.load(open(f'/home/claude/ariadna/R1/run2_E1/E1_bloques_{k}.json')) for k in ks}
for w in ['1-x', '(1-x)^2', 'box']:
    print('== weight', w); out[w] = {}
    print('relerr X identity max:', max(D[k]['res'][w]['relerr'] for k in ks))
    print('kernel sums (k=22):', D[22]['res'][w]['kernel'])
    for i, eps in enumerate([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]):
        Ls = np.array([D[k]['res'][w]['L_Y'][i]['L'] for k in ks])
        Ys = np.array([D[k]['res'][w]['L_Y'][i]['Y'] for k in ks]); Ns = 2.0 ** np.array(ks)
        ratio = Ls * Ys ** 2 / Ns          # L/N^2 divided by N/Y^2
        mx = [D[k]['res'][w]['L_Y'][i]['maxDelta'] for k in ks]
        # exponent fit of |L_w| = C N^a  (on N^2 scale: |L|/N^2 ~ N^(a-2))
        a = np.polyfit(np.log(Ns), np.log(np.abs(Ls) + 1e-30), 1)[0] + 2
        print(f'eps={eps:4}: L/N^2 =', ' '.join(f'{x:+.5f}' for x in Ls))
        print(f'          L*Y^2/N^3 =', ' '.join(f'{x:+.3f}' for x in ratio), f' fitted exponent of |L_w|: {a:.3f}  (pred. 2-2eps={2-2*eps:.2f})')
        out[w][str(eps)] = dict(L=Ls.tolist(), ratio=ratio.tolist(), exponent=a, maxDelta=mx)
    # top blocks
    for k in [16, 20, 22]:
        bl = D[k]['res'][w]['blocks']
        print(f'  k={k} blocks D>=N^(1/2):', ' '.join(f"{b['D']}:{b['B']:+.5f}/{b['B_abs']:.4f}" for b in bl if b['D'] >= 2 ** (k // 2)))
    out[w]['blocks'] = {k: D[k]['res'][w]['blocks'] for k in ks}
    if w != 'box':
        for k in [16, 20, 22]:
            for s in D[k]['res'][w]['smooth']:
                sw = s['switch_e1']
                print(f"  k={k} eps={s['eps']} delta={s['delta']}: S_Theta={s['S_smooth']:+.6f} exp.relerr={s['relerr']:.1e} "
                      f"tails(E)={[(t['E'], round(t['tail'],5)) for t in s['trunc']]} switch e=1: T(2)={sw['T2']:+.6f} "
                      f"pp sum/abs={sw['pp_sum_over_l']:+.6f}/{sw['pp_abs_over_l']:.6f} gb sum/abs={sw['gb_sum_over_l']:+.6f}/{sw['gb_abs_over_l']:.6f}")
json.dump(out, open('/home/claude/ariadna/R1/run2_E1/E1_resumen.json', 'w'), indent=1)
