# R^2 = 1 - (resid_sd/sd)^2 for the regressions quoted in Sec. num (OLS with intercept, np.std ddof=0 in both).
import json
src = '/home/claude/pruebas/manuscrito/v5_scratch/check_B2_S2_rerun.json'
d = json.load(open(src))['S2']; out = {}
for rng, v in d.items():
    for key in ('Xle30~End1', 'Xle30~End30'):
        f = v['fits'][key]
        out[f'{rng} {key}'] = dict(resid_sd=f['resid_sd'], sd=f['sd'], R2=1 - (f['resid_sd'] / f['sd'])**2)
json.dump(out, open('/home/claude/pruebas/manuscrito/v5_scratch/r2_v5.json', 'w'), indent=1)
for k, x in out.items(): print(k, {a: round(b, 4) for a, b in x.items()})
