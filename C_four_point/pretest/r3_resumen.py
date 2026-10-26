"""Collects exact [M] and model [H] values into one table (r3_resumen.json)."""
import json, numpy as np
D='/home/claude/ariadna/R3/'
ex=json.load(open(D+'r3_exacto_2e24.json')); big=json.load(open('/home/claude/cap6/bprima_bloques_2e28.json'))
A={r['k']:r for r in json.load(open(D+'r3_modelo_lineal_k10-16.json'))['rows']+json.load(open(D+'r3_modelo_lineal_k17-22.json'))['rows']}
B={}
for f in ('r3_modelo_2d_K8-9-10.json','r3_modelo_2d_K11-12.json','r3_modelo_2d_K13.json'):
    for k,v in json.load(open(D+f))['res'].items(): B[int(k)]=v
EL={r['k']:r for r in ex['Lam']}; ET={r['k']:r for r in ex['theta']}; EB={r['k']:r for r in big['rows']}
H2fit=lambda L: -1-6.447/L-11.41/L**2
rows=[]
print(" k  <L>   exactLam (se)    theta    B(2D)    H2      H1    | ratio exact/B  exact/H2  exact/H1")
for k in range(10,28):
    e=EL.get(k,EB.get(k)); L=e['meanlog']
    r=dict(k=k,meanlog=L,exact_Lam=e['off'],exact_se=EL[k]['off_se'] if k in EL else None,
           exact_theta=ET[k]['off'] if k in ET else None, B=B[k]['off'] if k in B else None,
           H2=A[k]['H2'] if k in A else None, H1=A[k]['H1'] if k in A else None,
           H2_extrap=None if k in A else H2fit(L))
    rows.append(r)
    f=lambda x: f"{x:+.3f}" if x is not None else "   -  "
    rat=lambda m: f"{r['exact_Lam']/m:.3f}" if m is not None else "  -  "
    print(f"{k:2d} {L:5.2f}  {r['exact_Lam']:+.3f} ({(r['exact_se'] or 0):.3f})  {f(r['exact_theta'])}  {f(r['B'])}  {f(r['H2'] if r['H2'] is not None else r['H2_extrap'])}{'*' if r['H2'] is None else ' '} {f(r['H1'])} |   {rat(r['B'])}       {rat(r['H2'] or r['H2_extrap'])}     {rat(r['H1'])}")
cls=[]
print("\nclases (off/S | off/S^2):")
for k in (11,12,13,16,22,23,27):
    line=dict(k=k)
    if k in EL: line['exact']={c:(EL[k]['clases'][str(c)]['off_over_S'],EL[k]['clases'][str(c)]['off_over_S2']) for c in (0,2,4)}
    elif k==27: line['exact']={int(c):(v['off_over_S'],v['off_over_S2']) for c,v in big['clases'].items()}
    if k in B: line['B']={c:(B[k]['clases'][str(c)]['off_over_S'],B[k]['clases'][str(c)]['off_over_S2']) for c in (0,2,4)}
    if k in A: line['H2']={c:(A[k]['clases'][str(c)]['H2_over_S'],A[k]['clases'][str(c)]['H2_over_S2']) for c in (0,2,4)}
    cls.append(line)
    for nm in ('exact','B','H2'):
        if nm in line: print(k,nm,"  ".join(f"c{c}: {v[0]:+.3f} | {v[1]:+.3f}" for c,v in line[nm].items()))
json.dump(dict(rows=rows,clases=cls,H2_fit_fixed_a="-1-6.447/L-11.41/L^2"),open(D+'r3_resumen.json','w'),indent=1)
