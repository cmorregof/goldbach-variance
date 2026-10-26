"""Builds the rows of Table tab:decomp (v3) from the three JSON files written by
/home/claude/pruebas/identidad_parseval.py (k=16..23: identidad_parseval_16_23.json in /home/claude/pruebas;
k=24 and k=14,15,17,19,21: rerun on 2026-10-25 in this folder)."""
import json
c1=-2.3329828407; c0=0.8016777480
d={}
for f in ["/home/claude/pruebas/identidad_parseval_16_23.json","identidad_parseval_24.json","identidad_parseval_14_15_17_19_21.json"]:
    d.update(json.load(open(f)))
out={}
for k in sorted(d,key=int):
    r=d[k]; L=r["logN"]; X=r["X1"]+r["X2"]
    pred=0.5*L*L+(c1-2)*L+(c0+1)          # Cor A0SS prediction / N^2
    dev=(r["A0"]+r["SS"])*L*L-pred         # [(A0+SS)-pred]/N^2
    out[k]=dict(A0=r["A0"],Ep=r["Ep"],SS=r["SS"],X=X,VarG=r["VarG"],VmE=r["VarG"]-r["Ep"],XoverN2=X*L*L,dev=dev,
               pred_A0SS_over_N2L2=pred/(L*L), A0SS=r["A0"]+r["SS"], fourterm_relerr=abs(r["suma"]-r["VarG"])/r["VarG"])
    print(f"2^{k}: A0 {r['A0']:.4f} Ep {r['Ep']:.4f} SS {r['SS']:+.4f} X {X:+.4f} VarG {r['VarG']:.4f} VarG-Ep {r['VarG']-r['Ep']:.4f} | X/N^2 {X*L*L:.2f} dev {dev:+.3f} pred(A0+SS)/N2L2 {pred/(L*L):.4f} meas {r['A0']+r['SS']:.4f} relerr {out[k]['fourterm_relerr']:.1e}")
json.dump(out,open("tabla_decomp_v3.json","w"),indent=1)
