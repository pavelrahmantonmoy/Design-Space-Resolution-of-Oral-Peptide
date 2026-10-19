"""Literature benchmark for the BIDT release module.
All inputs are published values transcribed from:
  Vueba et al. 2004, Eur J Pharm Biopharm 58:51-59, Table 4 (Korsmeyer-Peppas, K in %/h^n)
  Levina & Rajabi-Siahboomi 2004, J Pharm Sci 93:2746-2754, Tables 3-4 (K-P, k in %/min^n; T50 in min)
  Maggi et al. 2000, Int J Pharm 195:229-238, Table 3 (td75 in min, plain matrices)
No values are invented; release curves are reconstructed from the published model parameters."""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from SALib.sample import sobol as ss
from SALib.analyze import sobol as sa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows=[]
# --- Vueba 2004, Table 4 (lactose matrices, 20% w/w HPMC, ketoprofen) ---
vueba={"F5 HPMC K15M 20%":(12.047,0.621,0.9990),"F7 HPMC K100M 20%":(9.926,0.621,0.9978)}
# --- Levina 2004, Table 4 (CPM, 20% HPMC K4M, lactose filler) ---
levina={"CPM/lactose 4 kN":(3.5188,0.5822,0.9993,95),"CPM/lactose 10 kN":(1.2356,0.7268,0.9961,160),"CPM/lactose 14 kN":(1.2152,0.7367,0.9966,175)}
# --- Maggi 2000, Table 3 (plain matrices, 38.2% polymer, diltiazem) td75 min ---
maggi={"CY2 PEO WSR 303":246,"CM2 HPMC K100M":357}

ps=lambda t,kd,kr: kd*t**0.45+kr*t**0.89
t=np.array([0.5,1,2,4,6,8,10,12,16,20.])
fits={}
for name,(K,n,R2) in vueba.items():
    q=K*t**n/100; p,_=curve_fit(ps,t,q,bounds=(0,[1,1])); r=q-ps(t,*p)
    R2ps=1-np.sum(r**2)/np.sum((q-q.mean())**2); rmse=np.sqrt(np.mean(r**2))*100
    fits[name]=p
    rows.append(dict(Source="Vueba 2004 (Table 4)",System=name,Drug="Ketoprofen",K_KP=K,n_KP=n,R2_KP=R2,
        kd_PS=round(p[0],4),kr_PS=round(p[1],4),R2_PS=round(R2ps,5),RMSE_PS_pct=round(rmse,3),Q12h_pct=round(100*ps(12,*p),1)))
for name,(k,n,R2,t50) in levina.items():
    t50_model=(50/k)**(1/n)
    rows.append(dict(Source="Levina 2004 (Tables 3-4)",System="HPMC K4M 20%, "+name,Drug="Chlorpheniramine maleate",K_KP=k,n_KP=n,R2_KP=R2,
        T50_reported_min=t50,T50_from_KP_min=round(t50_model,0)))
for name,td in maggi.items():
    rows.append(dict(Source="Maggi 2000 (Table 3)",System=name+" 38.2%",Drug="Diltiazem HCl",td75_min=td))
bench=pd.DataFrame(rows); bench.to_csv("data/literature_benchmark_table.csv",index=False)
print(bench.to_string())

# --- Kinetic-parameter Sobol: kd, kr bounded by the K100M-K15M literature span ---
kd_lo,kr_lo=fits["F7 HPMC K100M 20%"]; kd_hi,kr_hi=fits["F5 HPMC K15M 20%"]
problem={'num_vars':2,'names':['k_d (diffusional)','k_r (relaxational)'],'bounds':[[kd_lo,kd_hi],[kr_lo,kr_hi]]}
X=ss.sample(problem,1024,calc_second_order=True,seed=42); Y=100*ps(12,X[:,0],X[:,1])
S=sa.analyze(problem,Y,calc_second_order=True,seed=42)
sob=pd.DataFrame({'S1':S['S1'],'S1_conf':S['S1_conf'],'ST':S['ST'],'ST_conf':S['ST_conf']},index=problem['names']).round(4)
sob.to_csv("data/sobol_kinetic_parameters.csv"); print("\nSobol (output Q12h, N=1024, evals=%d)\n"%len(Y),sob)

# --- Monte Carlo: +/-10% uniform on kd, kr around HPMC K100M fit ---
rng=np.random.default_rng(42); n=1000
kd=rng.uniform(0.9*kd_lo,1.1*kd_lo,n); kr=rng.uniform(0.9*kr_lo,1.1*kr_lo,n)
q=100*ps(12,kd,kr); nom=100*ps(12,kd_lo,kr_lo)
mc=dict(nominal_Q12h=round(nom,2),mean_Q12h=round(q.mean(),2),SD=round(q.std(ddof=1),3),
        spread_95_rel_pct=round(1.96*q.std(ddof=1)/q.mean()*100,2),P2_5=round(np.percentile(q,2.5),2),P97_5=round(np.percentile(q,97.5),2))
pd.Series(mc).to_csv("data/monte_carlo_Q12h.csv"); print("\nMonte Carlo:",mc)

# --- Figure: literature benchmark overlay ---
fig,ax=plt.subplots(figsize=(7.5,5),dpi=300); tt=np.linspace(0.25,20,200)
for (name,(K,nn,_)),c in zip(vueba.items(),["#3B6EA8","#1B3F6B"]):
    ax.plot(tt,K*tt**nn,color=c,lw=2,label=f"Vueba 2004 {name} (K–P)")
    ax.plot(tt,100*ps(tt,*fits[name]),color=c,lw=1,ls="--")
for (name,(k,nn,_,_)),c in zip(levina.items(),["#E3A35B","#D9822B","#9C5212"]):
    th=np.linspace(0.1,12,300); qq=k*(60*th)**nn; m=(qq>=5)&(qq<=60)
    ax.plot(th[m],qq[m],color=c,lw=2,label=f"Levina 2004 {name} (K–P, 5–60%)")
for (name,td),mk in zip(maggi.items(),["o","s"]):
    ax.plot(td/60,75,mk,color="#2E7D32",ms=7,label=f"Maggi 2000 {name}: t$_{{75}}$")
ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative release (%)"); ax.set_xlim(0,20); ax.set_ylim(0,100)
ax.set_title("Literature benchmark: hydrophilic HPMC/PEO matrices",fontsize=11,fontweight="bold")
ax.text(0.98,0.02,"dashed = Peppas–Sahlin fit",transform=ax.transAxes,ha="right",fontsize=8,color="#555")
ax.spines[["top","right"]].set_visible(False); ax.legend(fontsize=7,frameon=False,loc="lower right",bbox_to_anchor=(1,0.06))
plt.tight_layout(); plt.savefig("figures/Figure_Literature_Benchmark.png")

# --- Figure 7 replacement: kinetic-parameter Sobol ---
fig,ax=plt.subplots(figsize=(6.5,3.2),dpi=300); y=np.arange(2)[::-1]; h=0.35
ax.barh(y+h/2,sob.S1,h,xerr=sob.S1_conf,color="#3B6EA8",label="First-order (S$_i$)")
ax.barh(y-h/2,sob.ST,h,xerr=sob.ST_conf,color="#D9822B",label="Total-effect (S$_{Ti}$)")
for yy,a,b in zip(y,sob.S1,sob.ST): ax.text(a+0.09,yy+h/2,f"{a:.3f}",va="center",fontsize=8); ax.text(b+0.09,yy-h/2,f"{b:.3f}",va="center",fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(sob.index); ax.set_xlim(0,1.15); ax.set_xlabel("Sobol index (output: Q$_{12h}$)")
ax.set_title("Sobol analysis of literature-calibrated kinetic constants",fontsize=10,fontweight="bold")
ax.spines[["top","right"]].set_visible(False); ax.legend(frameon=False,fontsize=8,loc="center right")
plt.tight_layout(); plt.savefig("figures/Figure7_Sobol_kinetic.png")
