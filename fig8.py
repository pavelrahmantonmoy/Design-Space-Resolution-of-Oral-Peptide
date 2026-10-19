import numpy as np, pandas as pd, statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv('ccd_actual_stat_ease.csv'); d['a']=(d.A-25)/10; d['b']=(d.B-100)/50
M={y:smf.ols(f'{y}~a+b+a:b+I(a**2)+I(b**2)',d).fit() for y in ['Q','PW']}
A,B=np.meshgrid(np.linspace(15,35,801),np.linspace(50,150,801))
g=pd.DataFrame({'a':((A-25)/10).ravel(),'b':((B-100)/50).ravel()})
Q=M['Q'].predict(g).values.reshape(A.shape); P=M['PW'].predict(g).values.reshape(A.shape)
dq=np.where(Q<48.35,np.clip((Q-28.6)/(48.35-28.6),0,1),np.clip((68.1-Q)/(68.1-48.35),0,1)); dp=np.clip((P-1.3)/3.8,0,1); D=np.sqrt(dq*dp)
i=np.unravel_index(D.argmax(),D.shape); oa,ob=A[i],B[i]
def panel(ax,Z,title,cmap,lv):
    cf=ax.contourf(A,B,Z,levels=lv,cmap=cmap); cs=ax.contour(A,B,Z,levels=lv[::2],colors='k',linewidths=0.5); ax.clabel(cs,fontsize=7,fmt='%.1f')
    ax.scatter(d.A,d.B,s=18,c='white',edgecolors='k',zorder=3,label='Design points')
    ax.set_xlabel('A: HPMC K100M (% w/w)'); ax.set_ylabel('B: SNAC amount (mg/tablet)'); ax.set_title(title,fontsize=10,fontweight='bold'); plt.colorbar(cf,ax=ax)
fig,ax=plt.subplots(1,2,figsize=(11,4.4),dpi=300)
panel(ax[0],Q,'(a) R1: Q$_{12h}$ (%)','viridis',np.linspace(26,70,23))
panel(ax[1],P,'(b) R2: Enzymatic protection window (h)','magma',np.linspace(1,5.2,22))
plt.tight_layout(); plt.savefig('Figure8a_contours.png')
fig,ax=plt.subplots(figsize=(6,4.6),dpi=300)
panel(ax,D,'(c) Overall desirability D','YlGn',np.linspace(0,1,21))
ax.plot(oa,ob,'r*',ms=14,zorder=4); ax.annotate(f'Optimum\nA = {oa:.2f}%, B = {ob:.0f} mg\nD = {D.max():.3f}',(oa,ob),xytext=(26.5,118),fontsize=8,arrowprops=dict(arrowstyle='->'),bbox=dict(fc='white',ec='0.5'))
plt.tight_layout(); plt.savefig('Figure8b_desirability.png'); print(oa,ob,D.max())
