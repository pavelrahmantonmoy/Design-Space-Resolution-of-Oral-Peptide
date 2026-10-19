import numpy as np, pandas as pd, statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
d=pd.DataFrame({'Run':range(1,14),'Std':[5,8,13,9,10,6,7,4,1,11,2,3,12],
 'A':[17.93,32.07,17.93,32.07,15,35,25,25,25,25,25,25,25],
 'B':[64.64,64.64,135.36,135.36,100,100,50,150,100,100,100,100,100],
 'Q':[62.8,34.2,59.6,32.9,68.1,28.6,47.5,44.3,47,46.4,46,45.9,46.3],
 'PW':[1.8,2.3,4.2,4.7,2.9,3.4,1.3,5.1,4,3.7,4.2,3.5,4.2]})
d.to_csv('ccd_actual_stat_ease.csv',index=False)
d['a']=(d.A-25)/10; d['b']=(d.B-100)/50   # Stat-Ease coding: +/-1 = 15/35, 50/150
res={}
for y in ['Q','PW']:
    m=smf.ols(f'{y}~a+b+a:b+I(a**2)+I(b**2)',d).fit()
    c=d[(d.a==0)&(d.b==0)][y]; pe=((c-c.mean())**2).sum()
    print(y,'mean %.2f R2 %.4f adj %.4f SSmodel %.2f SSres %.4f PE %.4f F %.2f'%(d[y].mean(),m.rsquared,m.rsquared_adj,m.ess,m.ssr,pe,m.fvalue))
    print('  coded',m.params.round(4).to_dict()); print('  p',m.pvalues.round(4).to_dict())
    res[y]=m
# desirability as configured: Q target 48.35 (28.6-68.1), PW maximize (1.3-5.1), A,B in range
A,B=np.meshgrid(np.linspace(15,35,401),np.linspace(50,150,401))
g=pd.DataFrame({'a':((A-25)/10).ravel(),'b':((B-100)/50).ravel()})
q=res['Q'].predict(g).values; pw=res['PW'].predict(g).values
dq=np.where(q<48.35,np.clip((q-28.6)/(48.35-28.6),0,1),np.clip((68.1-q)/(68.1-48.35),0,1))
dp=np.clip((pw-1.3)/(5.1-1.3),0,1); D=np.sqrt(dq*dp)
i=D.argmax(); print('OPT D=%.4f A=%.2f B=%.1f Q=%.2f PW=%.3f dQ=%.4f dPW=%.4f'%(D[i],A.ravel()[i],B.ravel()[i],q[i],pw[i],dq[i],dp[i]))
j=np.argmin(abs(g.a)+abs(g.b)); print('center D=%.4f Q=%.2f PW=%.2f'%(D[j],q[j],pw[j]))

from statsmodels.stats.anova import anova_lm
rows=[]
for y,lab in [('Q','R1: Q12h'),('PW','R2: Protection window')]:
    m=res[y]; a=anova_lm(m,typ=3)
    c=d[(d.a==0)&(d.b==0)][y]; pe=((c-c.mean())**2).sum(); lof=m.ssr-pe
    msr=m.ssr/7; F_lof=(lof/3)/(pe/4)
    from scipy.stats import f as fd
    rows.append((lab,'Model',m.ess,5,m.ess/5,m.fvalue,m.f_pvalue))
    for t,n in [('a','A'),('b','B'),('a:b','AB'),('I(a ** 2)','A²'),('I(b ** 2)','B²')]:
        rows.append((lab,n,a.loc[t,'sum_sq'],1,a.loc[t,'sum_sq'],a.loc[t,'F'],a.loc[t,'PR(>F)']))
    rows.append((lab,'Residual',m.ssr,7,msr,None,None))
    rows.append((lab,'Lack of fit',lof,3,lof/3,F_lof,1-fd.cdf(F_lof,3,4)))
    rows.append((lab,'Pure error',pe,4,pe/4,None,None))
an=pd.DataFrame(rows,columns=['Response','Source','SS','df','MS','F','p']); an.to_csv('anova_actual.csv',index=False)
print(an.round(4).to_string())
