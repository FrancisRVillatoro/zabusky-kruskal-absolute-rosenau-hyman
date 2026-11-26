"""Check headline numerical results against the actual source exports."""
import io,json,zipfile
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
A=R/'archives'
def rd(z,p):
    with zipfile.ZipFile(A/z) as f:return pd.read_csv(io.BytesIO(f.read(p)))
def js(z,p):
    with zipfile.ZipFile(A/z) as f:return json.loads(f.read(p))
def close(a,b,rtol=1e-6,atol=1e-12):assert np.allclose(a,b,rtol=rtol,atol=atol),(a,b)
for z in A.glob('*.zip'):
    with zipfile.ZipFile(z) as f:assert f.testzip() is None,z
zA='Fase9A_HDF5_3831810.zip';zB='Fase9B_CNSNS_resultados_20261125.zip';zC='Fase9C_CNSNS_resultados_20261125.zip'
a=js(zA,'status.json');assert (a['completed'],a['common_profiles'],a['replay_profiles_compared'])==(42,42,12)
b=js(zB,'results/report/status.json');assert (b['expected'],b['complete'])==(14,13)
c=js(zC,'report/status.json');assert (c['expected'],c['complete'],c['valid_tangent'],c['valid_nonlinear'])==(42,41,23,19)
d=rd(zA,'minima/summary.csv');d=d[d.nu==0];e=d.groupby('j')[['tau','D','r','H','e5_over_H']].agg(['min','max']);e.columns=['_'.join(x) for x in e.columns];e.to_csv(R/'return_envelopes.csv');close(e.D_min.iloc[0],.04529883484885811);close(e.D_max.iloc[-1],.245942,rtol=5e-6)
# Central, uniform-step result and independent temporal tangents.
r=rd(zB,'results/report/summary/free_events.csv');r=r[r.case=='s7.2_N1800_d4'].sort_values('j');close(r.D,[.046072,.077754,.128703,.238326],rtol=2e-5);assert np.all(np.diff(r.D)>0)
q=rd(zB,'results/report/summary/mode_statistics.csv');q=q[q.n==7];ratio=q.loc[q.case=='s7.0_N900_d2','mean_e'].iloc[0]/q.loc[q.case=='s7.0_N1800_d2','mean_e'].iloc[0];close(ratio,255.17953918179614)
x=rd(zB,'results/report/summary/exchange.csv');x=x[(x.case=='s7.2_N1800_d4')&(x.selection=='prominent_chronological')&(x.n==5)].sort_values('j');assert x.defect.abs().max()<1.1e-7;assert np.all(np.diff(x.net/x.gross_gain)>0)
t=rd(zC,'report/return_gains.csv');t=t[(t.j==4)&t.case.str.match(r'p2\.25_M4_N1800_K(12800|25600)_nu0_(radau|avf)_tangent')];assert len(t)==4;assert t.D.max()/t.D.min()-1<.025;assert t.log10_G_cos2.max()-t.log10_G_cos2.min()>120
s=rd('CNSNS_Fase9A_20261124.zip','fase9a/results_local/scaling_summary.csv');z=s[s.p==2.25].iloc[0];assert abs(z.core_min-.25)<1e-5 and .08<z.global_min<.09
print(json.dumps({'archives_checked':len(list(A.glob('*.zip'))),'profiles':a['completed'],'length_complete':b['complete'],'tangent_study_complete':c['complete'],'null_power_ratio':ratio,'mode5_max_balance_defect':x.defect.abs().max(),'D4_range':list(map(float,[t.D.min(),t.D.max()])),'log10_gain_range':list(map(float,[t.log10_G_cos2.min(),t.log10_G_cos2.max()]))},indent=2))
