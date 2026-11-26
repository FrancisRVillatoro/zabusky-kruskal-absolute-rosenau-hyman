"""Numbers quoted in the article, recomputed from the archives and from results/.

Prints LaTeX rows for the tables and the in-text values, so that every figure in
the text can be traced to a file.  Run after the other scripts.
"""
import sys, json, argparse, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from common import *


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); a = ap.parse_args()
    S = Archives(a.root)
    cat = pd.read_csv(RESULTS / 'return_catalog.csv')
    def ret(series, case):
        d = cat[(cat.series == series) & (cat.case == case)].sort_values('j'); return d[d.j <= 4]
    def f3(x): return f'{x:.3f}'.lstrip('0') if x < 1 else f'{x:.3f}'
    def f4(x): return f'{x:.4f}'.lstrip('0')
    def table(zipname, suffix):
        return pd.read_csv(io.BytesIO(S.bytes(zipname, [n for n in S.names(zipname) if n.endswith(suffix)][0])))
    def at(d, t, col): return float(d.loc[(d.tau - t).abs().idxmin(), col])

    print('=== Table: returns at n*=7.2 across discretizations')
    sets = [('B', 's7.2_N1800_d4', 'Pad\\\'e--Radau, projected', '1800', '12800'), ('B', 's7.2_N900_d2', 'Pad\\\'e--Radau, projected', '900', '6400'),
            ('A', 'archive_nu0.00000_N450', 'Pad\\\'e--Radau, projected', '450', '3200'),
            ('C', 'p2.25_M4_N1800_K12800_nu0_radau_tangent', 'Pad\\\'e--Radau, projected', '1800', '12800 (C)'),
            ('C', 'p2.25_M4_N1800_K12800_nu0_radau_nonlinear_eps+0e+00', 'Pad\\\'e--Radau, no projection', '1800', '12800'),
            ('C', 'p2.25_M4_N900_K12800_nu0_radau_nonlinear_eps+0e+00', 'Pad\\\'e--Radau, no projection', '900', '12800'),
            ('AVF', 'AVF_N1800_K12800', 'Pad\\\'e--AVF, projected', '1800', '12800'), ('AVF', 'AVF_N1800_K3200', 'Pad\\\'e--AVF, projected', '1800', '3200'),
            ('G', 'central_K21', 'Galerkin', '21', 'adaptive'), ('G', 'central_K35', 'Galerkin', '35', 'adaptive'), ('G', 'central_K51', 'Galerkin', '51', 'adaptive')]
    for s, c, lab, N, Nt in sets:
        d = ret(s, c)
        print(f"{lab} & {N} & {Nt} & " + ' & '.join(f4(v) for v in d.D) + f" & {d.tau.iloc[-1]:.3f}\\\\")
    for c in ('free900', 'project900'):
        d = ret('X', c); print('  ', c, [f'{v:.10f}' for v in d.D])

    print('=== Table: lengths')
    sc = pd.read_csv(RESULTS / 'return_time_scaling.csv')
    for case in ('short336', 'short672', 'long560', 'long1120', 'null900', 'cubic450', 'cubic900'):
        r = sc[sc.case == case].iloc[0]; print(f"   {case}: p={r.p:.4f} n*={r.nstar} tau1={r.tau1:.4f} D1={r.D1:.4f} tau1/n*={r.tau1_over_nstar:.4f}")
    for ell in ('6.8', '7.0', '7.2', '7.4'):
        d = ret('B', f's{ell}_N1800_d4'); print(f"{float(ell):g} & {float(ell)/1.8:.2f} & Pad\\'e, 1800 & {d.tau.iloc[0]:.3f} & {d.tau.iloc[0]/float(ell):.3f} & " + ' & '.join(f3(v) for v in d.D) + f"   [tau4={d.tau.iloc[-1]:.3f}]")
    lr = pd.read_csv(RESULTS / 'length_returns.csv')
    for ns in sorted(lr.nstar.unique()):
        d = lr[lr.nstar == ns]
        print(f"{ns:g} & {ns/1.8:.2f} & Galerkin, {int(d.cutoff.iloc[0])} & " + '; '.join(f"tau={r.tau:.3f} D={r.D:.3f} (tau/n*={r.tau/ns:.4f}, dom {int(r.dominant_harmonic)}: {100*r.dominant_share:.0f}%)" for r in d.itertuples()))
    z = np.load(RESULTS / 'length_curves.npz'); print('   max D for n*=3.6:', float(z['3.6'][1].max()))

    print('=== first excursion and momentum, series B n*=7.2 N=1800 Nt=12800')
    tr = S.npz(ZB, 'results/report/plot_traces/s7.2_N1800_d4.npz'); tau = tr['tau']; c = tr['c']; n = tr['n']; e = 4 * np.abs(c) ** 2
    i = int(np.argmin(np.abs(tau - 0.5)))
    print(f"   tau=0.5: e1={e[i,0]:.3f} e3={e[i,1]:.3f} e5={e[i,2]:.3f} e7={e[i,3]:.4f} sum_all={e[i].sum():.3f} share135={(e[i,:3].sum()/e[i].sum()):.3f}")
    print(f"   max e7={e[:,3].max():.4f}; min total power={e.sum(axis=1).min():.3f} at tau={tau[e.sum(axis=1).argmin()]:.3f}; max total={e.sum(axis=1).max():.4f}; min r={tr['r'].min():.3f}; max D={tr['D'].max():.3f}")
    w = momentum_weights(n, 7.2); print('   weights n=1..13:', ', '.join(f'{int(k)}:{v:.3f}' for k, v in zip(n[:7], w[:7])), ' Q(0)=', 1 / (1 - (1 / 7.2) ** 2))
    print(f"   momentum balance at tau=0.5 from harmonics 1,3,5: {w[0]*(e[i,0]-1)+w[1]*e[i,1]+w[2]*e[i,2]:+.4f}")
    mt = pd.read_csv(RESULTS / 'momentum_traces.csv')
    for r in mt.itertuples():
        print(f"   {r.case:28s} {r.method:12s} N={r.N} n*={r.nstar}: max|Q/Q0-1| = {r.max_rel_dev_stored:.2e}, power range {r.L2_power_min:.3f}-{r.L2_power_max:.3f}")
    mf = pd.read_csv(RESULTS / 'momentum_field_returns.csv'); print('   field momentum at returns (columns):', list(mf.columns)[:14])
    print(mf.to_string(index=False, max_cols=14, max_rows=60)[:4000])

    print('=== mode 7 ratio across the sweep')
    f = RESULTS / 'mode7_ratio_sweep.csv'
    if f.exists(): print(pd.read_csv(f).to_string(index=False))
    for ell in ('6.8', '7.0', '7.2', '7.4'):
        t = S.npz(ZB, f'results/report/plot_traces/s{ell}_N1800_d4.npz'); print(f"   n*={ell}: max e7 = {(4*np.abs(t['c'][:,3])**2).max():.5f}")

    print('=== Table: fifth harmonic at the returns')
    sets = [('B', 's7.2_N1800_d4', "7.2 & Pad\\'e--Radau, 1800"), ('B', 's7.2_N900_d2', "7.2 & Pad\\'e--Radau, 900"), ('A', 'archive_nu0.00000_N450', "7.2 & Pad\\'e--Radau, 450"),
            ('AVF', 'AVF_N1800_K12800', "7.2 & Pad\\'e--AVF, 1800"), ('G', 'central_K51', '7.2 & Galerkin, 51'), ('G', 'central_K21', '7.2 & Galerkin, 21'),
            ('B', 's6.8_N1800_d4', "6.8 & Pad\\'e--Radau, 1800"), ('B', 's7.0_N1800_d4', "7.0 & Pad\\'e--Radau, 1800"), ('B', 's7.4_N1800_d4', "7.4 & Pad\\'e--Radau, 1800"),
            ('A', 'archive_nu0.00025_N1800', '7.2, nu=0.00025'), ('A', 'archive_nu0.00100_N1800', '7.2, nu=0.001'), ('A', 'archive_nu0.00400_N1800', '7.2, nu=0.004')]
    for s, cse, lab in sets:
        d = ret(s, cse); amp = d.amp5.to_numpy()
        print(f"{lab} & " + ' & '.join(f4(v) for v in amp) + ' & ' + ' & '.join(f'${v:.0f}$' for v in d.theta5_deg) + f"\\\\   ratios {amp[1]/amp[0]:.2f}, {amp[2]/amp[1]:.2f}, {amp[3]/amp[2]:.2f}; H-e5: " + ', '.join(f'{v:.5f}' for v in d.H_minus_e5) + '; e5/H: ' + ', '.join(f'{v:.3f}' for v in d.e5 / d.H) + '; r: ' + ', '.join(f'{v:.4f}' for v in d.r) + '; e5: ' + ', '.join(f'{v:.2e}' for v in d.e5))
    b = S.rows(ZBC, 'CNSNS_Fase9B_cierre_20261125/central_mode5_balance.csv'); print('   mode-5 balance columns:', list(b[0].keys())); [print('   ', {k: r[k] for k in list(r)[:8]}) for r in b]

    print('=== relative periodic orbits')
    f = RESULTS / 'rpo_orbits.json'
    if f.exists():
        for r in json.loads(f.read_text()):
            ev = np.array([complex(x, y) for x, y in r['multipliers']]); mods = np.abs(ev)
            others = sorted({round(float(m), 3) for m in mods[1:] if m > 1.005}, reverse=True)
            sh = r['unstable_vector_power_share']; ph = r['unstable_vector_phase_deg']
            print(f"{r['cutoff']} & {len(r['alpha'])} & {r['period']:.4f} & {r['distance_to_cosine']:.4f} & {r['leading_multiplier']:.3f} & {', '.join(f'{m:.3f}' for m in others) or '--'}\\\\   closure {r['closure_defect']:.1e}, share3 {100*sh['3']:.1f}%, share5 {100*sh['5']:.1f}%, phase5 {ph['5']:.1f}, alpha9..17: " + ', '.join(f'{v:+.4f}' for v in r['alpha'][4:9]) + f"; residual {r['newton_residuals'][-1]:.1e}")
    f = RESULTS / 'rpo_linear_check.csv'
    if f.exists(): print(pd.read_csv(f).round(4).to_string(index=False))

    print('=== gains in the Galerkin ladder')
    f = RESULTS / 'gain_ladder.csv'
    if f.exists():
        g = pd.read_csv(f)
        for K in sorted(g.cutoff.unique()):
            q = g[g.cutoff == K].sort_values('tau')
            print(f"{K} & " + ' & '.join(f'{v:.3g}' for v in q.gain_even) + ' & ' + ' & '.join(f'{v:.3g}' for v in q.gain_odd) + '\\\\   share above 20: ' + ', '.join(f'{100*v:.0f}%' for v in q.even_power_share_above_20))

    print('=== finite perturbations and tangent map, N=1800, Nt=12800, Radau')
    base = 'p2.25_M4_N1800_K12800_nu0_radau_'
    gt = table(ZC, 'report/gain_traces.csv'); nc = table(ZC, 'report/nonlinear_comparisons.csv'); nv = table(ZCC, 'nonlinear_validity.csv')
    tn = gt[gt.case == base + 'tangent']
    print('   log10 G2 of the tangent map at tau = 0.25, 0.5, 0.6, 1, 2, 3, 4: ' + ', '.join(f"{at(tn, t, 'log10_G_cos2'):.2f}" for t in (0.25, 0.5, 0.6, 1, 2, 3, 4)))
    for eps in ('+1e-08', '+1e-07', '+1e-06', '-1e-06'):
        case = base + 'nonlinear_eps' + eps
        q = gt[gt.case == case].sort_values('tau'); c = nc[nc.nonlinear == case]; v = nv[nv.case == case].iloc[0]
        late = q[q.tau >= 2].even_rms
        print(f"   eps={eps}: within 1% of the tangent vector until tau={v.first_threshold_violation_tau:g}; log10 gain at tau=1: {at(c, 1, 'log10_G_nonlinear'):.2f} (vector difference {100 * at(c, 1, 'relative_vector_difference'):.0f}%); "
              f"first tau with r.m.s. >= 1e-3: {float(q[q.even_rms >= 1e-3].tau.iloc[0]):.3f}; r.m.s. for tau >= 2: {late.min():.2e} to {late.max():.2e}; D4 = {q[(q.tau > 3.9) & (q.tau < 4.15)].D.min():.4f}")
    q = gt[gt.case == base + 'nonlinear_eps+0e+00'].sort_values('tau').reset_index(drop=True); D = q.D.to_numpy(); best = {}
    for i in range(1, len(q) - 1):
        if q.tau[i] > 0.2 and D[i] < 0.5 and D[i] < D[i - 1] and D[i] <= D[i + 1]:
            j = int(round(q.tau[i])); best[j] = i if j not in best or D[i] < D[best[j]] else best[j]
    print(f"   eps=0: r.m.s. of the even component {at(q, 0.25, 'even_rms'):.1e} at tau=0.25; at the returns: " + ', '.join(f"tau={q.tau[i]:.4f}: {q.even_rms[i]:.2e}" for j, i in sorted(best.items())) + f"; maximum {q.even_rms.max():.2e}; D4 = {D[best[4]]:.4f}")
    rp = table(ZC, 'report/replay_9b.csv')
    print(f"   two computations of the projected orbit: difference {at(rp, 1, 'normalized_L2'):.2e} at tau=1, {at(rp, 2, 'normalized_L2'):.2e} at tau=2, maximum {rp.normalized_L2.max():.4e} ({len(rp)} comparisons)")
    for r in table(ZCC, 'regularized_final_gains.csv').sort_values(['nu', 'N']).itertuples():
        print(f"   regularized tangent map: nu={r.nu:g}, N={r.N}: G2 = {10 ** r.log10_G_cos2:.1f} at tau={r.tau:g}")

    print('=== lattice coefficient of the initial velocity, N=65536')
    sc = table(ZAL, 'results_local/initial_scaling.csv'); sc = sc[(sc.family == 'Z') & (sc.N == sc.N.max()) & (sc.p < 2.5)]
    print(f"   largest relative difference between the 17-node coefficient and the lattice sum: {(sc.scaled_core / sc.lattice_core_prediction - 1).abs().max():.1e} ({sc.p.nunique()} exponents, {sc.alpha.nunique()} offsets)")
    for p in sorted(sc.p.unique()):
        d = sc[sc.p == p].sort_values('alpha'); print(f"   p={p:g}: coefficient relative to offset 0: " + ', '.join(f'{a:g}: {v:.4f}' for a, v in zip(d.alpha, d.scaled_core / d.scaled_core.iloc[0])))

    print('=== residual near the zeros at the returns, series A, N=1800, nu=0')
    for r in table(ZAC, 'finest_returns.csv').query("nu == 0 and selection == 'minima'").sort_values('j').itertuples():
        print(f"   j={r.j}: |u|<=0.2 covers {100 * r.area_amp:.1f}% of the domain and holds {100 * r.fraction_amp:.1f}%; within 0.4 of a zero ({100 * r.area_rad:.1f}%): {100 * r.fraction_rad:.1f}%; squared residual inside {r.residual_amp:.3e}, outside {r.residual_outside_amp:.3e}, within 0.4: {r.residual_rad:.3e}")

    print('=== zero structure (nu=0 unless stated)')
    zs = pd.read_csv(RESULTS / 'zero_summary.csv')
    print(zs[(zs.nu == 0)].round(4).to_string(index=False))
    print(zs[(zs.nu > 0) & (zs.N == 1800)].round(4).to_string(index=False))

    print('=== initial layer')
    for m in json.loads((RESULTS / 'initial_layer.json').read_text()):
        at = m['at']; s = lambda x, y: np.log(at[y]['sup'] / at[x]['sup']) / np.log(float(y) / float(x))
        print(f"   p={m['p']} N={m['N']}: resolved t>>{m['resolved_time']:.1e}; ||F_h||={m['F0_norm']:.4f}; sup(0.01)={at['0.01']['sup']:.4e} sup(0.1)={at['0.1']['sup']:.4e} sup(1)={at['1']['sup']:.4e}; slopes [0.01,0.1] {s('0.01','0.1'):.3f}, [0.03,0.3] {s('0.03','0.3'):.3f}; predicted {1/(4-m['p']):.3f}; early {m['slope_sup_early']:.3f}")
    pr = pd.read_csv(RESULTS / 'initial_layer_profiles.csv')
    for p in (1.75, 2.0, 2.25):
        for t in (0.01, 0.03, 0.1, 0.3):
            q = pr[(pr.p == p) & (pr.N == 8192) & np.isclose(pr.t, t)].sort_values('xi'); q = q[np.abs(q.xi) <= 6]
            print(f"   p={p} t={t}: min of scaled profile {q.scaled.min():.4f} at y/y*={q.xi.iloc[q.scaled.argmin()]:+.2f}; integral of square over |y/y*|<6: {np.trapezoid(q.scaled**2, q.xi):.3f}")


if __name__ == '__main__':
    main()
