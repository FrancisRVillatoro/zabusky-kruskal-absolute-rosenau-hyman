"""Unified catalogue of returns, return times and fifth-harmonic slice coefficients.

Read-only postprocessing of the archives of the repository.  A "return" is a chronological
local minimum of D with prominence >= 0.03 after tau = 0.2 (the rule used for the
length sweep in the article).  Slice coefficients are 2 a_n exp(-i n arg a_1).
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.signal import find_peaks
from common import *


def events_from_trace(x, nmax_events=6):
    t = x['tau']; D = x['D']; c = x['c']; n = x['n']
    keep = t <= 4.25 + 1e-12
    t, D, c, r, H = t[keep], D[keep], c[keep], x['r'][keep], x['H'][keep]
    out = []
    for j, (i, prom) in enumerate(prominent_minima(t, D), 1):
        s = slice_coefficients(c[i], n)
        e = 4 * np.abs(c[i]) ** 2
        row = dict(j=j, tau=float(t[i]), D=float(D[i]), prominence=prom, r=float(r[i]), H=float(H[i]))
        for m in (3, 5, 7, 9):
            k = np.flatnonzero(n == m)
            if len(k):
                k = k.item()
                row[f'e{m}'] = float(e[k]); row[f'amp{m}'] = float(abs(s[k])); row[f'theta{m}_deg'] = float(np.degrees(np.angle(s[k])))
        row['H_minus_e5'] = row['H'] - row.get('e5', np.nan)
        out.append(row)
        if j >= nmax_events:
            break
    return out


def minima_from_history(t, D, **kw):
    t = np.asarray(t); D = np.asarray(D)
    ids, props = find_peaks(-D, prominence=0.03)
    return [(float(t[i]), float(D[i]), float(p)) for i, p in zip(ids, props['prominences']) if t[i] >= 0.2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); a = ap.parse_args()
    S = Archives(a.root)
    cat = []
    # --- Pade/Radau traces: series B (uniform step) and series A (archived base-step histories)
    full = {}
    for mem, tag in (('results/report/summary/free_events.csv', 'B'), ('results/archive/summary/free_events.csv', 'A')):
        for r in S.rows(ZB, mem):
            full[(r['case'] if 'case' in r and r['case'] else '', int(r['j']))] = r
    arch_rows = S.rows(ZB, 'results/archive/summary/free_events.csv')
    for mem in sorted(S.names(ZB)):
        if not mem.endswith('.npz'):
            continue
        x = S.npz(ZB, mem); meta = json.loads(str(x['meta'])); case = Path(mem).stem
        series = 'B' if 'fixed step' in meta.get('history', '') else 'A'
        Nt = 3200 * int(meta['dt_divisor']) if series == 'B' else 'mixed'
        for ev in events_from_trace(x):
            cat.append(dict(case=case, series=series, method='Pade-Radau', projected=True, p=2.25, nstar=float(meta['nstar']), N=int(meta['N']),
                            Nt=Nt, nu=float(meta['nu_explicit']), **ev))
    # --- Galerkin plot traces
    for mem in sorted(S.names(ZBL)):
        if '/plot_traces/' not in mem or not mem.endswith('.npz'):
            continue
        x = S.npz(ZBL, mem); meta = json.loads(str(x['meta'])); case = Path(mem).stem
        for ev in events_from_trace(x):
            cat.append(dict(case=case, series='G', method='Galerkin' + ('' if meta['probe'] == 'none' else f" probe {meta['probe']}"), projected=True, p=2.25,
                            nstar=float(meta['nstar']), N=int(meta['cutoff']), Nt='adaptive', nu=0.0, **ev))
    # --- AVF long runs: JSON diagnostics plus slice phase from the all-node fields at the returns
    for N in (450, 900, 1800):
        for K in (3200, 6400, 12800):
            mem = f'controls/long_results/p2.25_M4_N{N}_K{K}.npz'
            if mem not in S.names(ZBASE):
                continue
            x = S.npz(ZBASE, mem); meta = S.json(ZBASE, mem.replace('.npz', '.json'))
            for j in (1, 2, 3, 4):
                info = meta['returns'].get(str(j))
                if info is None or f'u_at_return_{j}' not in x:
                    continue
                u = x[f'u_at_return_{j}']; aa = np.fft.rfft(u) / N
                n = np.arange(1, 10, 2); s = slice_coefficients(aa[n], n); e = 4 * np.abs(aa[n]) ** 2
                row = dict(case=f'AVF_N{N}_K{K}', series='AVF', method='Pade-AVF', projected=True, p=2.25, nstar=7.2, N=N, Nt=K, nu=0.0, j=j,
                           tau=info['tau'], D=info['D'], r=info['r'], H=info['H'], prominence='')
                for k, m in enumerate(n):
                    if m > 1:
                        row[f'e{m}'] = float(e[k]); row[f'amp{m}'] = float(abs(s[k])); row[f'theta{m}_deg'] = float(np.degrees(np.angle(s[k])))
                row['H_minus_e5'] = row['H'] - row['e5']
                row['max_I_drift'] = meta['max_absolute_relative_invariant_change']
                cat.append(row)
    # --- unprojected Radau: base-archive experiment (N=900, Nt=3200) and Fase9C nonlinear runs with eps=0
    for case in ('free900', 'project900', 'project450', 'pert900'):
        rows = S.rows(ZBASE, f'experiments/results/{case}/history.csv')
        t = np.array([fnum(r['tau']) for r in rows]); D = np.array([fnum(r['D']) for r in rows])
        H = np.array([fnum(r['H']) for r in rows]); rr = np.array([fnum(r['r']) for r in rows]); e5H = np.array([fnum(r['e5_over_H']) for r in rows])
        hd = np.array([fnum(r['half_defect']) for r in rows])
        st = S.json(ZBASE, f'experiments/results/{case}/status.json')
        for j, (i, prom) in enumerate(prominent_minima(t, D), 1):
            cat.append(dict(case=case, series='X', method='Pade-Radau', projected=bool(st['project']), p=2.25, nstar=7.2, N=int(st['N']), Nt=int(st['k']), nu=0.0,
                            epsilon=st['epsilon'], j=j, tau=float(t[i]), D=float(D[i]), prominence=prom, r=float(rr[i]), H=float(H[i]),
                            e5=float(e5H[i] * H[i]), H_minus_e5=float(H[i] * (1 - e5H[i])), half_defect=float(hd[i])))
    gt = S.rows(ZC, 'report/gain_traces.csv')
    by = {}
    for r in gt:
        by.setdefault(r['case'], []).append(r)
    camp = {r['case']: r for r in S.rows(ZC, 'campaign.csv')}
    for case, rows in by.items():
        info = camp[case]
        if float(info['p']) != 2.25:
            continue
        t = np.array([fnum(r['tau']) for r in rows]); D = np.array([fnum(r['D']) for r in rows]); ev = np.array([fnum(r['even_rms']) for r in rows])
        Irel = np.array([fnum(r['I_relative']) for r in rows])
        if t[-1] < 4.0:
            continue
        for j, (i, prom) in enumerate(prominent_minima(t, D), 1):
            if j > 4:
                break
            cat.append(dict(case=case, series='C', method='Pade-' + ('Radau' if info['method'] == 'radau' else 'AVF'), projected=info['kind'] == 'tangent',
                            p=2.25, nstar=7.2, N=int(info['N']), Nt=int(info['K']), nu=float(info['nu']), epsilon=float(info['eps']), j=j,
                            tau=float(t[i]), D=float(D[i]), prominence=prom, even_rms=float(ev[i]), I_relative=float(Irel[i])))
    write_csv(RESULTS / 'return_catalog.csv', cat)

    # --- first-return time versus length (Pade data only), including the earlier length controls
    scal = []
    def add(case, p, nstar, N, Nt, t1, D1, src, e_dom=''):
        scal.append(dict(case=case, p=p, nstar=nstar, N=N, Nt=Nt, tau1=t1, D1=D1, tau1_over_nstar=t1 / nstar,
                         t1_k1sq=2 * np.pi * t1 / ((p + 1) / (2 * p) * nstar), source=src, note=e_dom))
    for r in S.rows(ZB, 'results/report/summary/free_events.csv'):
        if int(r['j']) == 1:
            add(r['case'], 2.25, float(r['nstar']), int(r['N']), 3200 * int(r['dt_divisor']), float(r['tau']), float(r['D']), 'Fase9B free_events.csv')
    for case, p, M in (('short336', 2.25, 3), ('short672', 2.25, 3), ('long560', 2.25, 5), ('long1120', 2.25, 5), ('null900', 7 / 3, 4), ('cubic450', 3.0, 4), ('cubic900', 3.0, 4),
                       ('project450', 2.25, 4), ('project900', 2.25, 4)):
        rows = S.rows(ZBASE, f'experiments/results/{case}/history.csv')
        t = [fnum(r['tau']) for r in rows]; D = [fnum(r['D']) for r in rows]
        if f'experiments/results/{case}/extension.csv' in S.names(ZBASE):
            ext = S.rows(ZBASE, f'experiments/results/{case}/extension.csv')
            t += [fnum(r['tau']) for r in ext]; D += [fnum(r['D']) for r in ext]
        st = S.json(ZBASE, f'experiments/results/{case}/status.json')
        nstar = M * p / (p - 1)
        mins = minima_from_history(t, D)
        if mins:
            add(case, p, nstar, int(st['N']), int(st['k']), mins[0][0], mins[0][1], 'base experiments history.csv', f'all prominent minima: {[(round(a,4), round(b,4)) for a,b,_ in mins]}')
        else:
            scal.append(dict(case=case, p=p, nstar=nstar, N=int(st['N']), Nt=int(st['k']), tau1='', D1='', source='base experiments', note=f'no prominent minimum up to tau={t[-1]:.3f}'))
    # Galerkin lengths already in the archives
    for r in S.rows(ZBL, 'fase9b/results_local/summary/free_events.csv'):
        if int(r['j']) == 1 and r.get('probe', 'none') == 'none':
            add(r['case'], 2.25, float(r['nstar']), int(r['cutoff']), 'Galerkin', float(r['tau']), float(r['D']), 'Fase9B local Galerkin free_events.csv')
    write_csv(RESULTS / 'return_time_scaling.csv', scal)
    for s in scal:
        if s.get('tau1', '') != '':
            print(f"{s['case']:18s} p={s['p']:.3f} n*={s['nstar']:.3f} N={s['N']:5d} tau1={s['tau1']:.5f} D1={s['D1']:.4f} tau1/n*={s['tau1_over_nstar']:.5f} t1k1^2={s['t1_k1sq']:.4f}  {s['note'][:110]}")
        else:
            print(f"{s['case']:18s} {s['note']}")


if __name__ == '__main__':
    main()
