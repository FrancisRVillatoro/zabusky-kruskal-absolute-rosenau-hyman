"""Translation momentum Q = sum_n e_n/(1-k_n^2) on the archived trajectories.

Sources (read only):
  * Fase9B production: reduced modal traces (32 odd harmonics, n<=63) of the
    uniform-step Pade/Radau runs and of the archived base-step histories;
  * Fase9B local: Galerkin plot traces (all retained harmonics);
  * Fase9A HDF5 exports: all-node fields at the return minima;
  * base archive: AVF fields at the returns.
For the reduced traces the harmonics above 63 are not stored.  Their total power
is H - sum_{3<=n<=63} e_n and their weights lie in (w_65, 0), which brackets Q.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *


def trace_summary(x, case, origin, nstar, extra):
    t = x['tau']; c = x['c']; n = x['n']
    keep = t <= 4.25 + 1e-12
    t, c, H, r, D = t[keep], c[keep], x['H'][keep], x['r'][keep], x['D'][keep]
    e = 4 * np.abs(c) ** 2
    w = momentum_weights(n, nstar)
    w = np.where(np.abs(w) < 1e6, w, 0.0)          # exclude a harmonic at (or within roundoff of) the null
    Qs = (e * w).sum(axis=1)
    tail = np.clip(H - e[:, 1:].sum(axis=1), 0, None)   # power not stored (n>63, or even harmonics)
    wtail = momentum_weights([n[-1] + 2], nstar)[0]     # most negative weight of the unstored harmonics
    Qlo = Qs + wtail * tail
    Q0 = Qs[0]
    mid = 0.5 * (Qs + Qlo)
    row = dict(case=case, origin=origin, nstar=nstar, **extra, n_stored_max=int(n[-1]), samples=len(t),
               Q0=Q0, max_rel_dev_stored=float(np.max(np.abs(Qs - Q0)) / Q0),
               max_rel_dev_mid=float(np.max(np.abs(mid - Q0)) / Q0),
               max_bracket_halfwidth=float(np.max(0.5 * np.abs(wtail) * tail)),
               tail_weight=float(wtail), max_tail_power=float(tail.max()),
               L2_power_min=float((r ** 2 + H).min()), L2_power_max=float((r ** 2 + H).max()),
               positive_part_max=float((e * np.clip(w, 0, None)).sum(axis=1).max()),
               negative_part_min=float((e * np.clip(w, None, 0)).sum(axis=1).min()))
    ev = []
    for j, (i, prom) in enumerate(prominent_minima(t, D), 1):
        ev.append(dict(case=case, origin=origin, nstar=nstar, **extra, j=j, tau=float(t[i]), D=float(D[i]),
                       Q_minus_Q0_stored=float(Qs[i] - Q0), Q_minus_Q0_low=float(Qlo[i] - Q0),
                       one_minus_e1=float(1 - e[i, 0]),
                       sum_rule_e5=float(w[2] / w[0] * e[i, 2]) if len(n) > 2 and n[2] == 5 else np.nan,
                       weighted_n3=float(w[1] * e[i, 1]), weighted_n5=float(w[2] * e[i, 2]),
                       weighted_n7=float(w[3] * e[i, 3]), weighted_n9plus=float((w[4:] * e[i, 4:]).sum()),
                       L2_power=float(r[i] ** 2 + H[i])))
    return row, ev, (t, Qs, Qlo, r ** 2 + H)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); a = ap.parse_args()
    S = Archives(a.root)
    rows, events, curves = [], [], {}
    # 1. Pade/Radau uniform-step runs (series B) and archived base-step histories (series A).
    for m in sorted(S.names(ZB)):
        if not m.endswith('.npz'):
            continue
        x = S.npz(ZB, m); meta = json.loads(str(x['meta'])); case = Path(m).stem
        nstar = float(meta.get('nstar'))
        extra = dict(method='Pade-Radau', N=int(meta['N']), nu=float(meta['nu_explicit']),
                     history=meta.get('history', ''), dt_divisor=meta.get('dt_divisor', ''))
        row, ev, cur = trace_summary(x, case, 'series B' if 'fixed step' in extra['history'] else 'series A', nstar, extra)
        rows.append(row); events += ev; curves[case] = cur
    # 2. Galerkin plot traces (exact Hamiltonian truncations up to quadrature and tolerance).
    for m in sorted(S.names(ZBL)):
        if '/plot_traces/' not in m or not m.endswith('.npz'):
            continue
        x = S.npz(ZBL, m); meta = json.loads(str(x['meta'])); case = Path(m).stem
        if meta['probe'] != 'none':
            continue                                  # modified multipliers: not the model of the PDE
        extra = dict(method='Galerkin-DOP853', N=int(meta['cutoff']), nu=0.0, history=f"quad {meta['quad']}, rtol {meta['rtol']}", dt_divisor='')
        row, ev, cur = trace_summary(x, case, 'Galerkin', float(meta['nstar']), extra)
        rows.append(row); events += ev; curves[case] = cur
    write_csv(RESULTS / 'momentum_traces.csv', rows)
    write_csv(RESULTS / 'momentum_trace_returns.csv', events)
    np.savez_compressed(RESULTS / 'momentum_curves.npz', **{f'{k}__{nm}': v for k, cur in curves.items() if k in ('s7.2_N1800_d4', 's7.2_N900_d2', 'archive_nu0.00000_N1800', 'central_K51', 'central_K21')
                                                           for nm, v in zip(('tau', 'Q_stored', 'Q_low', 'L2'), cur)})
    # 3. All-node fields at the returns: Radau (Fase9A exports) and AVF (base archive).
    frows = []
    prof = S.rows(ZA, 'minima/profiles.csv')
    groups = {}
    for r in prof:
        key = (int(r['j']), int(r['N']), float(r['nu']), int(r['divisor']))
        groups.setdefault(key, []).append((int(r['index']), float(r['u']), float(r['tau'])))
    nstar = 7.2; Q0 = 1 / (1 - 1 / nstar ** 2)
    for (j, N, nu, div), lst in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][3], kv[0][0])):
        lst.sort(); u = np.array([v for _, v, _ in lst]); assert len(u) == N
        Q = momentum_from_field(u, nstar)
        frows.append(dict(method='Pade-Radau', source='Fase9A minima/profiles.csv', N=N, nu=nu, divisor=div, j=j, tau=lst[0][2],
                          Q=Q, Q_minus_Q0=Q - Q0, rel=(Q - Q0) / Q0))
    for N in (450, 900, 1800):
        for K in (3200, 6400, 12800):
            mem = f'controls/long_results/p2.25_M4_N{N}_K{K}.npz'
            if mem not in S.names(ZBASE):
                continue
            x = S.npz(ZBASE, mem); meta = S.json(ZBASE, mem.replace('.npz', '.json'))
            for j in (1, 2, 3, 4):
                if f'u_at_return_{j}' in x:
                    Q = momentum_from_field(x[f'u_at_return_{j}'], nstar)
                    info = meta['returns'].get(str(j), {})
                    frows.append(dict(method='Pade-AVF', source=mem, N=N, nu=0.0, divisor=K // 3200, K=K, j=j, tau=info.get('tau', ''),
                                      Q=Q, Q_minus_Q0=Q - Q0, rel=(Q - Q0) / Q0, D=info.get('D', ''),
                                      relative_invariant_change=info.get('relative_invariant_change', '')))
    write_csv(RESULTS / 'momentum_field_returns.csv', frows)
    # console summary
    print('Q(0) at nstar=7.2:', Q0)
    for r in rows:
        if r['nu'] == 0 and (r['case'].startswith('s7.') or r['case'].startswith('s6.') or r['case'].startswith('archive') or r['case'] in ('central_K21', 'central_K35', 'central_K51', 'L6.8_K35', 'L7.4_K35')):
            print(f"{r['case']:28s} {r['method']:16s} N={r['N']:5d}  max|Q-Q0|/Q0: stored {r['max_rel_dev_stored']:.2e}, mid {r['max_rel_dev_mid']:.2e} (bracket +-{r['max_bracket_halfwidth']:.1e})  L2 power {r['L2_power_min']:.4f}..{r['L2_power_max']:.4f}")
    for r in frows:
        if r['nu'] == 0:
            print(f"{r['method']:11s} N={r['N']:5d} div={r['divisor']} j={r['j']}  rel(Q-Q0)={r['rel']:+.3e}  " + (f"I rel {r['relative_invariant_change']:+.1e}" if r.get('relative_invariant_change', '') != '' else ''))


if __name__ == '__main__':
    main()
