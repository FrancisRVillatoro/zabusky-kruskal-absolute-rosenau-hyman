"""Figures of the article, one PDF file per panel.

Inputs: the immutable archives of the repository (read directly from the ZIP
files) and the tables written by the other scripts of this directory into
``results/``.  Nothing is integrated here.

    python3 make_figures.py --root <repository root> --out <directory>
"""
import sys, json, argparse, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10.5, 'axes.labelsize': 10.5, 'legend.fontsize': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'lines.linewidth': 1.3,
                     'figure.dpi': 160, 'savefig.bbox': 'tight', 'mathtext.fontset': 'dejavuserif'})
COL = ['#0072B2', '#D55E00', '#009E73', '#7A5195', '#555555', '#C4961A', '#000000']
MRK = ['o', 's', '^', 'D', 'v', 'P', 'X']
STY = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), '-']
SIZE = (3.2, 2.65)
L72 = 2 * np.pi * 7.2


def new(size=SIZE):
    return plt.subplots(figsize=size, layout='constrained')


def lattice_core(p, offset, m=512):
    """Lattice sum of Proposition 1 restricted to the 17 nodes nearest the zero.

    Norm of the inverse mass stencil applied to the third-difference stencil of
    sgn(y)|y|^p on the integer lattice shifted by ``offset``.  The inverse decays
    as 0.43^|j|, so that a periodic lattice of 2m nodes with the stencil set to
    zero next to the wrap is equivalent to the infinite one.
    """
    LD = np.longdouble
    j = np.arange(-m, m, dtype=LD) - LD(offset)
    g = lambda z: np.sign(z) * np.abs(z) ** LD(p)
    d = np.asarray((g(j + 2) - g(j - 2) - 2 * (g(j + 1) - g(j - 1))) / 2, float)
    d[:4] = 0; d[-4:] = 0
    c = np.cos(2 * np.pi * np.arange(m + 1) / (2 * m))
    v = np.fft.irfft(np.fft.rfft(d) / ((64 + 52 * c + 4 * c * c) / 120), n=2 * m)
    return float(np.linalg.norm(v[m - 8:m + 9]))


class Figures:
    def __init__(self, root, out):
        self.S = Archives(root); self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.cat = pd.read_csv(RESULTS / 'return_catalog.csv')
        self.made = []

    def save(self, fig, name):
        fig.savefig(self.out / (name + '.pdf')); plt.close(fig); self.made.append(name)

    def frame(self, zipname, suffix):
        member = [n for n in self.S.names(zipname) if n.endswith(suffix)][0]
        return pd.read_csv(io.BytesIO(self.S.bytes(zipname, member)))

    def trace(self, zipname, suffix):
        return self.S.npz(zipname, [n for n in self.S.names(zipname) if n.endswith(suffix)][0])

    def returns(self, series, case):
        d = self.cat[(self.cat.series == series) & (self.cat.case == case)].sort_values('j')
        return d[d.j <= 4]

    @staticmethod
    def jaxis(ax):
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('Return index $j$')

    # ------------------------------------------------------------------ recurrence
    def recurrence(self):
        sp = self.frame(ZBASE, 'plots/data/phase4/spacetime_N1800.csv')
        fig, ax = new((3.4, 2.65))
        piv = sp.pivot_table(index='tau', columns='x', values='u').sort_index()
        im = ax.pcolormesh(piv.columns.to_numpy() / L72, piv.index.to_numpy(), piv.to_numpy(), shading='auto', cmap='RdBu_r', rasterized=True)
        fig.colorbar(im, ax=ax, label='$u$', shrink=.9); ax.set_xlabel('$x/L$'); ax.set_ylabel(r'$\tau$')
        self.save(fig, 'recurrence_1')
        tr = self.trace(ZB, 'plot_traces/s7.2_N1800_d4.npz')
        ev = self.returns('B', 's7.2_N1800_d4')
        fig, ax = new()
        ax.plot(tr['tau'], tr['D'], color=COL[0], label='$D$'); ax.plot(tr['tau'], tr['r'], color=COL[1], ls='--', lw=1, label='$r$')
        ax.plot(ev.tau, ev.D, 'o', mfc='none', mec='k', ms=6, label='returns')
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel('$D$ and $r$'); ax.set_ylim(0, 1.2); ax.legend(frameon=False, ncol=3, loc='upper center')
        self.save(fig, 'recurrence_2')

    # ------------------------------------------------------------------ returns across discretizations and lengths
    def return_sequences(self):
        fig, ax = new()
        sets = [('B', 's7.2_N1800_d4', 'Padé–Radau, $N=1800$'), ('B', 's7.2_N900_d2', 'Padé–Radau, $N=900$'), ('AVF', 'AVF_N1800_K12800', 'Padé–AVF, $N=1800$'),
                ('G', 'central_K21', 'Galerkin, 21'), ('G', 'central_K35', 'Galerkin, 35'), ('G', 'central_K51', 'Galerkin, 51')]
        for i, (s, c, lab) in enumerate(sets):
            d = self.returns(s, c); ax.semilogy(d.j, d.D, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, mfc='none' if i > 2 else None, label=lab)
        self.jaxis(ax); ax.set_ylabel('$D_j$'); ax.legend(frameon=False, fontsize=7.5)
        self.save(fig, 'returns_1')
        fig, ax = new()
        for i, ell in enumerate(['6.8', '7.0', '7.2', '7.4']):
            d = self.returns('B', f's{ell}_N1800_d4'); ax.semilogy(d.j, d.D, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=fr'$n_*={float(ell):g}$')
        self.jaxis(ax); ax.set_ylabel('$D_j$'); ax.legend(frameon=False, loc='upper left')
        self.save(fig, 'returns_2')
        sc = pd.read_csv(RESULTS / 'return_time_scaling.csv')
        pade = sc[sc.case.isin(['short672', 's6.8_N1800_d4', 's7.0_N1800_d4', 's7.2_N1800_d4', 's7.4_N1800_d4', 'long1120'])]
        fig, ax = new()
        ax.plot(pade.nstar, pade.tau1, 'o', color=COL[0], ms=5, label='Padé–Radau')
        lr = RESULTS / 'length_returns.csv'
        if lr.exists():
            g = pd.read_csv(lr); g = g[g.D < 0.5].groupby('nstar', as_index=False).first()      # first deep minimum of each length
            ax.plot(g.nstar, g.tau, 's', mfc='none', color=COL[1], ms=6, label='Galerkin')
        x = np.linspace(3, 11.5, 10); ax.plot(x, 0.140 * x, color='.5', lw=.9, ls='--', label=r'$0.140\,n_*$')
        ax.set_xlabel('$n_*$'); ax.set_ylabel(r'First return time $\tau_1$'); ax.legend(frameon=False)
        self.save(fig, 'returns_3')
        lc = RESULTS / 'length_curves.npz'
        if lc.exists():
            z = np.load(lc); fig, ax = new()
            for i, k in enumerate(sorted(z.files, key=float)):
                t, D = z[k]; ax.plot(t / float(k), D, color=COL[i], ls=STY[i], lw=1.1, label=fr'$n_*={float(k):g}$')
            ax.set_xlabel(r'$\tau/n_*$'); ax.set_ylabel('$D$'); ax.set_ylim(0, 1.85); ax.legend(frameon=False, ncol=2, loc='upper center')
            self.save(fig, 'returns_4')

    # ------------------------------------------------------------------ modal exchange and momentum
    def modal_exchange(self):
        tr = self.trace(ZB, 'plot_traces/s7.2_N1800_d4.npz'); tau = tr['tau']; e = 4 * np.abs(tr['c']) ** 2; n = tr['n']
        sel = tau <= 1.06
        fig, ax = new()
        for i in range(4):
            ax.plot(tau[sel], e[sel, i], color=COL[i], ls=STY[i], label=f'$e_{{{n[i]}}}$')
        ax.plot(tau[sel], e[sel, 4:].sum(axis=1), color=COL[4], ls=STY[4], lw=1, label=r'$\sum_{n\geq 9}e_n$')
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel('Modal power'); ax.legend(frameon=False, ncol=2, fontsize=8)
        self.save(fig, 'modal_exchange_1')
        w = momentum_weights(n, 7.2)
        fig, ax = new()
        for i in range(4):
            ax.plot(tau[sel], w[i] * e[sel, i], color=COL[i], ls=STY[i], label=f'$n={n[i]}$')
        ax.plot(tau[sel], (w[4:] * e[sel, 4:]).sum(axis=1), color=COL[4], ls=STY[4], lw=1, label=r'$n\geq 9$')
        ax.plot(tau[sel], (w * e[sel]).sum(axis=1), color='k', lw=.8, label='$Q$')
        ax.axhline(0, color='.8', lw=.6); ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$e_n/(1-k_n^2)$'); ax.set_ylim(-0.12, 1.55); ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper center')
        self.save(fig, 'modal_exchange_4')
        Q = momentum_from_modes(tr['c'], n, 7.2); P2 = e.sum(axis=1)
        fig, ax = new()
        ax.plot(tau, P2, color=COL[1], ls='--', label=r'$\sum_n e_n$'); ax.plot(tau, Q / Q[0], color=COL[0], label='$Q/Q(0)$')
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel('Quadratic functionals'); ax.set_ylim(0.6, 1.1); ax.legend(frameon=False, loc='lower right')
        self.save(fig, 'modal_exchange_2')
        fig, ax = new()
        runs = [(ZB, 'archive/traces/archive_nu0.00000_N450.npz', 'Padé, $N=450$', 7.2), (ZB, 'plot_traces/s7.2_N900_d2.npz', 'Padé, $N=900$', 7.2),
                (ZB, 'plot_traces/s7.2_N1800_d4.npz', 'Padé, $N=1800$', 7.2), (ZBL, 'plot_traces/central_K51.npz', 'Galerkin, 51', 7.2)]
        for i, (zn, suf, lab, ns) in enumerate(runs):
            t = self.trace(zn, suf); q = momentum_from_modes(t['c'], t['n'], ns)
            ax.semilogy(t['tau'][1:], np.maximum(np.abs(q[1:] / q[0] - 1), 1e-12), color=COL[i], ls=STY[i], lw=1, label=lab)
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel('$|Q/Q(0)-1|$'); ax.set_ylim(1e-15, 1e-1); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='lower center')
        self.save(fig, 'modal_exchange_3')

    # ------------------------------------------------------------------ near-null law
    def null_law(self):
        near = self.frame(ZBL, 'summary/near_null.csv')
        fig, ax = new()
        ax.plot(near.nstar, near.mean_e7_over_m7_squared, 'o-', color=COL[0], ms=4, label='Galerkin, cutoff 21')
        ax.axhline(near.predicted_mean.iloc[0], color=COL[1], ls='--', label='Coefficient at the null')
        ax.set_xlabel('$n_*$'); ax.set_ylabel(r'$\langle e_7\rangle/m_7^2$'); ax.legend(frameon=False)
        ax.ticklabel_format(axis='x', useOffset=False); ax.set_xticks([6.99, 7.0, 7.01])
        self.save(fig, 'null_law_1')
        rows = []
        for name in self.S.names(ZBL):
            if 'plot_traces/' not in name or not name.endswith('.npz'):
                continue
            stem = Path(name).stem
            if not (stem.startswith('L') or stem.startswith('near') or stem in ('central_K21', 'central_K35')):
                continue
            t = self.S.npz(ZBL, name); nstar = 7.2 if stem.startswith('central') else float(stem.split('_')[0 if stem.startswith('L') else 1].lstrip('L'))
            K = int(stem.split('_K')[1]); m7 = multipliers(7, nstar)
            if abs(m7) < 1e-9 or K not in (21, 35):
                continue
            e7 = 4 * np.abs(t['c'][:, list(t['n']).index(7)]) ** 2
            rows.append((K, nstar, np.trapezoid(e7, t['tau']) / (t['tau'][-1] - t['tau'][0]) / m7 ** 2))
        for ell in ('6.8', '7.0', '7.2', '7.4'):
            t = self.trace(ZB, f'plot_traces/s{ell}_N1800_d4.npz'); m7 = pade_multiplier(7, float(ell), 1800)
            e7 = 4 * np.abs(t['c'][:, list(t['n']).index(7)]) ** 2
            rows.append((0, float(ell), np.trapezoid(e7, t['tau']) / (t['tau'][-1] - t['tau'][0]) / m7 ** 2))
        d = pd.DataFrame(rows, columns=['K', 'nstar', 'ratio']).sort_values('nstar'); d.to_csv(RESULTS / 'mode7_ratio_sweep.csv', index=False)
        fig, ax = new()
        for i, (K, lab) in enumerate([(21, 'Galerkin, cutoff 21'), (35, 'Galerkin, cutoff 35'), (0, 'Padé, $N=1800$')]):
            q = d[d.K == K]; ax.plot(q.nstar, q.ratio, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, mfc='none' if i else None, label=lab)
        ax.set_xlabel('$n_*$'); ax.set_ylabel(r'$\langle e_7\rangle/m_7^2$'); ax.legend(frameon=False)
        self.save(fig, 'null_law_2')

    # ------------------------------------------------------------------ return map of the fifth harmonic
    def return_map(self):
        d = self.returns('B', 's7.2_N1800_d4'); fig, ax = new()
        ax.semilogy(d.j, d.D ** 2, 'k-', marker='o', ms=4, label='$D_j^2$')
        ax.semilogy(d.j, d.e5, color=COL[1], ls='--', marker='s', ms=4, label='$e_5$')
        ax.semilogy(d.j, d.H_minus_e5, color=COL[2], ls='-.', marker='^', ms=4, label='$H-e_5$')
        ax.semilogy(d.j, (d.r - 1) ** 2, color=COL[3], ls=':', marker='D', ms=4, label='$(r-1)^2$')
        self.jaxis(ax); ax.set_ylabel('Contributions to $D_j^2$'); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'return_map_4')
        fig, ax = new()
        sets = [('B', 's7.2_N1800_d4', 'Padé–Radau, 1800'), ('B', 's7.2_N900_d2', 'Padé–Radau, 900'), ('A', 'archive_nu0.00000_N450', 'Padé–Radau, 450'),
                ('AVF', 'AVF_N1800_K12800', 'Padé–AVF, 1800'), ('G', 'central_K51', 'Galerkin, 51'), ('G', 'central_K21', 'Galerkin, 21')]
        for i, (s, c, lab) in enumerate(sets):
            d = self.returns(s, c); th = np.radians(d.theta5_deg)
            ax.plot(d.amp5 * np.cos(th), d.amp5 * np.sin(th), marker=MRK[i], ls='none', color=COL[i], ms=5, mfc='none', label=lab)
        ang = np.radians(-71.0); ax.plot([0, 0.26 * np.cos(ang)], [0, 0.26 * np.sin(ang)], color='.5', lw=.9, ls='--', label=r'$\arg=-71^\circ$')
        ax.axhline(0, color='.8', lw=.6); ax.axvline(0, color='.8', lw=.6); ax.set_xlim(-0.01, 0.2); ax.set_ylim(-0.255, 0.012)
        ax.set_xlabel(r'Re $\sigma_5$'); ax.set_ylabel(r'Im $\sigma_5$'); ax.legend(frameon=False, fontsize=7, loc='upper right')
        self.save(fig, 'return_map_1')
        fig, ax = new()
        for i, ell in enumerate(['6.8', '7.0', '7.2', '7.4']):
            d = self.returns('B', f's{ell}_N1800_d4'); ax.semilogy(d.j, d.amp5, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=fr'$n_*={float(ell):g}$')
        j = np.array([1, 4]); ax.semilogy(j, 0.004 * 2.71 ** (j - 1), color='.5', lw=.9, ls='--', label=r'$\propto 2.71^{\,j}$')
        self.jaxis(ax); ax.set_ylabel(r'$|\sigma_5|$ at the returns'); ax.legend(frameon=False, ncol=2, fontsize=8)
        self.save(fig, 'return_map_2')
        fig, ax = new()
        for i, nu in enumerate(['0.00000', '0.00025', '0.00100', '0.00400']):
            d = self.returns('A', f'archive_nu{nu}_N1800'); ax.semilogy(d.j, d.amp5, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=fr'$\nu={float(nu):g}$')
        self.jaxis(ax); ax.set_ylabel(r'$|\sigma_5|$ at the returns'); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'return_map_3')

    # ------------------------------------------------------------------ relative periodic orbits of the truncations
    def rpo(self):
        f = RESULTS / 'rpo_orbits.json'
        if not f.exists():
            return
        res = [r for r in json.loads(f.read_text()) if r.get('converged', True)]
        X = np.linspace(0, 2 * np.pi, 1441)
        fig, ax = new()
        for i, r in enumerate([q for q in res if q['cutoff'] in (21, 25, 29)]):
            al = np.array(r['alpha']); n = 2 * np.arange(len(al)) + 1
            u = (al[None, :] * np.cos(np.outer(X, n))).sum(axis=1)
            ax.plot(X / (2 * np.pi), u - np.cos(X), color=COL[i], ls=STY[i], lw=1.1, label=f"cutoff {r['cutoff']}")
        ax.plot(X / (2 * np.pi), 0.1 * np.cos(X), color='.7', lw=.8, label=r'$0.1\cos(k_1x)$')
        ax.set_xlabel('$x/L$'); ax.set_ylabel('Orbit point minus cosine'); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper center'); ax.set_ylim(-0.13, 0.24)
        self.save(fig, 'rpo_1')
        fig, ax = new()
        for i, r in enumerate([q for q in res if q['cutoff'] in (21, 25, 29)]):
            n = np.array(sorted(int(k) for k in r['unstable_vector_power_share']))
            amp = np.sqrt([r['unstable_vector_power_share'][str(k)] for k in n]); ph = np.radians([r['unstable_vector_phase_deg'][str(k)] for k in n])
            v = (amp[None, :] * np.cos(np.outer(X, n) + ph[None, :])).sum(axis=1)
            ax.plot(X / (2 * np.pi), v / np.abs(v).max(), color=COL[i], ls=STY[i], lw=1.1, label=f"cutoff {r['cutoff']}")
        ax.plot(X / (2 * np.pi), np.cos(X), color='.7', lw=.8, label=r'$\cos(k_1x)$')
        ax.set_xlabel('$x/L$'); ax.set_ylabel('Unstable eigenfunction'); ax.set_ylim(-1.15, 1.75); ax.legend(frameon=False, fontsize=7.5, ncol=2, loc='upper center')
        self.save(fig, 'rpo_4')
        fig, ax = new()
        th = np.linspace(0, 2 * np.pi, 361); ax.plot(np.cos(th), np.sin(th), color='.7', lw=.8)
        for i, r in enumerate(res):
            ev = np.array([complex(a, b) for a, b in r['multipliers']])
            ax.plot(ev.real, ev.imag, marker=MRK[i], ls='none', color=COL[i], ms=4.5, mfc='none', label=f"cutoff {r['cutoff']}")
        ax.set_aspect('equal'); ax.set_xlim(-1.25, 3.0); ax.set_ylim(-1.75, 1.75); ax.set_xlabel(r'Re $\mu$'); ax.set_ylabel(r'Im $\mu$'); ax.legend(frameon=False, fontsize=7, loc='upper right', ncol=1, handletextpad=0.2, borderaxespad=0.1)
        self.save(fig, 'rpo_2')
        lc = RESULTS / 'rpo_linear_check.csv'
        if lc.exists():
            d = pd.read_csv(lc); fig, ax = new()
            for i, K in enumerate([k for k in sorted(d.cutoff.unique()) if k in (21, 25, 29)]):
                q = d[d.cutoff == K]
                ax.semilogy(q.j, q.D_nonlinear, marker=MRK[i], ls='-', color=COL[i], ms=4, label=f'{K}, returns')
                ax.semilogy(q.j, q.D_linear, marker=MRK[i], ls='--', color=COL[i], ms=4, mfc='none', label=f'{K}, linear')
            self.jaxis(ax); ax.set_ylabel('$D_j$'); ax.legend(frameon=False, fontsize=6.3, loc='upper left', handlelength=1.6)
            self.save(fig, 'rpo_3')

    # ------------------------------------------------------------------ initial velocity (Proposition 1)
    def initial_velocity(self):
        s = self.frame(ZAL, 'results_local/initial_scaling.csv'); ss = self.frame(ZAL, 'results_local/scaling_summary.csv')
        fig, ax = new()
        for i, p in enumerate([1.75, 2.25, 2.5, 3.]):
            d = s[(s.family == 'Z') & (s.p == p) & (s.alpha == 0)].sort_values('N')
            ax.loglog(d.N, d.velocity_L2_extended, marker=MRK[i], ms=3.5, ls=STY[i], color=COL[i], label=f'$p={p:g}$')
        ax.set_xlabel('$N$'); ax.set_ylabel(r'$\|F_h\|_{2,h}$'); ax.legend(frameon=False)
        self.save(fig, 'initial_velocity_1')
        fig, ax = new()
        ax.plot(ss.p, ss.theoretical_core_exponent, 'k-', label=r'$5/2-p$'); ax.scatter(ss.p, (ss.core_min + ss.core_max) / 2, facecolors='none', edgecolors=COL[0], s=35, label='17-node cores')
        ax.fill_between(ss.p.to_numpy(), ss.global_min.to_numpy(), ss.global_max.to_numpy(), color=COL[1], alpha=.2)
        ax.plot(ss.p, (ss.global_min + ss.global_max) / 2, marker='s', ms=3.5, color=COL[1], label='whole grid'); ax.axhline(0, color='.65', lw=.6)
        ax.set_xlabel('$p$'); ax.set_ylabel('Last-doubling exponent'); ax.legend(frameon=False)
        self.save(fig, 'initial_velocity_2')
        fig, ax = new()
        for i, p in enumerate([1.75, 2.0, 2.25, 2.375]):
            d = s[(s.family == 'Z') & (s.p == p) & (s.N == s.N.max())].sort_values('alpha')
            ref = float(d.lattice_core_prediction.iloc[0])
            off = np.linspace(0, 1, 81); lat = np.array([lattice_core(p, a) for a in off])
            chk = np.max(np.abs(np.interp(d.alpha, off, lat / lat[0]) - d.lattice_core_prediction / ref))
            assert chk < 1e-5, f'lattice sum differs from the archived prediction by {chk:.1e}'
            ax.plot(off, lat / lat[0], color=COL[i], ls=STY[i], lw=1)
            ax.plot(np.r_[d.alpha, 1.0], np.r_[d.scaled_core, d.scaled_core.iloc[0]] / ref, ls='none', marker=MRK[i], ms=4.5, mfc='none', color=COL[i], label=f'$p={p:g}$')
        ax.set_xlabel('Offset of the zero in its cell'); ax.set_ylabel('Coefficient relative to offset 0'); ax.set_xticks([0, .25, .5, .75, 1]); ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center'); ax.set_ylim(0.99, 1.24)
        self.save(fig, 'initial_velocity_4')
        fig, ax = new()
        d = s[(s.family == 'Z') & (s.p == 2.5) & (s.alpha == 0)].dropna(subset=['increment_squared_per_log2'])
        ax.semilogx(d.N, 1e4 * d.increment_squared_per_log2, marker='o', color=COL[0], ms=3.5, label='Padé operator')
        ax.axhline(1e4 * d.critical_log_slope.iloc[-1], color=COL[1], ls='--', label=r'$2\sum_z c_z^2$')
        ax.set_xlabel('$N$'); ax.set_ylabel(r'$10^4\,\Delta\|F_h\|_{2,h}^2/\log 2$'); ax.legend(frameon=False, loc='best')
        self.save(fig, 'initial_velocity_3')

    # ------------------------------------------------------------------ initial layer
    def initial_layer(self):
        f = RESULTS / 'initial_layer.csv'
        if not f.exists():
            return
        d = pd.read_csv(f); fig, ax = new()
        for i, p in enumerate([1.75, 2.0, 2.25, 3.0]):
            q = d[(d.p == p) & (d.N == 8192)]; ax.loglog(q.t, q.sup_window, color=COL[i], ls=STY[i], label=f'$p={p:g}$')
            q4 = d[(d.p == p) & (d.N == 4096)].iloc[::18]; ax.loglog(q4.t, q4.sup_window, ls='none', marker=MRK[i], ms=2.6, mfc='none', color=COL[i])
            t = np.array([2e-2, 1.0]); ref = float(np.interp(0.1, q.t, q.sup_window))
            ax.loglog(t, 1.9 * ref * (t / 0.1) ** (1 / (4 - p)), color=COL[i], lw=.7)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\max_{|y|<1}|U(t)-U(0)|$'); ax.set_xlim(1e-7, 1.5); ax.set_ylim(1e-8, 0.5); ax.legend(frameon=False, loc='lower right')
        self.save(fig, 'initial_layer_1')
        pr = pd.read_csv(RESULTS / 'initial_layer_profiles.csv'); fig, ax = new()
        for i, t in enumerate([0.01, 0.03, 0.1, 0.3]):
            q = pr[(pr.p == 2.25) & (pr.N == 8192) & (np.isclose(pr.t, t))].sort_values('xi'); q = q[np.abs(q.xi) <= 8]
            ax.plot(q.xi, q.scaled, color=COL[i], ls=STY[i], lw=1.1, label=f'$t={t:g}$')
        ax.set_xlabel(r'$y/y_*(t)$'); ax.set_ylabel(r'$[U(t)-U(0)]/(\kappa\,y_*)$'); ax.legend(frameon=False, loc='lower left', fontsize=8)
        self.save(fig, 'initial_layer_2')

    # ------------------------------------------------------------------ return residual and zero structure
    def localization(self):
        prof = self.frame(ZA, 'minima/profiles.csv'); z = self.frame(ZAC, 'finest_returns.csv'); inv = z[(z.nu == 0) & (z.selection == 'minima')].sort_values('j')
        for k, j in ((1, 1), (2, 4)):
            d = prof[(prof.N == 1800) & (prof.nu == 0) & (prof.j == j)]; d = d[d.divisor == d.divisor.max()].sort_values('x')
            fig, ax = new()
            ax.plot(d.x / L72, d.aligned_initial, 'k--', lw=1, label='Translated cosine'); ax.plot(d.x / L72, d.u, color=COL[0], label='Computed field')
            ax.plot(d.x / L72, d.residual, color=COL[1], lw=1, label='Residual')
            ax.set_xlabel('$x/L$'); ax.set_ylabel('$u$ and residual'); ax.set_ylim(-1.15, 1.75); ax.legend(frameon=False, loc='upper center', fontsize=7.5, ncol=2)
            self.save(fig, f'return_localization_{k}')
        fig, ax = new()
        ax.semilogy(inv.j, inv.residual_amp, marker='o', color=COL[0], label=r'Inside $|u|\leq0.2$'); ax.semilogy(inv.j, inv.residual_outside_amp, marker='s', color=COL[1], ls='--', label='Outside')
        ax.semilogy(inv.j, inv.residual_rad, marker='^', color=COL[2], ls=':', label='Within 0.4 of a zero')
        self.jaxis(ax); ax.set_ylabel('Normalized squared residual'); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'return_localization_3')
        fig, ax = new()
        ax.semilogy(inv.j, 100 * inv.fraction_amp, marker='o', color=COL[0], label=r'Inside $|u|\leq0.2$'); ax.semilogy(inv.j, 100 * inv.fraction_rad, marker='^', color=COL[2], ls=':', label='Within 0.4 of a zero')
        self.jaxis(ax); ax.set_ylabel('Share of the squared residual (%)'); ax.set_ylim(1, 150); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'return_localization_4')
        f = RESULTS / 'zero_local_profiles.csv'
        if not f.exists():
            return
        d = pd.read_csv(f); d = d[d.j == 1]
        beta = abs(float(d[d.N == 1800].slope_w_cross.iloc[0])); sgn = np.sign(float(np.polyfit(d[d.N == 1800].y, d[d.N == 1800].w, 1)[0]))
        yy = np.linspace(-0.3, 0.3, 801)
        fig, ax = new()
        ax.plot(yy, sgn * np.sign(yy) * np.abs(beta * yy) ** (1 / P), color='.55', lw=1, label=r'$\mathrm{sgn}(y)|\beta y|^{1/p}$')
        for i, N in enumerate([450, 900, 1800]):
            q = d[(d.N == N) & (np.abs(d.y) <= 0.3)].sort_values('y'); ax.plot(q.y, q.u, marker=MRK[i], ms=3.2, ls='none', mfc='none', color=COL[i], label=f'$N={N}$')
        ax.set_xlabel('$y$'); ax.set_ylabel('$u$'); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'zero_structure_1')
        fig, ax = new()
        ax.plot(yy, sgn * beta * yy, color='.55', lw=1, label=r'$\beta y$')
        for i, N in enumerate([450, 900, 1800]):
            q = d[(d.N == N) & (np.abs(d.y) <= 0.3)].sort_values('y'); ax.plot(q.y, q.w, marker=MRK[i], ms=3.2, ls='none', mfc='none', color=COL[i], label=f'$N={N}$')
        ax.set_xlabel('$y$'); ax.set_ylabel(r'$w=f_p(u)$'); ax.legend(frameon=False, fontsize=8)
        self.save(fig, 'zero_structure_2')

    # ------------------------------------------------------------------ sensitivity
    def transverse(self):
        f = self.frame(ZCC, 'fixed_time_gains.csv'); r = self.frame(ZCC, 'return_gains.csv')
        d = f[(f.p == 2.25) & (f.M == 4) & (f.K == 12800) & (f.nu == 0) & (f.method == 'radau') & (f.strict == 0) & f.tau.isin([1, 2, 3, 4])]
        fig, ax = new()
        for i, N in enumerate([450, 900, 1800, 3600]):
            q = d[d.N == N].sort_values('tau'); ax.plot(np.r_[0, q.tau], np.r_[0, q.log10_G_cos2], marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=f'$N={N}$')
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel(r'$\log_{10}G_2$'); ax.legend(frameon=False)
        self.save(fig, 'transverse_1')
        fig, ax = new()
        for i, (method, K) in enumerate([('radau', 12800), ('radau', 25600), ('avf', 12800), ('avf', 25600)]):
            q = r[r.case == f'p2.25_M4_N1800_K{K}_nu0_{method}_tangent'].sort_values('j')
            ax.plot(q.j, q.log10_G_cos2, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=f'{"Radau" if method == "radau" else "AVF"}, $N_t={K}$')
        self.jaxis(ax); ax.set_ylabel(r'$\log_{10}G_2$ at the returns'); ax.legend(frameon=False, fontsize=7.5)
        self.save(fig, 'transverse_2')
        gt = self.frame(ZC, 'report/gain_traces.csv'); base = 'p2.25_M4_N1800_K12800_nu0_radau_'
        fig, ax = new()
        tn = gt[gt.case == base + 'tangent'].sort_values('tau'); lin = 1e-8 * np.sqrt(0.5) * 10.0 ** tn.log10_G_cos2.to_numpy()
        ax.semilogy(tn.tau[lin < 1], lin[lin < 1], color='k', lw=.7, label='tangent map')
        for i, (eps, lab) in enumerate([('+1e-06', r'$\epsilon=10^{-6}$'), ('+1e-07', r'$\epsilon=10^{-7}$'), ('+1e-08', r'$\epsilon=10^{-8}$'), ('+0e+00', r'$\epsilon=0$')]):
            q = gt[gt.case == base + 'nonlinear_eps' + eps].sort_values('tau'); q = q[q.even_rms > 0]
            ax.semilogy(q.tau, q.even_rms, color=COL[i], ls=STY[i], lw=1.1, label=lab)
        ax.set_xlabel(r'$\tau$'); ax.set_ylabel('Even component (r.m.s.)'); ax.set_ylim(1e-15, 3); ax.legend(frameon=False, fontsize=7.5, loc='lower right')
        self.save(fig, 'transverse_4')
        g = RESULTS / 'gain_ladder_traces.npz'
        if g.exists():
            z = np.load(g); fig, ax = new()
            for i, K in enumerate(sorted({int(k.split('_')[0]) for k in z.files})):
                t = z[f'{K}_tau']; ax.semilogy(t, z[f'{K}_G_even'], color=COL[i], lw=1.1, label=f'cutoff {K}'); ax.semilogy(t, z[f'{K}_G_odd'], color=COL[i], lw=.9, ls='--')
            ax.set_xlabel(r'$\tau$'); ax.set_ylabel('Gain'); ax.legend(frameon=False, fontsize=8, loc='upper left')
            self.save(fig, 'transverse_3')

    # ------------------------------------------------------------------ regularization and positive data
    def regularization(self):
        z = self.frame(ZAC, 'finest_returns.csv')
        fa, aa = new(); fb, ab = new()
        for i, nu in enumerate([0., .00025, .001, .004]):
            d = z[(z.nu == nu) & (z.selection == 'minima')].sort_values('j')
            aa.semilogy(d.j, d.D, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=fr'$\nu={nu:g}$')
            ab.plot(d.j, 100 * d.fraction_amp, marker=MRK[i], ls=STY[i], color=COL[i], ms=4, label=fr'$\nu={nu:g}$')
        for ax, yl in ((aa, '$D_j$'), (ab, r'Residual within $|u|\leq0.2$ (%)')):
            self.jaxis(ax); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=8)
        self.save(fa, 'regularization_positive_1'); self.save(fb, 'regularization_positive_2')
        rg = self.frame(ZCC, 'regularized_final_gains.csv'); fig, ax = new()
        for i, N in enumerate([900, 1800]):
            q = rg[rg.N == N].sort_values('nu'); ax.semilogx(q.nu, q.G_cos2, marker=MRK[i], ls=STY[i], color=COL[i], ms=5, label=f'$N={N}$')
        ax.set_xticks([2.5e-4, 1e-3, 4e-3]); ax.set_xticklabels(['0.00025', '0.001', '0.004']); ax.minorticks_off()
        ax.set_xlabel(r'$\nu$'); ax.set_ylabel(r'$G_2$ at $\tau=4.25$'); ax.set_ylim(0, 200); ax.legend(frameon=False, loc='lower left')
        self.save(fig, 'regularization_positive_4')
        fig, ax = new()
        for i, N in enumerate([1000, 2000]):
            d = self.frame(ZBASE, f'experiments/results/positive{N}/history.csv'); ax.plot(d.tau[::17], (d.min_u / d.max_u)[::17], ls=STY[i], color=COL[i], label=f'$N={N}$')
        ax.axhline(0, color='k', lw=.7); ax.axvline(2, color='.6', lw=.7, ls=':'); ax.set_xlabel(r'$\tau$'); ax.set_ylabel('Minimum / maximum'); ax.legend(frameon=False)
        self.save(fig, 'regularization_positive_3')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); ap.add_argument('--out', default=str(HERE.parent / 'figures')); ap.add_argument('--only', default='')
    a = ap.parse_args()
    F = Figures(a.root, a.out)
    todo = ['recurrence', 'return_sequences', 'modal_exchange', 'null_law', 'return_map', 'rpo', 'initial_velocity', 'initial_layer', 'localization', 'transverse', 'regularization']
    for name in ([s for s in a.only.split(',') if s] or todo):
        getattr(F, name)()
    print('written:', ', '.join(F.made))


if __name__ == '__main__':
    main()
