"""Tangent gains of the Galerkin truncations along the orbit of the cosine.

For each cutoff K (all harmonics 1..K are available to the perturbation; the base
orbit stays in the odd harmonics) the variational equations are integrated with
DOP853 for two initial directions:
  even : cos(2 k_1 x)   (breaks the half-period antisymmetry; "transverse")
  odd  : cos(3 k_1 x)   (inside the symmetric subspace)
Gains are ratios of RMS norms.  For the odd direction the neutral directions
(translation u_x and time shift u_tau) are projected out before taking the norm.
The spectrum of the evolved even perturbation is summarised by the share of its
power in harmonics above 20.
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import RESULTS, write_csv
from galerkin_ext import Galerkin, diagnostics


def remove_neutral(g, c, v):
    """Remove from v its least-squares components along u_x and u_tau at state c."""
    k = g.n_odd / g.nstar
    basis = np.array([1j * k * c, g.rhs(0.0, c)])
    A = np.column_stack([np.concatenate([b.real, b.imag]) for b in basis])
    y = np.concatenate([v.real, v.imag])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    return r[:len(v)] + 1j * r[len(v):]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nstar', type=float, default=7.2)
    ap.add_argument('--cutoffs', default='21,35,51'); ap.add_argument('--rtol', type=float, default=1e-10)
    ap.add_argument('--tag', default=''); a = ap.parse_args()
    rows, traces = [], {}
    for cutoff in [int(s) for s in a.cutoffs.split(',')]:
        quad = 512 if cutoff <= 27 else (1024 if cutoff <= 61 else 2048)
        g = Galerkin(a.nstar, cutoff, quad)
        se = np.zeros(len(g.n_even), complex); se[0] = 0.5          # cos(2 k_1 x)
        so = np.zeros(len(g.n_odd), complex); so[1] = 0.5           # cos(3 k_1 x)
        out = g.tangent_run(seeds_even=[se], seeds_odd=[so], tau_end=4.25, sample=400, rtol=a.rtol)
        tau, c = out['tau'], out['c']
        D, r, H = diagnostics(c)
        ve, vo = out['even'][:, 0, :], out['odd'][:, 0, :]
        Ge = np.sqrt((np.abs(ve) ** 2).sum(axis=1) / (np.abs(se) ** 2).sum())
        vperp = np.array([remove_neutral(g, c[i], vo[i]) for i in range(len(tau))])
        Go = np.sqrt((np.abs(vperp) ** 2).sum(axis=1) / (np.abs(vperp[0]) ** 2).sum())
        Go_raw = np.sqrt((np.abs(vo) ** 2).sum(axis=1) / (np.abs(so) ** 2).sum())
        high = (np.abs(ve[:, g.n_even > 20]) ** 2).sum(axis=1) / (np.abs(ve) ** 2).sum(axis=1)
        traces[str(cutoff)] = dict(tau=tau, D=D, G_even=Ge, G_odd=Go, G_odd_raw=Go_raw, high_share_even=high)
        for t in (1.0, 2.0, 3.0, 4.0):
            i = int(np.argmin(np.abs(tau - t)))
            rows.append(dict(nstar=a.nstar, cutoff=cutoff, quad=quad, tau=tau[i], D=D[i], gain_even=Ge[i], gain_odd=Go[i], gain_odd_raw=Go_raw[i],
                             even_power_share_above_20=high[i], nfev=out['nfev'], seconds=out['seconds']))
        print(f"cutoff {cutoff} quad {quad}: " + '; '.join(f"tau={r_['tau']:.0f}: even {r_['gain_even']:.3g}, odd {r_['gain_odd']:.3g} (raw {r_['gain_odd_raw']:.3g}), >20: {100*r_['even_power_share_above_20']:.1f}%" for r_ in rows[-4:]) + f"  [{out['seconds']:.0f} s]", flush=True)
    write_csv(RESULTS / f'gain_ladder{a.tag}.csv', rows)
    np.savez_compressed(RESULTS / f'gain_ladder_traces{a.tag}.npz', **{f'{k}_{q}': v for k, d in traces.items() for q, v in d.items()})


if __name__ == '__main__':
    main()
