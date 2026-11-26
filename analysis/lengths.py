"""Returns of the cosine for domain lengths outside the window 6.8 <= n_* <= 7.4.

Odd-harmonic Galerkin models (p = 9/4) with the cutoff chosen so that the largest
retained wavenumber is close to that of the cutoff-51 model at n_* = 7.2
(k_max = 51/7.2).  Chronological minima of D with prominence >= 0.03 after
tau = 0.2, as in the length sweep of the article.  The Pade first returns at
n_* = 5.4 and 9.0 (base experiments short672 and long1120) provide a check.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import RESULTS, write_csv, prominent_minima
from galerkin_ext import Galerkin, diagnostics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nstar', default='3.6,5.4,9.0,10.8')
    ap.add_argument('--kmax', type=float, default=51 / 7.2); ap.add_argument('--span', type=float, default=0.62)
    a = ap.parse_args()
    rows = []; curves = {}
    for nstar in [float(s) for s in a.nstar.split(',')]:
        cutoff = int(round(a.kmax * nstar)); cutoff += 1 - cutoff % 2
        g = Galerkin(nstar, cutoff, 1024)
        tau_end = a.span * nstar
        t = np.linspace(0, tau_end, int(round(tau_end / 0.0025)) + 1)
        sol = g.flow(g.cosine(), tau_end, rtol=1e-10, t_eval=t)
        c = sol.y.T; D, r, H = diagnostics(c)
        e = 4 * np.abs(c) ** 2
        I = np.array([g.invariants(ci) for ci in c[::20]])
        curves[f'{nstar:g}'] = np.vstack([t, D])
        ev = prominent_minima(t, D)
        for j, (i, prom) in enumerate(ev, 1):
            k = int(np.argmax(e[i, 1:])) + 1
            rows.append(dict(nstar=nstar, cutoff=cutoff, j=j, tau=t[i], D=D[i], prominence=prom, tau_over_nstar=t[i] / nstar, r=r[i], H=H[i],
                             dominant_harmonic=int(g.n_odd[k]), dominant_share=e[i, k] / H[i],
                             I_drift=float(np.max(np.abs(I[:, 0] / I[0, 0] - 1))), Q_drift=float(np.max(np.abs(I[:, 1] / I[0, 1] - 1))), nfev=sol.nfev))
        print(f"n*={nstar:g} K={cutoff}: " + '; '.join(f"tau={t[i]:.3f} D={D[i]:.3f}" for i, _ in ev) + f"   max D over run {D.max():.2f}, drifts I {rows[-1]['I_drift'] if ev else 0:.1e} Q {rows[-1]['Q_drift'] if ev else 0:.1e}", flush=True)
    write_csv(RESULTS / 'length_returns.csv', rows)
    np.savez_compressed(RESULTS / 'length_curves.npz', **curves)


if __name__ == '__main__':
    main()
