"""Do the returns of the cosine follow from the linearization about the periodic orbit?

For a converged orbit (rpo.py) we take delta = cosine - orbit point, propagate it
with the monodromy matrix, and evaluate the distance of the result to the
translation orbit of the cosine after optimal alignment in translation and in
time (the latter to first order, using the flow direction).  The prediction is
compared with the nonlinear return of the cosine in the same truncation.
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *
from galerkin_ext import Galerkin, diagnostics
from rpo import floquet


def check(res, tau_end=4.25):
    g = Galerkin(res['nstar'], res['cutoff'], res['quad'])
    n = g.n_odd; no = len(n)
    alpha = np.array(res['alpha']); z = np.r_[alpha, res['tau_h']]
    M, closure, phi = floquet(g, z, *((1e-10, 1e-14) if res['cutoff'] <= 25 else (1e-9, 1e-12)))
    ev, vec = np.linalg.eig(M); order = np.argsort(-np.abs(ev)); ev = ev[order]; vec = vec[:, order]
    e1 = np.zeros(no); e1[0] = 1.0
    d = np.r_[(e1 - alpha) / 2, np.zeros(no)]                   # cosine minus orbit point, coordinates (Re a, Im a)
    coef = np.linalg.solve(vec, d.astype(complex))
    print(f"cutoff {res['cutoff']}: leading multiplier {abs(ev[0]):.4f}; component of (cosine - orbit) along the unstable vector: {abs(coef[0]) * np.linalg.norm(vec[:, 0]):.5f} of |delta| = {np.linalg.norm(d):.5f}")
    flow = g.rhs(0.0, alpha.astype(complex) / 2)                # flow direction at the orbit point (purely imaginary)
    # nonlinear returns of the cosine in the same model
    t = np.arange(round(tau_end * 3200) + 1) / 3200
    sol = g.flow(g.cosine(), tau_end, rtol=1e-9, t_eval=t)
    D, r, H = diagnostics(sol.y.T)
    rows = []
    y = d.copy()
    for j in range(1, 5):
        y = M @ y
        rho = (alpha / 2 + y[:no] + 1j * y[no:]) - e1 / 2       # state after j periods minus cosine (frame of the orbit)
        # align: translation of the cosine (direction -i e1) and time shift of the state (direction flow)
        A = np.zeros((2 * no, 2)); A[no, 0] = 1.0               # Im of first mode
        A[:no, 1] = flow.real; A[no:, 1] = flow.imag
        b = np.r_[rho.real, rho.imag]
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        rr = b - A @ c; rr = rr[:no] + 1j * rr[no:]
        Dlin = 2 * np.linalg.norm(rr)
        amp5 = 2 * abs(rr[2]); ph5 = np.degrees(np.angle(rr[2])); share5 = abs(rr[2]) ** 2 / (np.abs(rr[1:]) ** 2).sum()
        ids = np.flatnonzero((t >= j - .2) & (t <= j + .2)); i = ids[np.argmin(D[ids])]
        s = slice_coefficients(sol.y[:, i], n)
        rows.append(dict(cutoff=res['cutoff'], leading_multiplier=float(abs(ev[0])), unstable_component=float(abs(coef[0]) * np.linalg.norm(vec[:, 0])), delta_norm=float(np.linalg.norm(d)), j=j, D_linear=Dlin, D_nonlinear=float(D[i]), amp5_linear=amp5, amp5_nonlinear=float(abs(s[2])),
                         theta5_linear=ph5, theta5_nonlinear=float(np.degrees(np.angle(s[2]))), share5_linear=share5,
                         share5_nonlinear=float(abs(s[2]) ** 2 / H[i]), tau_nonlinear=float(t[i]), tau_linear=float(j * res['period'] + c[1])))
        print(f"  j={j}: D lin {Dlin:.4f} / nonlin {D[i]:.4f};  amp5 lin {amp5:.4f} / nonlin {abs(s[2]):.4f};  theta5 lin {ph5:+.1f} / nonlin {np.degrees(np.angle(s[2])):+.1f};  e5/H lin {share5:.3f} / nonlin {abs(s[2])**2/H[i]:.3f};  tau lin {j*res['period']+c[1]:.4f} / nonlin {t[i]:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--file', default=str(RESULTS / 'rpo_orbits.json'))
    ap.add_argument('--tau_end', type=float, default=4.25); a = ap.parse_args()
    rows = []
    for res in json.load(open(a.file)):
        if res.get('converged', True):
            rows += check(res, a.tau_end)
    write_csv(RESULTS / 'rpo_linear_check.csv', rows)


if __name__ == '__main__':
    main()
