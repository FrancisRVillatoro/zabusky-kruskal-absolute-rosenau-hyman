"""Structure of the numerical solution at its zero crossings (return minima).

Read-only analysis of the all-node profiles exported from the production HDF5
files (Fase9A_HDF5 archive, minima/profiles.csv).  At every sign change of a
profile we examine the flux variable w = sgn(u)|u|^p cell by cell:
  * slope of w in the crossing cell and in the two cells on either side;
  * increment of u across the crossing cell and its secant slope;
  * position of the zero inside the cell (linear interpolation of w).
A transversal zero of w with slope beta means u ~ sgn(y)|beta y|^(1/p) (a cusp
with vertical tangent), for which the increment of u across a cell is O(h^(1/p)).
The total variation of each profile and its number of sign changes are recorded.
The nodes around the first crossing of selected profiles are written for plotting.
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *


def analyse(u, L, p=P):
    N = len(u); h = L / N
    w = np.sign(u) * np.abs(u) ** p
    tv = float(np.abs(np.diff(np.r_[u, u[0]])).sum())
    out = []
    for i in range(N):
        if u[i] * u[(i + 1) % N] >= 0:
            continue
        s = lambda k: float((w[(i + k + 1) % N] - w[(i + k) % N]) / h)       # slope of w in the cell i+k, i+k+1
        slopes = [s(k) for k in (-2, -1, 0, 1, 2)]
        theta = float(abs(w[i]) / (abs(w[i]) + abs(w[(i + 1) % N])))
        jump = float(abs(u[(i + 1) % N] - u[i]))
        neigh = float(np.mean([slopes[0], slopes[1], slopes[3], slopes[4]]))
        out.append(dict(cell=i, x_zero=(i + theta) * h, theta=theta, slope_w_left2=slopes[0], slope_w_left1=slopes[1], slope_w_cross=slopes[2],
                        slope_w_right1=slopes[3], slope_w_right2=slopes[4], cross_over_neighbours=slopes[2] / neigh if neigh else np.nan,
                        jump_u=jump, secant_u=jump / h))
    return tv, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root'); a = ap.parse_args()
    S = Archives(a.root)
    L = 2 * np.pi * 7.2
    groups = {}
    for r in S.rows(ZA, 'minima/profiles.csv'):
        key = (float(r['nu']), int(r['N']), int(r['divisor']), int(r['j']))
        groups.setdefault(key, []).append((int(r['index']), float(r['u']), float(r['tau']), float(r['aligned_initial'])))
    rows, summary, local = [], [], []
    for key in sorted(groups):
        nu, N, div, j = key
        lst = sorted(groups[key]); u = np.array([v[1] for v in lst]); tau = lst[0][2]; ref = np.array([v[3] for v in lst])
        tv, cr = analyse(u, L)
        h = L / N
        for c in cr:
            rows.append(dict(nu=nu, N=N, divisor=div, j=j, tau=tau, h=h, **c))
        # crossings of the aligned cosine: the nearest crossing of the profile to each
        zref = [(i + abs(ref[i]) / (abs(ref[i]) + abs(ref[(i + 1) % N]))) * h for i in range(N) if ref[i] * ref[(i + 1) % N] < 0]
        main_c = [min(cr, key=lambda c: abs((c['x_zero'] - z + L / 2) % L - L / 2)) for z in zref]
        summary.append(dict(nu=nu, N=N, divisor=div, j=j, tau=tau, h=h, sign_changes=len(cr), total_variation=tv,
                            slope_w_cross=float(np.mean([abs(c['slope_w_cross']) for c in main_c])),
                            slope_w_neighbours=float(np.mean([abs(c['slope_w_left1']) + abs(c['slope_w_right1']) for c in main_c]) / 2),
                            jump_u=float(np.mean([c['jump_u'] for c in main_c])), secant_u=float(np.mean([c['secant_u'] for c in main_c])),
                            theta=';'.join(f"{c['theta']:.3f}" for c in main_c)))
        if nu == 0.0 and div == 1 and j in (1, 3):
            c = main_c[0]; i0 = c['cell']; m = int(round(0.75 / h))
            for k in range(i0 - m, i0 + m + 2):
                y = (k % N) * h - c['x_zero']; y = (y + L / 2) % L - L / 2
                local.append(dict(N=N, j=j, tau=tau, y=y, u=u[k % N], w=np.sign(u[k % N]) * abs(u[k % N]) ** P, slope_w_cross=c['slope_w_cross']))
    write_csv(RESULTS / 'zero_crossings.csv', rows)
    write_csv(RESULTS / 'zero_summary.csv', summary)
    write_csv(RESULTS / 'zero_local_profiles.csv', local)
    print('nu=0, base step (divisor 1):')
    for s in summary:
        if s['nu'] == 0.0:
            print(f"  j={s['j']} N={s['N']:5d} div={s['divisor']}: crossings {s['sign_changes']}, TV {s['total_variation']:.3f}, w-slope cross {s['slope_w_cross']:.4f} neighbours {s['slope_w_neighbours']:.4f}, "
                  f"jump u {s['jump_u']:.4f}, secant {s['secant_u']:.3f}, theta {s['theta']}")
    print('nu>0, N=1800:')
    for s in summary:
        if s['nu'] > 0 and s['N'] == 1800:
            print(f"  nu={s['nu']} j={s['j']} div={s['divisor']}: crossings {s['sign_changes']}, TV {s['total_variation']:.3f}, w-slope cross {s['slope_w_cross']:.4f} neighbours {s['slope_w_neighbours']:.4f}, jump u {s['jump_u']:.4f}")


if __name__ == '__main__':
    main()
