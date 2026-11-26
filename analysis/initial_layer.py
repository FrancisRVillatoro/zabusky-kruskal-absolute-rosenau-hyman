"""Departure from the cosine at very short times: test of the similarity layer.

Semidiscrete Pade system  S dU/dt = -(B + C) f_p(U)  (stencils of the article),
U(0) = cos(k_1 x), integrated with the implicit midpoint rule on a geometric
time grid, t_{n+1} = t_n + max(dt0, rho t_n).  Independent small implementation
(sparse LU of the cyclic pentadiagonal Newton matrix); the production solver is
not used.

Near a simple zero x_0 with u_0 ~ a y, y = x - x_0, a = k_1, the balance
u_t ~ (f_p(u))_yyy gives a layer of width y_*(t) = (a^(p-1) t)^(1/(4-p)) and
amplitude a y_*.  Diagnostics, all in the window |y| < ell around the ascending
zero x_0 = 3L/4:
    m(t)  = max |U - U_0|          (predicted ~ t^(1/(4-p)) for t >> h^(4-p)/a^(p-1))
    e(t)  = ( h sum (U - U_0)^2 )^(1/2)   (predicted ~ t^(3/(8-2p)) for p < 5/2)
and rescaled profiles (U - U_0)/(a y_*) against y/y_* at selected times.
The global norm E(t) = ||U - U_0||_{2,h} is recorded as well.
"""
import sys, time, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from common import RESULTS, write_csv

SNAP = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def circulant(N, stencil, scale=1.0):
    """Cyclic banded matrix from a five-point stencil in the order j-2..j+2."""
    rows = np.repeat(np.arange(N), 5)
    cols = (np.arange(N)[:, None] + np.arange(-2, 3)[None, :]) % N
    data = np.tile(np.asarray(stencil, float) * scale, N)
    return sp.csc_matrix((data, (rows, cols.ravel())), shape=(N, N))


def run(N, p, nstar=7.2, offset=0.25, t0=1e-9, t_end=1.0, rho=0.01, tol=1e-11, ell=1.0):
    L = 2 * np.pi * nstar; h = L / N; a = 1.0 / nstar
    x = (np.arange(N) + offset) * h
    U0 = np.cos(x / nstar)
    y = x - 0.75 * L                                   # distance to the ascending zero
    win = np.abs(y) < ell
    S = circulant(N, (1, 26, 66, 26, 1), 1 / 120)
    K = circulant(N, (-1, -10, 0, 10, 1), 1 / (24 * h)) + circulant(N, (-1, 2, 0, -2, 1), 1 / (2 * h ** 3))
    f = lambda u: np.abs(u) ** (p - 1) * u
    fp = lambda u: p * np.abs(u) ** (p - 1)
    F0 = splu(S).solve(-(K @ f(U0)))
    U = U0.copy(); t = 0.0
    out, snaps = [], {}
    newton_max = 0; rejected = 0
    targets = [s for s in SNAP if s <= t_end * (1 + 1e-12)]
    start = time.perf_counter()
    while t < t_end * (1 - 1e-12):
        dt = max(t0, rho * t)
        nxt = [s for s in targets if s > t * (1 + 1e-12)]
        if nxt and t + dt > nxt[0] * (1 - 1e-12):
            dt = nxt[0] - t                            # land exactly on the snapshot times
        while True:
            V = U.copy(); ok = False                   # midpoint value; initial guess = current state
            for it in range(25):
                R = 2 * (S @ (V - U)) + dt * (K @ f(V))
                if np.sqrt(np.mean(R ** 2)) < tol:
                    ok = True; break
                J = (2 * S + dt * (K @ sp.diags(fp(V)))).tocsc()
                V = V - splu(J).solve(R)
            if ok:
                break
            dt *= 0.5; rejected += 1
            if dt < 1e-14:
                raise RuntimeError(f'Newton failure at t={t}')
        newton_max = max(newton_max, it)
        U = 2 * V - U; t += dt
        d = U - U0
        out.append((t, float(np.sqrt(h * np.sum(d ** 2))), float(np.max(np.abs(d[win]))), float(np.sqrt(h * np.sum(d[win] ** 2)))))
        for s in targets:
            if abs(t - s) <= 1e-12 * s and s not in snaps:
                sel = np.abs(y) < 2.5 * ell
                snaps[s] = (y[sel].copy(), d[sel].copy())
    meta = dict(N=N, p=p, nstar=nstar, offset=offset, h=h, a=a, ell=ell, rho=rho, t0=t0, steps=len(out), newton_max=newton_max,
                rejected=rejected, F0_norm=float(np.sqrt(h * np.sum(F0 ** 2))), resolved_time=h ** (4 - p) / a ** (p - 1),
                seconds=time.perf_counter() - start)
    return np.array(out), snaps, meta


def slope(t, q, lo, hi):
    sel = (t >= lo) & (t <= hi)
    return float(np.polyfit(np.log(t[sel]), np.log(q[sel]), 1)[0]) if sel.sum() > 5 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', default='4096,8192,16384'); ap.add_argument('--p', default='1.75,2.0,2.25,3.0')
    ap.add_argument('--tend', type=float, default=1.0); ap.add_argument('--rho', type=float, default=0.01)
    ap.add_argument('--tag', default=''); a = ap.parse_args()
    rows, prof, metas = [], [], []
    for p in [float(s) for s in a.p.split(',')]:
        for N in [int(s) for s in a.N.split(',')]:
            data, snaps, meta = run(N, p, t_end=a.tend, rho=a.rho)
            t, E, m, e = data.T
            tr = meta['resolved_time']
            lo, hi = max(30 * tr, 1e-3), a.tend
            meta.update(sup_exponent_predicted=1 / (4 - p), l2_exponent_predicted=(3 / (8 - 2 * p) if p < 2.5 else 1.0),
                        fit_window=[lo, hi], slope_sup=slope(t, m, lo, hi), slope_l2=slope(t, e, lo, hi),
                        slope_sup_early=slope(t, m, 1e-9, 0.03 * tr), slope_global=slope(t, E, lo, hi),
                        at={f'{s:g}': dict(E=float(np.interp(s, t, E)), sup=float(np.interp(s, t, m)), l2=float(np.interp(s, t, e))) for s in SNAP if s <= a.tend})
            metas.append(meta)
            for k in range(0, len(t), 4):
                rows.append(dict(p=p, N=N, t=t[k], E_global=E[k], sup_window=m[k], l2_window=e[k]))
            ystar = lambda s: (meta['a'] ** (p - 1) * s) ** (1 / (4 - p))
            for s, (yy, dd) in snaps.items():
                step = max(1, len(yy) // 1500)
                for k in range(0, len(yy), step):
                    prof.append(dict(p=p, N=N, t=s, y=yy[k], difference=dd[k], xi=yy[k] / ystar(s), scaled=dd[k] / (meta['a'] * ystar(s))))
            print(f"p={p} N={N}: resolved for t >> {tr:.1e}; window [{lo:.1e},{hi:g}]: sup slope {meta['slope_sup']:.3f} (pred {meta['sup_exponent_predicted']:.3f}), "
                  f"L2 slope {meta['slope_l2']:.3f} (pred {meta['l2_exponent_predicted']:.3f}), global {meta['slope_global']:.3f}; early sup slope {meta['slope_sup_early']:.3f}; "
                  f"sup(0.01)={meta['at']['0.01']['sup']:.4e} sup(0.1)={meta['at']['0.1']['sup']:.4e} [{meta['steps']} steps, {meta['rejected']} rej, {meta['seconds']:.0f} s]", flush=True)
    write_csv(RESULTS / f'initial_layer{a.tag}.csv', rows)
    write_csv(RESULTS / f'initial_layer_profiles{a.tag}.csv', prof)
    (RESULTS / f'initial_layer{a.tag}.json').write_text(json.dumps(metas, indent=1))


if __name__ == '__main__':
    main()
