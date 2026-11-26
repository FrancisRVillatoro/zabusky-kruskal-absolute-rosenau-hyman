"""Consistency of galerkin_ext.py with the archived Galerkin results of phase 9B."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *
from galerkin_ext import Galerkin, diagnostics

S = Archives(sys.argv[1] if len(sys.argv) > 1 else None)
ref = {(r['case'], int(r['j'])): r for r in S.rows(ZBL, 'fase9b/results_local/summary/events.csv') if r['tau']}
for case, cutoff, quad in (('central_K21', 21, 512), ('central_K35', 35, 1024)):
    g = Galerkin(7.2, cutoff, quad)
    t = np.arange(round(4.25 * 3200) + 1) / 3200
    t0 = time.perf_counter()
    sol = g.flow(g.cosine(), 4.25, rtol=1e-9, t_eval=t)
    D, r, H = diagnostics(sol.y.T)
    worst = 0
    for j in (1, 2, 3, 4):
        ids = np.flatnonzero((t >= j - .2) & (t <= j + .2)); i = ids[np.argmin(D[ids])]
        rr = ref[(case, j)]
        worst = max(worst, abs(D[i] - float(rr['D'])), abs(t[i] - float(rr['tau'])))
        print(f"{case} j={j}: tau={t[i]:.6f} (archived {float(rr['tau']):.6f})  D={D[i]:.8f} (archived {float(rr['D']):.8f})")
    I = [g.invariants(c) for c in sol.y.T[::400]]
    print(f"   max difference from archive {worst:.2e}; rel. drift of I {max(abs(np.array([i[0] for i in I]) / I[0][0] - 1)):.1e}, of Q {max(abs(np.array([i[1] for i in I]) / I[0][1] - 1)):.1e}; {time.perf_counter() - t0:.1f} s")
    assert worst < 5e-6
print('TEST_GALERKIN_OK')
