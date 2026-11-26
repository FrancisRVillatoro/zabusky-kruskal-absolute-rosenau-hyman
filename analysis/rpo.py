"""Reflection-symmetric relative periodic orbits of the Galerkin truncations.

The truncated system is reversible under a_n -> conj(a_n) (x -> -x, tau -> -tau)
and equivariant under translations a_n -> a_n exp(-i k_n s).  A solution that
starts from real coefficients alpha_n/2 and, after a time tau_h, is again a
translate of a real state closes up after 2 tau_h as a relative periodic orbit.

Unknowns : alpha_1, alpha_3, ..., alpha_K (real) and tau_h.
Equations: imaginary parts of the first-mode-slice coefficients at tau_h for
           n >= 3, and the invariants I_{p+1} and Q fixed at their values on the
           unit cosine.
Newton iteration; the Jacobian comes from the variational equations.
The flux quadrature breaks continuous translation invariance slightly, so the
closure after a full period is exact only up to the quadrature error; it is
reported as "closure_defect".
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from common import *
from galerkin_ext import Galerkin, diagnostics


def slice_parts(c, n):
    phi = np.angle(c[0])
    return 2 * c * np.exp(-1j * n * phi), phi


def residual_and_jacobian(g, z, I0, Q0, rtol=1e-10, jac=True, atol=1e-14):
    alpha, tauh = z[:-1], z[-1]
    n = g.n_odd; M = len(n)
    c0 = alpha.astype(complex) / 2
    seeds = 0.5 * np.eye(M, dtype=complex) if jac else np.zeros((0, M), complex)
    out = g.tangent_run(seeds_odd=seeds, tau_end=tauh, rtol=rtol, atol=atol, c0=c0, t_eval=[tauh])
    c = out['c'][-1]
    s, phi = slice_parts(c, n)
    I, Q = g.invariants(c0)
    F = np.concatenate([s.imag[1:], [I - I0, Q - Q0]])
    if not jac:
        return F, None
    J = np.zeros((M + 1, M + 1))
    dC = np.vstack([out['odd'][-1], g.rhs(0.0, c)[None, :]])        # derivatives of c(tau_h) w.r.t. alpha_k and tau_h
    for k in range(M + 1):
        dc = dC[k]
        dphi = (dc[0] / c[0]).imag
        ds = 2 * np.exp(-1j * n * phi) * (dc - 1j * n * c * dphi)
        J[:M - 1, k] = ds.imag[1:]
    b = g.flux_coeffs(g.field(c0))[n]
    J[M - 1, :M] = b.real                                           # dI/dalpha_k = <f(u) cos(k_k x)>
    J[M, :M] = 2 * alpha / (1 - (n / g.nstar) ** 2)                 # dQ/dalpha_k
    return F, J


def solve(nstar, cutoff, quad, z0=None, tauh0=0.506, maxit=25, tol=3e-10, verbose=True, jtol=(1e-8, 1e-11)):
    g = Galerkin(nstar, cutoff, quad)
    I0, Q0 = g.invariants(g.cosine())
    M = len(g.n_odd)
    z = np.zeros(M + 1); z[0] = 1.0; z[-1] = tauh0
    if z0 is not None:                                              # continuation from a smaller cutoff
        m0 = len(z0) - 1; z[:min(m0, M)] = z0[:min(m0, M)]; z[-1] = z0[-1]
    hist = []
    for it in range(maxit):
        F, _ = residual_and_jacobian(g, z, I0, Q0, jac=False)
        nrm = float(np.linalg.norm(F)); hist.append(nrm)
        if verbose:
            print(f'  it {it:2d}: |F| = {nrm:.3e}   tau_h = {z[-1]:.9f}', flush=True)
        if nrm < tol or (it > 3 and nrm > 0.5 * hist[-2] and nrm < 1e-8):
            break
        _, J = residual_and_jacobian(g, z, I0, Q0, rtol=jtol[0], atol=jtol[1])
        dz = np.linalg.solve(J, -F)
        lam = 1.0
        while lam > 1e-3:
            Fn, _ = residual_and_jacobian(g, z + lam * dz, I0, Q0, jac=False)
            if np.linalg.norm(Fn) < nrm:
                break
            lam *= 0.5
        z = z + lam * dz
    return g, z, hist, (I0, Q0)


def solve_hybrid(nstar, cutoff, quad, z0=None, tauh0=0.506, tol=3e-10, verbose=True):
    """Powell's hybrid method (MINPACK hybrj): trust region with Broyden updates.

    Residuals are cheap (one base orbit); the variational Jacobian is expensive and
    is evaluated only when the updated matrix stops producing progress, with
    relaxed tolerances.  A final Newton correction uses the exact Jacobian.
    """
    from scipy.optimize import root
    g = Galerkin(nstar, cutoff, quad)
    I0, Q0 = g.invariants(g.cosine())
    M = len(g.n_odd)
    z = np.zeros(M + 1); z[0] = 1.0; z[-1] = tauh0
    if z0 is not None:
        m0 = len(z0) - 1; z[:min(m0, M)] = z0[:min(m0, M)]; z[-1] = z0[-1]
    hist = []; count = dict(f=0, j=0)

    def fun(x):
        F, _ = residual_and_jacobian(g, x, I0, Q0, jac=False)
        count['f'] += 1; hist.append(float(np.linalg.norm(F)))
        if verbose and (count['f'] % 5 == 1 or hist[-1] < 1e-6):
            print(f"  f {count['f']:3d}: |F| = {hist[-1]:.3e}   tau_h = {x[-1]:.9f}", flush=True)
        return F

    def jac(x):
        t0 = time.perf_counter()
        _, J = residual_and_jacobian(g, x, I0, Q0, rtol=1e-8, atol=1e-11)
        count['j'] += 1
        if verbose:
            print(f"  Jacobian {count['j']} ({time.perf_counter() - t0:.0f} s)", flush=True)
        return J

    sol = root(fun, z, jac=jac, method='hybr', options=dict(xtol=1e-13, factor=0.1, maxfev=150))
    z = sol.x
    for it in range(4):                                             # Newton polish
        F, J = residual_and_jacobian(g, z, I0, Q0, rtol=1e-9, atol=1e-12)
        nrm = float(np.linalg.norm(F)); hist.append(nrm)
        if verbose:
            print(f'  polish {it}: |F| = {nrm:.3e}   tau_h = {z[-1]:.9f}', flush=True)
        if nrm < tol:
            break
        z = z + np.linalg.solve(J, -F)
    return g, z, hist, (I0, Q0)


def solve_lm(nstar, cutoff, quad, z0=None, tauh0=0.506, tol=3e-10, verbose=True, max_nfev=60):
    """Levenberg-Marquardt iteration (MINPACK lmder) on the same square system."""
    from scipy.optimize import least_squares
    g = Galerkin(nstar, cutoff, quad)
    I0, Q0 = g.invariants(g.cosine())
    M = len(g.n_odd)
    z = np.zeros(M + 1); z[0] = 1.0; z[-1] = tauh0
    if z0 is not None:
        m0 = len(z0) - 1; z[:min(m0, M)] = z0[:min(m0, M)]; z[-1] = z0[-1]
    hist = []

    def fun(x):
        F, _ = residual_and_jacobian(g, x, I0, Q0, jac=False)
        hist.append(float(np.linalg.norm(F)))
        if verbose:
            print(f"  f {len(hist):3d}: |F| = {hist[-1]:.3e}   tau_h = {x[-1]:.9f}", flush=True)
        return F

    def jac(x):
        return residual_and_jacobian(g, x, I0, Q0, rtol=1e-8, atol=1e-11)[1]

    sol = least_squares(fun, z, jac=jac, method='lm', xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=max_nfev)
    z = sol.x
    F, _ = residual_and_jacobian(g, z, I0, Q0, jac=False); hist.append(float(np.linalg.norm(F)))
    return g, z, hist, (I0, Q0)


def floquet(g, z, rtol=1e-10, atol=1e-14):
    """Monodromy of a -> g_{-2s} Phi_{2 tau_h}(a) in the real coordinates (Re a, Im a)."""
    alpha, tauh = z[:-1], z[-1]
    c0 = alpha.astype(complex) / 2; no = len(g.n_odd)
    seeds = np.vstack([np.eye(no, dtype=complex), 1j * np.eye(no, dtype=complex)])
    out = g.tangent_run(seeds_odd=seeds, tau_end=2 * tauh, rtol=rtol, atol=atol, c0=c0, t_eval=[2 * tauh])
    cT = out['c'][-1]
    phi = np.angle(cT[0] * np.sign(alpha[0]))
    rot = np.exp(-1j * g.n_odd * phi)
    closure = float(np.linalg.norm(cT * rot - c0) / np.linalg.norm(c0))
    W = out['odd'][-1] * rot[None, :]
    M = np.vstack([W.real.T, W.imag.T])
    return M, closure, phi


def analyse(g, z, hist, I0Q0, rtol=1e-10, atol=1e-14):
    M, closure, phi = floquet(g, z, rtol, atol)
    ev, vec = np.linalg.eig(M)
    order = np.argsort(-np.abs(ev)); ev = ev[order]; vec = vec[:, order]
    no = len(g.n_odd); alpha = z[:-1]
    vu = vec[:no, 0].real + 1j * vec[no:, 0].real
    vu = vu / np.linalg.norm(vu)
    if vu[2].imag > 0:
        vu = -vu
    dist = float(np.sqrt((alpha[0] - 1) ** 2 + (alpha[1:] ** 2).sum()))
    # distance in the D sense: least-squares over translations is trivial here (both states real, same slice)
    return dict(nstar=g.nstar, cutoff=g.cutoff, quad=g.quad, newton_residuals=hist, tau_h=float(z[-1]), period=float(2 * z[-1]),
                shift_per_period_over_L=float((-phi / (2 * np.pi)) % 1.0), alpha=alpha.tolist(), distance_to_cosine=dist,
                closure_defect=closure, det_monodromy=float(np.linalg.det(M)),
                multipliers=[[float(e.real), float(e.imag)] for e in ev],
                leading_multiplier=float(abs(ev[0])), leading_is_real=bool(abs(ev[0].imag) < 1e-6),
                second_modulus=float(abs(ev[1])),
                unstable_vector_power_share={int(n): float(abs(v) ** 2) for n, v in zip(g.n_odd, vu)},
                unstable_vector_phase_deg={int(n): float(np.degrees(np.angle(v))) for n, v in zip(g.n_odd, vu)})


def report(res):
    al = res['alpha']
    print(f"   tau_h={res['tau_h']:.8f}  period={res['period']:.8f}  distance to cosine={res['distance_to_cosine']:.5f}  closure defect {res['closure_defect']:.2e}  det M={res['det_monodromy']:.6f}")
    print('   alpha:', ', '.join(f'{2*k+1}:{v:+.5f}' for k, v in enumerate(al[:9])))
    ev = [complex(a, b) for a, b in res['multipliers']]
    print('   largest multipliers:', ', '.join(f'{abs(e):.4f}' + (f'(arg {np.degrees(np.angle(e)):+.0f})' if abs(e.imag) > 1e-6 else ('' if e.real > 0 else '(neg)')) for e in ev[:8]), ' smallest:', ', '.join(f'{abs(e):.4f}' for e in ev[-3:]))
    sh = res['unstable_vector_power_share']; ph = res['unstable_vector_phase_deg']
    print('   unstable vector: share', ', '.join(f'{n}:{100*s:.1f}%' for n, s in list(sh.items())[:6]), ' phase(deg)', ', '.join(f'{n}:{v:+.0f}' for n, v in list(ph.items())[:6]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nstar', type=float, default=7.2)
    ap.add_argument('--cutoffs', default='21,23,25', help='comma list or first:last:step; each cutoff starts from the previous orbit')
    ap.add_argument('--quads', default='', help='quadrature points per cutoff (default 1024)')
    ap.add_argument('--tauh', type=float, default=0.506, help='initial guess of the half period when starting from the cosine')
    ap.add_argument('--method', default='newton', choices=['newton', 'hybrid', 'lm'])
    ap.add_argument('--start', default='', help='file with stored orbits to continue from (default: the cosine)')
    ap.add_argument('--start-cutoff', type=int, default=0, help='cutoff of the stored orbit to start from (default: the last one)')
    ap.add_argument('--out', default='rpo_orbits.json', help='file in results/ for the converged orbits')
    ap.add_argument('--append', action='store_true', help='add the new orbits to those already in --out')
    a = ap.parse_args()
    if ':' in a.cutoffs:                                            # first:last:step
        lo, hi, st = (int(s) for s in a.cutoffs.split(':')); cutoffs = list(range(lo, hi + 1, st))
    else:
        cutoffs = [int(s) for s in a.cutoffs.split(',')]
    quads = [int(s) for s in a.quads.split(',')] if a.quads else [1024] * len(cutoffs)
    out = RESULTS / a.out
    results = json.loads(out.read_text()) if (a.append and out.exists()) else []
    z = None
    if a.start:                                                     # continue from a stored orbit
        stored = json.loads(Path(a.start).read_text())
        prev = stored[-1] if not a.start_cutoff else [r for r in stored if r['cutoff'] == a.start_cutoff][0]
        z = np.array(prev['alpha'] + [prev['tau_h']])
    for cutoff, quad in zip(cutoffs, quads):
        t0 = time.perf_counter()
        print(f'--- nstar={a.nstar} cutoff={cutoff} quad={quad}', flush=True)
        g, z, hist, IQ = dict(newton=solve, hybrid=solve_hybrid, lm=solve_lm)[a.method](a.nstar, cutoff, quad, z0=z, tauh0=a.tauh)
        if hist[-1] >= 1e-8:
            print(f'   not converged (norm of the closing conditions {hist[-1]:.2e}): continuation stopped', flush=True); break
        res = analyse(g, z, hist, IQ, *((1e-10, 1e-14) if cutoff <= 27 else (1e-9, 1e-12))); res['seconds'] = time.perf_counter() - t0
        res['converged'] = True; res['method'] = a.method
        results = [r for r in results if r['cutoff'] != cutoff] + [res]; results.sort(key=lambda r: r['cutoff'])
        report(res); print(f"   {res['seconds']:.0f} s", flush=True)
        out.write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
