"""Fourier-Galerkin models of the inviscid absolute Rosenau-Hyman equation.

Conventions follow fase9b.galerkin/modal: u = sum_n a_n exp(i k_n x) + c.c.,
k_n = n/nstar, time tau = t/T, da_n/dtau = -i T m_n b_n, m_n = k_n(1-k_n^2),
b_n = Fourier coefficient of |u|^(p-1) u evaluated by periodic quadrature.

Two extensions with respect to fase9b.galerkin:
  * arbitrary sets of retained harmonics (odd only, or all up to the cutoff);
  * the variational (tangent) equations dv_n/dtau = -i T m_n [p|u|^(p-1) v]_n,
    integrated together with the base orbit for several initial directions.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.integrate import solve_ivp
from common import crossing, multipliers, momentum_weights, P


class Galerkin:
    def __init__(self, nstar: float, cutoff: int, quad: int = 1024, p: float = P, probe: str = 'none'):
        self.nstar, self.cutoff, self.quad, self.p = float(nstar), int(cutoff), int(quad), float(p)
        self.T = crossing(nstar, p)
        self.n_odd = np.arange(1, cutoff + 1, 2)
        self.n_even = np.arange(2, cutoff + 1, 2)
        self.m_odd = multipliers(self.n_odd, nstar)
        self.m_even = multipliers(self.n_even, nstar)
        if probe != 'none':
            mode, fac = probe.split(':')
            self.m_odd[self.n_odd == int(mode)] *= float(fac)
        self._v = np.zeros(quad // 2 + 1, complex)
        if 2 * cutoff >= quad:
            raise ValueError('quadrature too coarse for this cutoff')

    # --- fields and fluxes -------------------------------------------------
    def field(self, c, n=None):
        v = self._v; v[:] = 0; v[self.n_odd if n is None else n] = c
        return np.fft.irfft(v, n=self.quad, norm='forward')

    def flux_coeffs(self, u):
        return np.fft.rfft(np.abs(u) ** (self.p - 1) * u, norm='forward')

    def rhs(self, tau, c):
        return -1j * self.T * self.m_odd * self.flux_coeffs(self.field(c))[self.n_odd]

    def invariants(self, c):
        u = self.field(c)
        I = np.mean(np.abs(u) ** (self.p + 1)) / (self.p + 1)
        w = momentum_weights(self.n_odd, self.nstar)                 # a harmonic exactly at the null is excluded
        Q = np.sum(4 * np.abs(c) ** 2 * np.where(np.isfinite(w), w, 0.0))
        return I, Q

    # --- flows --------------------------------------------------------------
    def flow(self, c0, tau_end, rtol=1e-10, atol=None, t_eval=None, dense=False):
        atol = rtol * 0.005 if atol is None else atol
        sol = solve_ivp(self.rhs, (0.0, tau_end), np.asarray(c0, complex), method='DOP853', rtol=rtol, atol=atol,
                        t_eval=t_eval, dense_output=dense)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def cosine(self):
        c0 = np.zeros(len(self.n_odd), complex); c0[0] = 0.5
        return c0

    # --- variational equations (batched FFTs) --------------------------------
    def tangent_run(self, seeds_even=(), seeds_odd=(), tau_end=4.25, sample=400, rtol=1e-10, c0=None, t_eval=None, atol=None):
        """Integrate a base orbit together with tangent vectors.

        seeds_even / seeds_odd: sequences of complex coefficient vectors over
        self.n_even / self.n_odd (initial tangent directions in each sector).
        The tangent equations are real-linear in the perturbation field, so a
        complex seed i*e_k is a different direction from e_k.
        Returns sampled times, base coefficients and tangent coefficients.
        """
        no, ne = len(self.n_odd), len(self.n_even)
        Vo = np.asarray(seeds_odd, complex).reshape(-1, no); Ve = np.asarray(seeds_even, complex).reshape(-1, ne)
        ko, ke = len(Vo), len(Ve)
        y0 = np.concatenate([self.cosine() if c0 is None else np.asarray(c0, complex), Vo.ravel(), Ve.ravel()])
        T, p, Q = self.T, self.p, self.quad
        spec = np.zeros((1 + ko + ke, Q // 2 + 1), complex)
        mo, me, io, ie = self.m_odd, self.m_even, self.n_odd, self.n_even

        def rhs(tau, y):
            spec[0, io] = y[:no]
            if ko:
                spec[1:1 + ko, io] = y[no:no + ko * no].reshape(ko, no)
            if ke:
                spec[1 + ko:, ie] = y[no + ko * no:].reshape(ke, ne)
            f = np.fft.irfft(spec, n=Q, axis=1, norm='forward')
            u = f[0]; au = np.abs(u); w = au ** (p - 1)
            out = np.empty_like(y)
            out[:no] = -1j * T * mo * np.fft.rfft(w * u, norm='forward')[io]
            if ko + ke:
                tf = np.fft.rfft((p * w)[None, :] * f[1:], axis=1, norm='forward')
                if ko:
                    out[no:no + ko * no] = (-1j * T * mo[None, :] * tf[:ko][:, io]).ravel()
                if ke:
                    out[no + ko * no:] = (-1j * T * me[None, :] * tf[ko:][:, ie]).ravel()
            return out

        t = np.linspace(0.0, tau_end, int(round(tau_end * sample)) + 1) if t_eval is None else np.asarray(t_eval)
        start = time.perf_counter()
        sol = solve_ivp(rhs, (0.0, tau_end), y0, method='DOP853', rtol=rtol, atol=rtol * 0.005 if atol is None else atol, t_eval=t)
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y.T
        return dict(tau=sol.t, c=y[:, :no], odd=y[:, no:no + ko * no].reshape(len(sol.t), ko, no),
                    even=y[:, no + ko * no:].reshape(len(sol.t), ke, ne), nfev=sol.nfev, seconds=time.perf_counter() - start)


def diagnostics(c):
    e = 4 * np.abs(c) ** 2
    r = 2 * np.abs(c[..., 0]); H = e[..., 1:].sum(axis=-1)
    return np.sqrt((r - 1) ** 2 + H), r, H


def rms_gain(v, v0):
    """Ratio of RMS norms of the tangent field to its initial value."""
    return np.sqrt((np.abs(v) ** 2).sum(axis=-1) / (np.abs(v0) ** 2).sum())
