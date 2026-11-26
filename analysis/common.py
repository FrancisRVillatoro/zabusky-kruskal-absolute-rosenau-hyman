"""Shared readers and modal definitions of the analysis package.

All inputs are read directly from the immutable ZIP archives of the repository
(directory ``archives/`` of the repository root).  Nothing in this package
re-runs a production campaign.  Time is tau = t/T with T the crossing time.
"""
from __future__ import annotations
import io, json, csv, zipfile
from pathlib import Path
import numpy as np

P = 2.25
HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)

ZB = 'Fase9B_CNSNS_resultados_20261125.zip'
ZBL = 'CNSNS_Fase9B_20261125.zip'
ZBC = 'CNSNS_Fase9B_cierre_20261125.zip'
ZA = 'Fase9A_HDF5_3831810.zip'
ZAL = 'CNSNS_Fase9A_20261124.zip'
ZAC = 'CNSNS_Fase9A_cierre_20261125.zip'
ZC = 'Fase9C_CNSNS_resultados_20261125.zip'
ZCC = 'CNSNS_Fase9C_cierre_20261125.zip'
ZBASE = 'base_revision.zip'


def repository_root(arg: str | None = None) -> Path:
    """Directory that contains ``archives/``."""
    cands = [Path(arg)] if arg else [HERE.parent, Path.cwd()]
    for c in cands:
        if (c / 'archives').is_dir():
            return c
    raise FileNotFoundError('repository root with archives/ not found; use --root')


class Archives:
    def __init__(self, root: str | None = None):
        self.root = repository_root(root)
        self._z: dict[str, zipfile.ZipFile] = {}

    def zip(self, name: str) -> zipfile.ZipFile:
        if name not in self._z:
            self._z[name] = zipfile.ZipFile(self.root / 'archives' / name)
        return self._z[name]

    def names(self, name: str) -> list[str]:
        return self.zip(name).namelist()

    def bytes(self, name: str, member: str) -> bytes:
        return self.zip(name).read(member)

    def rows(self, name: str, member: str) -> list[dict]:
        text = self.bytes(name, member).decode('utf-8')
        return list(csv.DictReader(io.StringIO(text)))

    def json(self, name: str, member: str):
        return json.loads(self.bytes(name, member))

    def npz(self, name: str, member: str) -> dict:
        with np.load(io.BytesIO(self.bytes(name, member)), allow_pickle=False) as f:
            return {k: f[k] for k in f.files}


def fnum(x, default=np.nan) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def crossing(nstar: float, p: float = P) -> float:
    """Crossing time T = L/c_char for unit amplitude."""
    return 2 * np.pi * nstar * 2 * p / (p + 1)


def multipliers(n, nstar: float):
    """Continuum multiplier m_n = k_n (1 - k_n^2), k_n = n/nstar."""
    k = np.asarray(n, float) / nstar
    return k * (1 - k * k)


def pade_multiplier(n, nstar: float, N: int):
    """Real multiplier of the Pade operator S^{-1}(B+C) (manuscript Eq. symbol)."""
    n = np.asarray(n, float)
    h = 2 * np.pi * nstar / N
    th = 2 * np.pi * n / N
    mass = (66 + 52 * np.cos(th) + 2 * np.cos(2 * th)) / 120
    return np.sin(th) / h * ((np.cos(th) + 5) / 6 - 4 * (np.sin(th / 2) / h) ** 2) / mass


def momentum_weights(n, nstar: float):
    """Weights 1/(1-k_n^2) of the translation momentum; inf at an exact null."""
    k2 = (np.asarray(n, float) / nstar) ** 2
    w = np.full(k2.shape, np.inf)
    ok = np.abs(1 - k2) > 1e-12
    w[ok] = 1.0 / (1 - k2[ok])
    return w


def momentum_from_modes(c, n, nstar: float):
    """Q = sum_n 4|a_n|^2/(1-k_n^2) over the stored positive harmonics.

    A harmonic sitting exactly at the null (integer nstar) is excluded: its
    coefficient is a separate invariant.
    """
    w = momentum_weights(n, nstar)
    w = np.where(np.isfinite(w), w, 0.0)
    return (4 * np.abs(c) ** 2 * w).sum(axis=-1)


def momentum_from_field(u, nstar: float):
    """Q from a real periodic field sampled on N nodes (all harmonics, mean excluded)."""
    u = np.asarray(u, float)
    N = u.shape[-1]
    a = np.fft.rfft(u, axis=-1) / N
    n = np.arange(a.shape[-1])
    e = 4 * np.abs(a) ** 2
    if N % 2 == 0:
        e[..., -1] *= 0.5
    w = momentum_weights(n, nstar)
    w = np.where(np.isfinite(w), w, 0.0)
    w[0] = 0.0
    return (e * w).sum(axis=-1)


def slice_coefficients(c, n):
    """First-mode slice: 2 a_n exp(-i n arg a_1), so the fundamental is real positive."""
    c = np.asarray(c)
    rot = np.conj(c[..., :1]) / np.abs(c[..., :1])
    return 2 * c * rot ** np.asarray(n)[None, :] if c.ndim == 2 else 2 * c * rot ** np.asarray(n)


def prominent_minima(tau, D, prominence=0.03, tmin=0.2):
    """Chronological prominent minima of D (same rule as fase9b.analyze.free_catalog)."""
    from scipy.signal import find_peaks
    ids, props = find_peaks(-np.asarray(D), prominence=prominence)
    return [(int(i), float(pr)) for i, pr in zip(ids, props['prominences']) if tau[i] >= tmin]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def sig(x, n=6):
    return float(f'{x:.{n}g}')
