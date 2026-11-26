"""Primary cosmic-ray fluxes at the top of the atmosphere: the spec kinds of the flux templates, the registry of
analytical models, the built-in models, tabulated fluxes, and the split of a primary flux into nucleons for MCEq.

Three kinds of template spec (FluxTemplate.get_or_build, FluxHistory in flux):

  'Baseline'    steady background: a crflux model, by default Hillas-Gaisser H3a ({} means this), or a steady
                Analytical / Tabulated flux: {"kind": "Baseline", "model": "H4a"},
                {"kind": "Baseline", "model": ("GlobalSplineFitBeta", None)}, {"kind": "Baseline", "flux": {...}}
  Analytical    a function f(E_gev, t_yr, **params) -> {nucleus: flux}, registered under a short name
                (register_flux_model) and used as {"kind": "SN", "distance_pc": 100}, or passed directly as
                {"kind": "Analytical", "function": f, ...}. Built in: "SN", "Egal"; list_flux_models() shows every
                registered model with its parameters.
  'Tabulated'   arrays, one entry per nucleus (see _tabulated_data):
                {"kind": "Tabulated", "name": ..., "time_yr": t[n_time],
                 "H": {"energy_gev": E[n_energy], "flux_cm2_s_sr_gev": F[n_time, n_energy]}, "He": {...}}

Every flux is per nucleus, per GeV of total energy per nucleus, in cm^-2 s^-1 sr^-1, at the top of the atmosphere
and before the geomagnetic cutoff, which is applied afterwards at the site's cutoff (_apply_cutoff, as crflux's
nucleus_flux). Nuclei are named by mass group ('H', 'He', 'CNO', 'MgAlSi', 'Fe'), by nucleus ('O16', 'Fe56') or by
CORSIKA id; each is split into its nucleons for MCEq (superposition: nucleon_spectra). Transient fluxes (events) are
the excess over the baseline. Time grids are fixed per model (log-spaced; keys t_min_yr, t_max_yr, n_times; 'tcut' =
t_max_yr): the detector refines its own time grid around the template times.

A spec becomes, through _primary_source, a factory of PrimarySource objects (one per template time), which the
solvers of flux turn into secondary fluxes.
"""

import functools
import hashlib
import inspect
import json
import re

import numpy as np

from ._core import SECONDS_PER_YEAR, fragment_za

__all__ = ['list_flux_models', 'register_flux_model']

# crflux is needed only to build templates (MCEq); without it the specs can still be read. crflux >= 2.3 renamed the
# Fe group id 5426 -> 5626 (A = 56): LEGACY_IDS maps the old id.
try:
    from crflux.models import PrimaryFlux
except ImportError:
    PrimaryFlux = object
try:
    from crflux.models import LEGACY_IDS as _CRFLUX_LEGACY_IDS
except ImportError:
    _CRFLUX_LEGACY_IDS = {}


CM_PER_PC = 3.086e18
SPEED_OF_LIGHT_CM_S = 2.998e10
MASS_GROUP_IDS = {'H': 14, 'p': 14, 'proton': 14, 'He': 402, 'CNO': 1206, 'MgAlSi': 2814, 'Fe': 5626}
DEFAULT_TIME_GRID = {'t_min_yr': 10.0, 't_max_yr': 1e6, 'n_times': 200}
_CRFLUX_BASELINES = {'H3a': ('HillasGaisser2012', 'H3a'), 'H4a': ('HillasGaisser2012', 'H4a')}


@functools.lru_cache(maxsize=None)
def nucleus_id(name):
    """CORSIKA id (100 A + Z, protons 14) of a mass group ('H', 'He', 'CNO', 'MgAlSi', 'Fe' = Fe56), a nucleus
    ('O16', 'Fe56', 'alpha'; parsed by _core.fragment_za) or an id (crflux's old Fe id 5426 is read as 5626)."""
    if isinstance(name, (int, np.integer)):
        return _CRFLUX_LEGACY_IDS.get(int(name), int(name))
    if name in MASS_GROUP_IDS:
        return MASS_GROUP_IDS[name]
    za = fragment_za(str(name))
    if za is not None:
        return 14 if za == (1, 1) else 100 * za[1] + za[0]
    raise ValueError(f"unknown nucleus {name!r}: use {sorted(MASS_GROUP_IDS)}, a nucleus like 'Fe56' or a CORSIKA id.")


def charge_and_mass(cid):
    """(Z, A) of a CORSIKA id (14 = proton)."""
    return (1, 1) if cid == 14 else (cid % 100, cid // 100)


# --- Analytical models: registry --------------------------------------------------------------------------------------
_FLUX_MODELS = {}


def register_flux_model(name, time_grid=None):
    """Decorator registering an analytical primary flux under a short name, usable as {"kind": name, ...}.

    The function is called as f(E_gev, t_yr, **params) and returns {nucleus: flux} (see the section comment for
    units and nucleus names); E_gev is an array of total energies per nucleus, t_yr the time since the event
    (None for a steady model). Parameters are the function's keyword arguments, with their defaults.

    Args:
        name (str): The 'kind' of the specs.
        time_grid (dict, optional): Default template times {'t_min_yr', 't_max_yr', 'n_times'} (log-spaced).
            None for a steady flux (a baseline). Defaults to None.
    """
    def decorate(function):
        """Registers function under name and returns it unchanged."""
        _FLUX_MODELS[name] = {'function': function, 'time_grid': None if time_grid is None else dict(time_grid)}
        return function
    return decorate


def flux_model_parameters(name_or_function):
    """{parameter: default} of an analytical model (registered name or function); required ones map to None."""
    function = _FLUX_MODELS[name_or_function]['function'] if isinstance(name_or_function, str) else name_or_function
    out = {}
    for p in list(inspect.signature(function).parameters.values())[2:]:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        out[p.name] = None if p.default is p.empty else p.default
    return out


def list_flux_models(verbose=True):
    """Registered analytical flux models: name, function, default time grid and parameters with defaults.

    Args:
        verbose (bool, optional): Print them. Defaults to True.

    Returns:
        dict: {name: {'function', 'time_grid' (None = steady), 'parameters' {name: default, None if required},
        'description' (first sentence or clause of the docstring), 'doc' (the whole docstring)}}.
    """
    out = {}
    for name, entry in _FLUX_MODELS.items():
        f = entry['function']
        text = ' '.join((inspect.getdoc(f) or '').split('\n\n')[0].split())
        first = re.split(r'(?<=[a-z0-9)])\. |: ', text, maxsplit=1)[0]
        out[name] = {'function': f, 'time_grid': entry['time_grid'], 'parameters': flux_model_parameters(name),
                     'description': first, 'doc': inspect.getdoc(f) or ''}
    if verbose:
        for name, info in out.items():
            grid = info['time_grid']
            when = ("steady" if grid is None else
                    f"times {grid['t_min_yr']:g}-{grid['t_max_yr']:g} yr ({grid['n_times']}, log)")
            print(f'{{"kind": "{name}"}}  {info["function"].__name__}: {info["description"]}')
            print(f"    {when}")
            for p, default in info['parameters'].items():
                print(f"    {p} = {'(required)' if default is None else repr(default)}")
    return out


def _check_parameters(function, params, label):
    """The function's keyword parameters filled with params; raises on unknown or missing ones."""
    sig = list(inspect.signature(function).parameters.values())[2:]
    takes_any = any(p.kind == p.VAR_KEYWORD for p in sig)
    known = {p.name for p in sig if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
    unknown = set(params) - known
    if unknown and not takes_any:
        raise ValueError(f"{label}: unknown parameters {sorted(unknown)}; known: {sorted(known)}.")
    missing = [p.name for p in sig if p.default is p.empty and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
               and p.name not in params]
    if missing:
        raise ValueError(f"{label}: missing parameters {missing}.")
    return {**{p.name: p.default for p in sig if p.default is not p.empty and p.name in known}, **params}


def _time_grid(grid):
    """Template times [yr] of a time grid {'t_min_yr', 't_max_yr', 'n_times'}: log-spaced. Called by
    _function_source."""
    t_min, t_max, n = float(grid['t_min_yr']), float(grid['t_max_yr']), int(grid['n_times'])
    if not 0 < t_min < t_max:
        raise ValueError(f"time grid needs 0 < t_min_yr < t_max_yr, got {t_min}, {t_max}.")
    return np.geomspace(t_min, t_max, n)


class PrimarySource:
    """A primary flux at the top of the atmosphere, as the solvers use it: the flux of each nucleus
    [m^-2 s^-1 sr^-1 GeV^-1, crflux units] at given total energies per nucleus, before the geomagnetic cutoff.
    The split into nucleons and the cutoff are applied by nucleon_spectra / total_nucleus_flux (response solver)
    or by crflux through _SourceAsCrflux (direct solver, MCEq)."""

    nucleus_ids = ()
    name = 'primary'

    def fluxes(self, energies):
        """{CORSIKA id: flux} at {CORSIKA id: energies per nucleus [GeV]}."""
        raise NotImplementedError


class _FunctionSource(PrimarySource):
    """An analytical flux function (see register_flux_model) at one time. The function is called once per
    evaluation, with the energies of every nucleus concatenated, and its fluxes [cm^-2] are converted to crflux units
    [m^-2] (x 1e4)."""

    def __init__(self, function, t_yr, params):
        self.function, self.t_yr, self.params = function, t_yr, dict(params)
        probe = function(np.array([10.0]), t_yr, **self.params)
        self._names = {nucleus_id(k): k for k in probe}
        self.nucleus_ids = sorted(self._names)
        self.name = getattr(function, '__name__', 'flux model')

    def fluxes(self, energies):
        """See PrimarySource.fluxes."""
        ids = [cid for cid in self.nucleus_ids if cid in energies]
        arrays = [np.atleast_1d(np.asarray(energies[cid], dtype=float)) for cid in ids]
        edges = np.cumsum([0] + [a.size for a in arrays])
        values = self.function(np.concatenate(arrays), self.t_yr, **self.params)
        out = {}
        for k, cid in enumerate(ids):
            v = np.broadcast_to(np.asarray(values[self._names[cid]], dtype=float), (edges[-1],))
            out[cid] = v[edges[k]:edges[k + 1]] * 1e4
        return out


class _CrfluxSource(PrimarySource):
    """A crflux model (the baselines), taken without its own geomagnetic cutoff."""

    def __init__(self, model):
        if not getattr(model, 'nucleus_ids', None):
            raise ValueError(f"{type(model).__name__} gives nucleons only, without nuclei: not supported.")
        self.model = model
        self.nucleus_ids = list(model.nucleus_ids)
        self.name = getattr(model, 'name', type(model).__name__)

    def fluxes(self, energies):
        """See PrimarySource.fluxes."""
        return {cid: self.model.nucleus_flux(cid, E) for cid, E in energies.items()}


def _cutoff_energy(Z, A, rigidity_gv):
    """Total energy per nucleus at the cutoff rigidity, with the installed crflux's convention: sqrt((Z R)^2 + M^2)
    (crflux >= 2.3) or Z R (earlier), so the response and direct solvers cut the same way."""
    if hasattr(PrimaryFlux, 'cutoff_energy'):
        return PrimaryFlux.cutoff_energy(Z, A, rigidity_gv)
    return Z * rigidity_gv


def _apply_cutoff(cid, E, flux, geomagnetic_cutoff):
    """Flux set to 0 below the geomagnetic cutoff (as crflux's nucleus_flux; no cut for None or 0)."""
    if not geomagnetic_cutoff:
        return flux
    Z, A = charge_and_mass(cid)
    return np.where(E >= _cutoff_energy(Z, A, geomagnetic_cutoff), flux, 0.0)


def nucleon_spectra(source, e_nucleon_gev, geomagnetic_cutoff):
    """Proton and neutron fluxes at the given energies per nucleon (superposition, as crflux's p_and_n_flux): a
    nucleus (Z, A) of energy A E gives Z A phi(A E) protons and (A - Z) A phi(A E) neutrons per GeV."""
    e = np.atleast_1d(np.asarray(e_nucleon_gev, dtype=float))
    energies = {cid: e * charge_and_mass(cid)[1] for cid in source.nucleus_ids}
    fluxes = source.fluxes(energies)
    p, n = 0, 0
    for cid in source.nucleus_ids:
        Z, A = charge_and_mass(cid)
        phi = _apply_cutoff(cid, energies[cid], fluxes[cid], geomagnetic_cutoff)
        p = p + Z * A * phi
        n = n + (A - Z) * A * phi
    return p, n


def total_nucleus_flux(source, E_gev, geomagnetic_cutoff):
    """All-nucleus flux at the given energies per nucleus, after the cutoff (as crflux's total_flux)."""
    E = np.atleast_1d(np.asarray(E_gev, dtype=float))
    fluxes = source.fluxes({cid: E for cid in source.nucleus_ids})
    return sum(_apply_cutoff(cid, E, fluxes[cid], geomagnetic_cutoff) for cid in source.nucleus_ids)


class _SourceAsCrflux(PrimaryFlux):
    """A PrimarySource as a crflux model, for MCEq (direct solver); crflux applies the cutoff and the split."""

    def __init__(self, source, geomagnetic_cutoff=5.0):
        super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
        self.source = source
        self.nucleus_ids = list(source.nucleus_ids)
        self.name = self.sname = source.name

    def _nucleus_flux(self, corsika_id, E):
        """crflux's hook: the flux of one nucleus [m^-2 s^-1 sr^-1 GeV^-1] at energies per nucleus E."""
        E = np.atleast_1d(np.asarray(E, dtype=float))
        if corsika_id not in self.nucleus_ids:
            return np.zeros_like(E)
        return self.source.fluxes({corsika_id: E})[corsika_id]


# --- Built-in analytical models ---------------------------------------------------------------------------------------
@register_flux_model("SN", time_grid={'t_min_yr': 10.0, 't_max_yr': 5e5, 'n_times': 250})
def thomas_yelland_sn(E_gev, t_yr, distance_pc, q0_per_gev=3e52, index=2.2, e_cut_gev=1e6, d0_cm2_s=2e28,
                      delta=1.0 / 3.0):
    """Nearby supernova, diffusive transport: Thomas & Yelland 2023, ApJ 950, 41, Sect. 2. Protons released at
    once with Q(E) = Q0 (E / GeV)^-index exp(-E / E_c), D = D0 (E / GeV)^delta, n = Q / (pi^3/2 r_diff^3)
    exp(-r^2 / r_diff^2) with r_diff = sqrt(4 D t), I = c n / 4 pi. Defaults are the paper's."""
    E = np.asarray(E_gev, dtype=float)
    D = d0_cm2_s * E ** delta
    r_diff = np.sqrt(4 * D * t_yr * SECONDS_PER_YEAR)
    n = (q0_per_gev * E ** -index * np.exp(-E / e_cut_gev) / (np.pi ** 1.5 * r_diff ** 3)
         * np.exp(-(distance_pc * CM_PER_PC) ** 2 / r_diff ** 2))
    return {'H': SPEED_OF_LIGHT_CM_S / (4 * np.pi) * n}


# Hillas-Gaisser H3a population 3 (a [m^-2 s^-1 sr^-1 GeV^gamma], gamma, Z), as in crflux
_H3A_POP3 = {'H': (1.7, 1.4, 1), 'He': (1.7, 1.4, 2), 'CNO': (1.14, 1.4, 6), 'MgAlSi': (1.14, 1.4, 14),
             'Fe': (1.14, 1.4, 26)}


@register_flux_model("Egal", time_grid={'t_min_yr': 10.0, 't_max_yr': 1e6, 'n_times': 200})
def egal_boost(E_gev, t_yr, boost=1.0, cutoff_rigidity_gv=2e9, duration_yr=1e5, decay_time_yr=1e5):
    """Phenomenological boost of H3a population 3: boost * a E^(-gamma-1) exp(-E / (Z R_c)), constant for
    duration_yr, then decaying as exp(-(t - duration_yr) / decay_time_yr). Not a physical source model."""
    E = np.asarray(E_gev, dtype=float)
    f_t = 1.0 if t_yr <= duration_yr else float(np.exp(-(t_yr - duration_yr) / decay_time_yr))
    return {name: boost * f_t * a * 1e-4 * E ** (-gamma - 1.0) * np.exp(-E / (Z * cutoff_rigidity_gv))
            for name, (a, gamma, Z) in _H3A_POP3.items()}


# --- Tabulated fluxes -------------------------------------------------------------------------------------------------
# energy-like key -> (flux key, conversion factor to energy per nucleus: 'one', 'A' or 'Z')
_TABLE_AXES = {
    'energy_gev': ('flux_cm2_s_sr_gev', 'one'),
    'energy_per_nucleon_gev': ('flux_cm2_s_sr_gev', 'A'),
    'rigidity_gv': ('flux_cm2_s_sr_gv', 'Z'),
}


def _normalised_composition(composition):
    """Fractions of a composition {nucleus: fraction} normalised to sum to 1. Called by _legacy_tabulated (an array
    flux split among nuclei, primary_spectrum_spec).

    Raises:
        ValueError: If the fractions do not add up to more than 0.
    """
    total = float(sum(composition.values()))
    if total <= 0:
        raise ValueError("composition fractions must add up to more than 0.")
    return {k: float(v) / total for k, v in composition.items()}


def _tabulated_data(spec):
    """Arrays of a 'Tabulated' spec, per nucleus on its own energy grid, converted to energy per nucleus.

    Spec: {"time_yr": t[n_time] (omit for a steady flux), "<nucleus>": {"energy_gev": E[n_energy],
    "flux_cm2_s_sr_gev": F[n_time, n_energy]}, ...}; the energy may instead be "energy_per_nucleon_gev" (flux per GeV
    per nucleon) or "rigidity_gv" with "flux_cm2_s_sr_gv". Flux rows are times, columns energies; a steady flux is
    1D. The older form {"energy_gev": E, "flux": {...} or array, "composition": ..., "t_since_yr": ...}
    (primary_spectrum_spec before) is accepted too.

    Returns:
        dict: 'time_yr' (array or None), 'nuclei' {CORSIKA id: (E per nucleus, flux rows per GeV per nucleus)},
        'hash'.
    """
    spec = dict(spec)
    if 'flux' in spec and 'energy_gev' in spec:
        spec = _legacy_tabulated(spec)
    times = spec.pop('time_yr', None)
    times = None if times is None else np.asarray(times, dtype=float)
    if times is not None and (times.ndim != 1 or times.size < 1 or np.any(np.diff(times) <= 0) or times[0] <= 0):
        raise ValueError("Tabulated: 'time_yr' must be 1D, positive and increasing.")
    nuclei = {}
    for name, entry in spec.items():
        if name in ('kind', 'name'):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Tabulated: {name!r} should be a dict like "
                             f"{{'energy_gev': ..., 'flux_cm2_s_sr_gev': ...}}.")
        axes = [k for k in _TABLE_AXES if k in entry]
        if len(axes) != 1:
            raise ValueError(f"Tabulated {name!r}: give exactly one of {list(_TABLE_AXES)}.")
        flux_key, factor = _TABLE_AXES[axes[0]]
        if flux_key not in entry:
            raise ValueError(f"Tabulated {name!r}: with {axes[0]!r} the flux key is {flux_key!r}.")
        cid = nucleus_id(name)
        Z, A = charge_and_mass(cid)
        c = {'one': 1.0, 'A': float(A), 'Z': float(Z)}[factor]
        x = np.asarray(entry[axes[0]], dtype=float)
        f = np.asarray(entry[flux_key], dtype=float)
        if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0) or x[0] <= 0:
            raise ValueError(f"Tabulated {name!r}: {axes[0]} must be 1D, positive and increasing.")
        expected = (x.size,) if times is None else (times.size, x.size)
        if f.shape != expected:
            raise ValueError(f"Tabulated {name!r}: flux has shape {f.shape}, expected {expected} "
                             f"([n_time, n_energy]{'' if times is not None else ', steady: [n_energy]'}).")
        if np.any(f < 0):
            raise ValueError(f"Tabulated {name!r}: negative flux.")
        nuclei[cid] = (x * c, np.atleast_2d(f) / c)
    if not nuclei:
        raise ValueError("Tabulated: no nucleus given.")
    digest = _spec_hash({'time_yr': times, 'nuclei': {str(cid): list(nuclei[cid]) for cid in sorted(nuclei)}})
    return {'time_yr': times, 'nuclei': nuclei, 'hash': digest}


def _legacy_tabulated(spec):
    """The older single-grid tabulated form {"energy_gev": E, "flux": dict or array, "composition", "t_since_yr",
    "name"} as a 'Tabulated' spec. Called by _tabulated_data and primary_spectrum_spec."""
    energy = np.asarray(spec['energy_gev'], dtype=float)
    flux = spec['flux']
    if isinstance(flux, dict):
        groups = flux
    else:
        comp = _normalised_composition(spec.get('composition') or {'H': 1.0})
        groups = {g: np.asarray(flux, dtype=float) * frac for g, frac in comp.items()}
    out = {'kind': 'Tabulated', 'name': spec.get('name', 'primary')}
    if spec.get('t_since_yr') is not None:
        out['time_yr'] = spec['t_since_yr']
    merged = {}
    for g, f in groups.items():
        cid = nucleus_id(g)
        merged[cid] = merged.get(cid, 0.0) + np.asarray(f, dtype=float)
    for cid, f in merged.items():
        out[cid] = {'energy_gev': energy, 'flux_cm2_s_sr_gev': f}
    return out


def primary_spectrum_spec(energy_gev, flux, composition=None, t_since_yr=None, name="primary"):
    """A 'Tabulated' spec from one energy grid: flux an array (split with composition, at equal energy per
    nucleus; default all protons) or a dict nucleus -> array; [n_time, n_energy] with t_since_yr for an event."""
    return _legacy_tabulated({'energy_gev': energy_gev, 'flux': flux, 'composition': composition,
                              't_since_yr': t_since_yr, 'name': name})


class _TableSource(PrimarySource):
    """A 'Tabulated' flux at one of its times (the nearest), interpolated log-log in energy and zero outside the
    table (no extrapolation, unlike the templates), in crflux units [m^-2] (x 1e4)."""

    def __init__(self, data, t_yr=None):
        row = 0 if t_yr is None or data['time_yr'] is None else int(np.argmin(np.abs(data['time_yr'] - t_yr)))
        self._tables = {cid: (np.log(E), np.log(np.clip(F[row] * 1e4, 1e-300, None)))
                        for cid, (E, F) in data['nuclei'].items()}
        self.nucleus_ids = sorted(self._tables)
        self.name = 'Tabulated'

    def fluxes(self, energies):
        """See PrimarySource.fluxes."""
        out = {}
        for cid, E in energies.items():
            E = np.atleast_1d(np.asarray(E, dtype=float))
            log_e, log_f = self._tables[cid]
            out[cid] = np.exp(np.interp(np.log(E), log_e, log_f, left=-np.inf, right=-np.inf))
        return out


# --- From a spec to a primary model -----------------------------------------------------------------------------------
# Functions defined here were in flux before primaries was split out: they keep that module in _spec_hash, so the
# template names (and the files saved under them) do not change.
_HASH_MODULE_ALIASES = {__name__: __name__.rsplit('.', 1)[0] + '.flux'}


def _spec_hash(values):
    """Short stable hash of a parameter dict, for template names: arrays by shape and content, functions by
    module (_HASH_MODULE_ALIASES applied) and qualified name. Used by _function_source (analytical specs) and
    _tabulated_data (tables)."""

    def encode(v):
        """JSON-able form of a value: arrays and functions by the identities above, containers recursively."""
        if isinstance(v, np.ndarray):
            v = np.ascontiguousarray(v, dtype=float)
            return f"{v.shape}:" + hashlib.sha1(v.tobytes()).hexdigest()
        if isinstance(v, dict):
            return {str(k): encode(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        if callable(v):
            module = getattr(v, '__module__', '')
            return f"{_HASH_MODULE_ALIASES.get(module, module)}.{getattr(v, '__qualname__', repr(v))}"
        return v
    return hashlib.sha1(json.dumps(encode(values), sort_keys=True, default=str).encode()).hexdigest()[:10]


def _baseline_source(params):
    """'Baseline' spec: a crflux model ('model', default 'H3a') or a steady flux of another kind ('flux'). Called by
    _primary_source, as every builder below: returns (label, factory, times), see there."""
    if 'flux' in params:
        inner = dict(params.pop('flux'))
        if params:
            raise ValueError(f"Baseline: unexpected keys {sorted(params)} next to 'flux'.")
        label, factory, times = _primary_source(inner)
        if times is not None:
            raise ValueError("Baseline: the 'flux' of a baseline must be steady (no time axis).")
        return 'Baseline_' + label, factory, None
    model = params.pop('model', 'H3a')
    if params:
        raise ValueError(f"Baseline: unknown keys {sorted(params)} (use 'model' or 'flux').")
    cls_name, tag = _CRFLUX_BASELINES.get(model, model) if isinstance(model, str) else tuple(model)
    label = 'Baseline' if model == 'H3a' else f"Baseline_{cls_name}_{tag}"

    def factory(t):
        """The crflux model, without its own geomagnetic cutoff (applied later at the site's)."""
        import crflux.models as pm
        return _CrfluxSource(getattr(pm, cls_name)(tag, geomagnetic_cutoff=None))
    return label, factory, None


def _tabulated_source(params):
    """'Tabulated' spec (_tabulated_data); the label carries the name and a hash of the arrays."""
    data = _tabulated_data(params)
    label = f"Tabulated_{params.get('name', 'primary')}_{data['hash']}"
    return label, (lambda t: _TableSource(data, t)), data['time_yr']


def _analytical_source(params):
    """'Analytical' spec: {"function": f, "steady": bool, parameters of f, optional time grid}."""
    function = params.pop('function')
    steady = bool(params.pop('steady', False))
    return _function_source(function, params, getattr(function, '__name__', 'analytical'),
                            None if steady else DEFAULT_TIME_GRID)


def _function_source(function, params, label, grid_default):
    """An analytical flux function with its parameters (checked against its signature) and time grid ('tcut' is
    the former name of t_max_yr); the label gets a hash of the function and of every parameter. Steady if the model
    has no default grid and none is given. Called by _analytical_source and for the registered models."""
    if 'tcut' in params:
        params.setdefault('t_max_yr', params.pop('tcut'))
    grid = {k: params.pop(k) for k in ('t_min_yr', 't_max_yr', 'n_times') if k in params}
    physics = _check_parameters(function, params, label)
    label = f"{label}_{_spec_hash({'function': function, **physics, **grid})}"
    if grid_default is None and not grid:
        return label, (lambda t: _FunctionSource(function, None, physics)), None
    times = _time_grid({**(grid_default or DEFAULT_TIME_GRID), **grid})
    return label, (lambda t: _FunctionSource(function, t, physics)), times


_SPEC_KINDS = {'Baseline': _baseline_source, 'Tabulated': _tabulated_source, 'Analytical': _analytical_source}


def _primary_source(params):
    """The primary flux of a spec (without its site keys), by kind: _SPEC_KINDS, or a model registered with
    register_flux_model. Called by flux.FluxTemplate.get_or_build.

    Args:
        params (dict): The spec; 'kind' defaults to 'Baseline'.

    Returns:
        tuple: (label, factory, times): the label (template name without the site), factory(t) the PrimarySource at
        time t [yr] (None for a steady flux), times the template times [yr] or None.

    Raises:
        ValueError: If the kind is unknown or the spec is invalid.
    """
    params = dict(params)
    kind = params.pop('kind', 'Baseline')
    if kind in _SPEC_KINDS:
        return _SPEC_KINDS[kind](params)
    if kind in _FLUX_MODELS:
        return _function_source(_FLUX_MODELS[kind]['function'], params, kind, _FLUX_MODELS[kind]['time_grid'])
    raise ValueError(f"Unknown template kind {kind!r}: use 'Baseline', 'Tabulated', 'Analytical' or a "
                     f"registered model ({', '.join(repr(k) for k in _FLUX_MODELS)}).")
