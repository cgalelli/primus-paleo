"""Cosmic-ray fluxes at the site of a sample: primary fluxes, the atmospheric secondaries MCEq makes from them
(FluxTemplate) and their history over time (FluxHistory), as read by Paleodetector.

Timeline. FluxHistory works in absolute time [kyr]: 0 is the present, negative values the past (Paleodetector maps
its local time, 0 = start of exposure, onto it). A baseline holds at all times; an event (a supernova, a boost)
adds, from its start_time_kyr on, the flux of a transient template, whose own time axis t_since_yr is the time since
the event in years; a geomagnetic excursion swaps the cutoff of the baseline and events over [start, end).

Specs. A template is described by a spec dict of one of three kinds (see the "Primary fluxes" section):
  * 'Baseline' (default, {} = Hillas-Gaisser H3a): a steady crflux model, or a steady analytical / tabulated flux;
  * an analytical model, a function f(E_gev, t_yr, **params) -> {nucleus: flux} registered under a short name
    (register_flux_model; built in: 'SN', 'Egal'; list_flux_models() shows them), or {'kind': 'Analytical',
    'function': f};
  * 'Tabulated': arrays per nucleus, on their own energy grids and a time axis.
The site, given once to FluxHistory, completes every spec: h_obs_km (altitude, default sea level),
geomagnetic_cutoff [GV] (default 5) and interaction_model (default DEFAULT_INTERACTION_MODEL).

Solvers. 'response' (default, DEFAULT_SOLVER): MCEq is linear in the primary nucleon fluxes, so a ResponseMatrix of
the atmosphere (one MCEq solve per primary nucleon type and energy cell, cached as <template_dir>/Response_*.npz) turns
any primary spectrum into the secondaries by a matrix product. 'direct': one full MCEq solve per spectrum. Below a
few GeV, where MCEq is not meant for nucleons, the nucleon fluxes are continued by fixed shapes normalised to MCEq
(LOW_E_SPLICE, scaled with the cutoff by CUTOFF_SCALING) and the neutrinos by a log-normal (NEUTRINO_LOW_E).

Units. A template stores, per species of MCEQ_SPECIES, the zenith-integrated flux crossing a horizontal plane
[cm^-2 s^-1 GeV^-1] (the zenith quadrature _hemisphere_quadrature has weights summing to pi), and, as 'primary',
the all-nucleus primary flux at the top of the atmosphere after the cutoff [cm^-2 s^-1 sr^-1 GeV^-1].

Caching. Templates are kept in memory for the session; templates of the 'direct' solver (or all of them, with
SAVE_RESPONSE_TEMPLATES) are also saved to <template_dir>/<name>.npz (default config.flux_template_dir()).
"""
import functools
import hashlib
import inspect
import json
import os
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfc, roots_jacobi

from . import config
from ._core import SECONDS_PER_YEAR, fragment_za, log_interp1d

__all__ = ['FluxHistory', 'FluxTemplate', 'MCEQ_SPECIES', 'list_flux_models', 'register_flux_model']

def _hemisphere_quadrature(n_angles):
    """Zenith angles and weights for the flux crossing a horizontal plane, from a zenith-dependent intensity.

        Phi_horizontal = integral I(theta) cos(theta) dOmega = 2 pi integral_0^1 I(mu) mu dmu,   mu = cos(theta)

    Gauss-Jacobi quadrature with weight function mu on [0, 1], so the factor mu is part of the rule (exact for
    polynomials in mu up to degree 2 n_angles - 1; a constant I = 1 gives pi). Summing MCEq's solutions
    [cm^-2 s^-1 sr^-1 GeV^-1] with these weights gives the flux crossing a horizontal plane [cm^-2 s^-1 GeV^-1].
    Used for DEFAULT_ANGLES_DEG / DEFAULT_ANGLE_WEIGHTS (_zenith_averaged_secondaries).

    Args:
        n_angles (int): Number of zenith angles.

    Returns:
        tuple[np.ndarray, np.ndarray]: Zenith angles [deg] and weights (summing to pi).
    """
    x, w_jacobi = roots_jacobi(n_angles, alpha=0, beta=1)
    mu = 0.5 * (x + 1.0)
    weights = 0.5 * np.pi * w_jacobi
    theta_deg = np.rad2deg(np.arccos(mu))
    return theta_deg, weights


DEFAULT_ANGLES_DEG, DEFAULT_ANGLE_WEIGHTS = _hemisphere_quadrature(11)

DEFAULT_INTERACTION_MODEL = "EPOS-LHC-R"


class FluxTemplate:
    """
    A single precomputed (time-since-onset x energy) flux grid for one or
    more particle species, produced by one MCEq configuration.

    A template is either:
      * "steady" (`is_steady=True`): a single energy spectrum, constant in
        time. Used for baselines.
      * "transient": a 2D (t_since_yr, energy_gev) grid representing the
        flux *excess* relative to a baseline, decaying back towards zero
        as t_since_yr grows. Used for discrete events.

    Templates loaded from disk involve no MCEq / crflux import and are
    essentially instantaneous; only `get_or_build` (on a cache miss) pays
    the MCEq cost.
    """

    def __init__(self, name, energy_gev, species_grids, t_since_yr=None, meta=None):
        self.name = name
        self.meta = dict(meta or {})          # site of the template: h_obs_km, geomagnetic_cutoff, interaction_model
        self.energy_gev = np.asarray(energy_gev, dtype=float)
        self.t_since_yr = None if t_since_yr is None else np.asarray(t_since_yr, dtype=float)
        self.is_steady = self.t_since_yr is None

        self._grids = {}
        self._interp = {}
        for species, grid in species_grids.items():
            grid = np.asarray(grid, dtype=float)
            self._grids[species] = grid
            self._interp[species] = self._build_interpolator(grid)

    def _build_interpolator(self, grid):
        """Interpolator of one species: linear in log-log energy (steady: _core.log_interp1d, extrapolating) or in
        (time, log energy) with log flux (transient: constant beyond the time axis, extrapolated in energy)."""
        if self.is_steady:
            return log_interp1d(self.energy_gev, grid)
        log_e = np.log10(self.energy_gev)
        log_flux = np.log10(np.clip(grid, 1e-300, None))
        return RegularGridInterpolator(
            (self.t_since_yr, log_e), log_flux,
            method='linear', bounds_error=False, fill_value=None,
        )

    @property
    def species(self):
        return list(self._grids.keys())

    def flux(self, species, energy_gev, t_since_yr=None):
        """
        Evaluate this template for one species.

        Args:
            species: one of the species stored in this template.
            energy_gev: energies [GeV], array-like.
            t_since_yr: required for transient templates (ignored for
                steady ones); broadcastable against energy_gev.
        """
        energy_gev = np.asarray(energy_gev, dtype=float)
        if species not in self._interp:
            return np.zeros_like(energy_gev)

        if self.is_steady:
            return self._interp[species](energy_gev)
        log_e = np.log10(energy_gev)

        if t_since_yr is None:
            raise ValueError(f"Template '{self.name}' is transient: t_since_yr is required.")

        t = np.asarray(t_since_yr, dtype=float)

        t_b, e_b = np.broadcast_arrays(t, log_e)
        pts = np.stack([t_b.ravel(), e_b.ravel()], axis=-1)

        out = np.power(10.0, self._interp[species](pts)).reshape(t_b.shape)
        t_orig_b = np.broadcast_to(t, t_b.shape)
        return np.where(t_orig_b < 0.0, 0.0, out)

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = dict(
            energy_gev=self.energy_gev,
            is_steady=np.array(self.is_steady),
        )
        if self.t_since_yr is not None:
            payload['t_since_yr'] = self.t_since_yr
        for species, grid in self._grids.items():
            payload[f'species__{species}'] = grid
        for key, value in self.meta.items():
            payload[f'meta__{key}'] = np.asarray('' if value is None else value)
        np.savez_compressed(filepath, **payload)

    @classmethod
    def load(cls, filepath):
        """Load a template previously written with `.save()`. No MCEq needed."""
        with np.load(filepath, allow_pickle=False) as data:
            energy_gev = data['energy_gev']
            key = next((k for k in ('t_since_yr', 't_since_kyr') if k in data), None)     # 't_since_kyr': old files
            t_since_yr = data[key] if key else None
            species_grids = {
                key.split('__', 1)[1]: data[key]
                for key in data.files if key.startswith('species__')
            }
            meta = {key.split('__', 1)[1]: data[key].item() for key in data.files if key.startswith('meta__')}
        meta = {k: (None if v == '' else v) for k, v in meta.items()}
        name = os.path.splitext(os.path.basename(filepath))[0]
        return cls(name, energy_gev, species_grids, t_since_yr=t_since_yr, meta=meta)

    @classmethod
    def get_or_build(cls, params, template_dir=None, force_rebuild=False):
        """
        The template of a spec dict: a 'Baseline' (default, {} = H3a), an analytical model ({"kind": "SN",
        "distance_pc": 100}, see register_flux_model) or a 'Tabulated' flux (see the primary-flux section), plus the
        site keys 'h_obs_km' (default sea level), 'geomagnetic_cutoff' [GV] (default 5.0), 'interaction_model' and
        'solver' ('response' or 'direct', default DEFAULT_SOLVER; not part of the name: both give the same template).
        In a FluxHistory, give the site once to the history instead.

        Caching: templates built with the response matrix are rebuilt from it on demand (a fraction of a second)
        and kept in memory for the session, unless SAVE_RESPONSE_TEMPLATES is True; only the matrices
        (Response_*.npz) live on disk. 'direct' templates are saved to and loaded from <template_dir>/<name>.npz.
        """
        template_dir = config.flux_template_dir() if template_dir is None else template_dir
        params = dict(params)
        solver = params.pop("solver", None) or DEFAULT_SOLVER
        on_disk = solver == "direct" or SAVE_RESPONSE_TEMPLATES
        geomagnetic_cutoff = params.pop("geomagnetic_cutoff", 5.0)
        h_obs_km = params.pop("h_obs_km", None)
        interaction_model = params.pop("interaction_model", DEFAULT_INTERACTION_MODEL)

        label, factory, times = _primary_source(params)
        name = f"{label}_{geomagnetic_cutoff}GV" + (f"_{h_obs_km}km" if h_obs_km else "")

        path = os.path.join(template_dir, name + ".npz")
        memory_key = (os.path.abspath(template_dir), name, interaction_model)
        if not force_rebuild:
            if memory_key in _TEMPLATE_CACHE:
                return _TEMPLATE_CACHE[memory_key]
            if on_disk and os.path.exists(path):
                return cls.load(path)

        template = _build_template(name, factory, t_since_yr=times, interaction_model=interaction_model,
                                   h_obs_km=h_obs_km, geomagnetic_cutoff=geomagnetic_cutoff, solver=solver,
                                   template_dir=template_dir)
        if on_disk:
            template.save(path)
        _TEMPLATE_CACHE[memory_key] = template
        return template

    @classmethod
    def from_primary_spectrum(cls, energy_gev, flux, composition=None, t_since_yr=None, name="primary",
                              template_dir=None, force_rebuild=False, **site):
        """Shorthand for get_or_build(primary_spectrum_spec(...)): a 'Tabulated' template from one energy grid."""
        spec = primary_spectrum_spec(energy_gev, flux, composition=composition, t_since_yr=t_since_yr, name=name)
        return cls.get_or_build({**spec, **site}, template_dir=template_dir, force_rebuild=force_rebuild)


SITE_KEYS = ("h_obs_km", "geomagnetic_cutoff", "interaction_model")


class FluxHistory:
    """
    Baseline template + a list of discrete events (plain dicts), summed on
    demand. Structurally the flux-side analogue of `overburden_history`:
    a small spec goes in, one object answers every downstream query.

    Timeline: 0 = present, negative = past, in kyr (see the module docstring).
    """

    def __init__(self, baseline=None, events=None, excursions=None, template_dir=None, h_obs_km=None,
                 geomagnetic_cutoff=None, interaction_model=None):
        """
        Args:
            baseline: a params dict (built on demand; None or {} = H3a, see FluxTemplate.get_or_build), a
                FluxTemplate or a cached template name (str).
            events: list of dicts, each with 'start_time_kyr' and 'template' (FluxTemplate / name / params
                dict), and optionally 'weight' / 'label'.
            excursions: list of dicts, each with 'start_time_kyr',
                'end_time_kyr', 'geomagnetic_cutoff', and optionally
                'label' -- see `add_excursion`.
            template_dir: directory used to load/save templates. Defaults to None: <data root>/flux_data
                (config.flux_template_dir()).
            h_obs_km, geomagnetic_cutoff, interaction_model: the site, given once and applied to the baseline,
                every event and the excursion variants. One not given here is taken from the baseline spec if it
                has it (older notebooks), else the get_or_build default. An entry with a different value raises.
        """
        self.template_dir = config.flux_template_dir() if template_dir is None else template_dir
        baseline = {} if baseline is None else baseline
        site = {k: v for k, v in (("h_obs_km", h_obs_km), ("geomagnetic_cutoff", geomagnetic_cutoff),
                                  ("interaction_model", interaction_model)) if v is not None}
        if isinstance(baseline, dict):
            site = {**{k: baseline[k] for k in SITE_KEYS if k in baseline}, **site}
        self.site = site
        self._baseline_spec = self._with_site(baseline, "baseline")
        self._baseline_cutoff_variants = {}
        self.baseline = self._resolve(self._baseline_spec)
        self.events = []
        for event in (events or []):
            missing = [k for k in ("start_time_kyr", "template") if k not in event]
            if missing:
                raise KeyError(f"flux event {event!r} has no {missing}.")
            self.add_event(event["start_time_kyr"], event["template"], weight=event.get("weight", 1.0),
                           label=event.get("label"))
        self.excursions = []
        for excursion in (excursions or []):
            self.add_excursion(
                excursion["start_time_kyr"], excursion["end_time_kyr"],
                excursion["geomagnetic_cutoff"], label=excursion.get("label"),
            )

    def _with_site(self, spec, label):
        """The spec with the site of the history; raises if it states a different site."""
        if isinstance(spec, FluxTemplate):
            self._check_site(spec, label)
            return spec
        if not isinstance(spec, dict):
            return spec
        out = dict(spec)
        for key, value in self.site.items():
            if key in out and out[key] != value:
                raise ValueError(f"{label}: {key} = {out[key]!r} differs from the flux history's {value!r}; give "
                                 "the site once, to FluxHistory.")
            out[key] = value
        return out

    def _check_site(self, template, label):
        for key, value in self.site.items():
            if key in template.meta and template.meta[key] != value:
                raise ValueError(f"{label}: template {template.name!r} was built with {key} = "
                                 f"{template.meta[key]!r}, the flux history has {value!r}.")

    def _resolve(self, template):
        if isinstance(template, FluxTemplate):
            return template
        if isinstance(template, dict):
            return FluxTemplate.get_or_build(template, self.template_dir)
        if isinstance(template, str):
            path = os.path.join(self.template_dir, f"{template}.npz")
            if os.path.exists(path):
                loaded = FluxTemplate.load(path)
                self._check_site(loaded, template)
                return loaded
            raise FileNotFoundError(
                f"No cached template named '{template}' in {self.template_dir}. "
                f"Pass a params dict (e.g. {{'kind': 'SN', 'distance_pc': ...}}) "
                f"instead of a bare name if you want it built on demand."
            )
        raise TypeError(f"Unsupported template spec: {template!r}")

    def add_event(self, start_time_kyr, template, weight=1.0, label=None):
        """
        Add one more flux event. `template` may be an already-loaded
        FluxTemplate, the name of an already-cached one, or a params dict
        (e.g. `{"kind": "SN", "distance_pc": 200}`) -- in the last case
        the template is built via MCEq on first use and simply loaded
        from disk every time after that. Multiple events (even the same
        template at the same or nearby times, e.g. two supernovae close
        together) are superposed automatically.
        """
        template = self._with_site(template, label or f"event at {start_time_kyr} kyr")
        resolved = self._resolve(template)
        self.events.append({
            "template": resolved, "spec": template, "start_time_kyr": float(start_time_kyr),
            "weight": weight, "label": label, "_cutoff_variants": {},
        })
        return resolved

    def add_excursion(self, start_time_kyr, end_time_kyr, geomagnetic_cutoff, label=None):
        """
        Register a geomagnetic excursion: a transient rigidity-cutoff
        change, not a new physical flux source. Over the absolute-time
        window [start_time_kyr, end_time_kyr) (same convention as
        everywhere else: 0 = present, negative = past), every live
        baseline/event's rigidity cutoff is swapped from whatever it was
        built with to `geomagnetic_cutoff` (GV); outside the window it
        reverts to normal. This modulates the *existing* flux history
        (baseline + whichever events are active at that time) rather
        than adding a signal of its own.

        Implementation: MCEq has no notion of a time-dependent cutoff
        within a single run, so rather than varying geomagnetic_cutoff
        continuously, each affected baseline/event grows a second, fully
        independent template built at `geomagnetic_cutoff` -- cached
        under its own name by `FluxTemplate.get_or_build` (which already
        keys its cache filename off `geomagnetic_cutoff`, so this needs
        no new caching logic). `flux()` then simply picks, per time
        sample, whether to read from the normal-cutoff stack or the
        excursion-cutoff stack.

        Only a baseline/event added as a params dict (i.e. it still has
        a 'kind' to rebuild from) can grow one of these variants; one
        added as an already-built FluxTemplate or a bare cached name has
        no recoverable geomagnetic_cutoff to vary, and only raises when
        a query actually lands inside this excursion's window (not now).

        Multiple excursions may be registered; where two overlap in
        time, the one added *later* wins for the overlapping samples.
        """
        if end_time_kyr <= start_time_kyr:
            raise ValueError(
                f"Excursion end_time_kyr ({end_time_kyr}) must be after "
                f"start_time_kyr ({start_time_kyr})."
            )
        excursion = {
            "start_time_kyr": float(start_time_kyr),
            "end_time_kyr": float(end_time_kyr),
            "geomagnetic_cutoff": float(geomagnetic_cutoff),
            "label": label,
        }
        self.excursions.append(excursion)
        return excursion

    def _cutoff_variant(self, spec, cache, geomagnetic_cutoff):
        """
        Look up (building/loading via MCEq if needed) the version of a
        baseline/event's template at a different `geomagnetic_cutoff`,
        with the rest of its physical params (kind, distance, etc.)
        unchanged. `cache` is the per-baseline/event dict that memoizes
        this so repeated queries at the same excursion cutoff don't
        touch disk twice.
        """
        if geomagnetic_cutoff in cache:
            return cache[geomagnetic_cutoff]
        if not isinstance(spec, dict):
            raise TypeError(
                "A geomagnetic excursion needs a params dict to rebuild "
                f"this baseline/event at a different cutoff, but it was "
                f"added as a {type(spec).__name__}. Add it as e.g. "
                "{'kind': 'Baseline'} instead of a pre-built FluxTemplate "
                "or a bare cached name if it should respond to excursions."
            )
        variant_spec = dict(spec)
        variant_spec["geomagnetic_cutoff"] = geomagnetic_cutoff
        variant = FluxTemplate.get_or_build(variant_spec, self.template_dir)
        cache[geomagnetic_cutoff] = variant
        return variant

    def _flux_stack(self, species, energy_gev, t_kyr, geomagnetic_cutoff=None):
        """
        Baseline + all events summed, at absolute time(s) `t_kyr`, using
        either the normal (`geomagnetic_cutoff=None`) or an
        excursion-swapped template for each.
        """
        if geomagnetic_cutoff is None:
            baseline = self.baseline
        else:
            baseline = self._cutoff_variant(self._baseline_spec, self._baseline_cutoff_variants, geomagnetic_cutoff)

        total = baseline.flux(species, energy_gev)
        total = np.broadcast_to(total, np.broadcast_shapes(np.shape(t_kyr), np.shape(energy_gev))).copy()
        for event in self.events:
            if geomagnetic_cutoff is None:
                template = event["template"]
            else:
                template = self._cutoff_variant(event["spec"], event["_cutoff_variants"], geomagnetic_cutoff)
            t_since_yr = (np.asarray(t_kyr, dtype=float) - event["start_time_kyr"]) * 1.e3
            total = total + event["weight"] * template.flux(species, energy_gev, t_since_yr=t_since_yr)
        return total

    def flux(self, species, energy_gev, t_kyr):
        """
        Total flux (baseline + all active events) for one species, at
        absolute time(s) `t_kyr` (0 = present, negative = past -- see
        module docstring), with any registered geomagnetic excursions
        (see `add_excursion`) swapped in over their windows.
        """
        total = self._flux_stack(species, energy_gev, t_kyr)

        if not self.excursions:
            return total

        t_arr = np.asarray(t_kyr, dtype=float)
        shape = np.broadcast_shapes(t_arr.shape, np.shape(energy_gev))
        total = np.broadcast_to(total, shape).copy()
        t_b = np.broadcast_to(t_arr, shape)

        for excursion in self.excursions:
            mask = (t_b >= excursion["start_time_kyr"]) & (t_b < excursion["end_time_kyr"])
            if not np.any(mask):
                continue
            excursion_total = self._flux_stack(
                species, energy_gev, t_kyr, geomagnetic_cutoff=excursion["geomagnetic_cutoff"]
            )
            excursion_total = np.broadcast_to(excursion_total, shape)
            total = np.where(mask, excursion_total, total)
        return total

    def get_map(self, species, t_grid_kyr, e_grid_gev=None):
        """
        Returns `(t_grid_kyr, e_grid_gev, flux_map)`, `flux_map.shape ==
        (len(t_grid_kyr), len(e_grid_gev))`, ready for
        `plt.pcolormesh(e_grid, t_grid, flux_map.T)`.
        """
        if e_grid_gev is None:
            e_grid_gev = self.baseline.energy_gev
        t_grid_kyr = np.asarray(t_grid_kyr, dtype=float)
        e_grid_gev = np.asarray(e_grid_gev, dtype=float)
        tt, ee = np.meshgrid(t_grid_kyr, e_grid_gev, indexing='ij')
        return t_grid_kyr, e_grid_gev, self.flux(species, ee, tt)

try:
    from crflux.models import PrimaryFlux
except ImportError:
    PrimaryFlux = object
try:                                    # crflux >= 2.3: Fe group id 5426 -> 5626 (A = 56)
    from crflux.models import LEGACY_IDS as _CRFLUX_LEGACY_IDS
except ImportError:
    _CRFLUX_LEGACY_IDS = {}


# =====================================================================================================================
# Primary fluxes at the top of the atmosphere
# =====================================================================================================================
# Three kinds of template spec (FluxTemplate.get_or_build, FluxHistory):
#
#   "Baseline"     steady background: a crflux model, by default Hillas-Gaisser H3a ({} means this), or a steady
#                  Analytical / Tabulated flux:   {"kind": "Baseline", "model": "H4a"},
#                  {"kind": "Baseline", "model": ("GlobalSplineFitBeta", None)}, {"kind": "Baseline", "flux": {...}}
#   Analytical     a function f(E_gev, t_yr, **params) -> {nucleus: flux}, registered under a short name
#                  (register_flux_model) and used as {"kind": "SN", "distance_pc": 100}, or passed directly as
#                  {"kind": "Analytical", "function": f, ...}. Built in: "SN", "Egal"; list_flux_models()
#                  shows every registered model with its parameters.
#   "Tabulated"    arrays, one entry per nucleus (see _tabulated_data):
#                  {"kind": "Tabulated", "name": ..., "time_yr": t[n_time],
#                   "H": {"energy_gev": E[n_energy], "flux_cm2_s_sr_gev": F[n_time, n_energy]}, "He": {...}}
#
# Every flux is per nucleus, per GeV of total energy per nucleus, in cm^-2 s^-1 sr^-1, at the top of the
# atmosphere and before the geomagnetic cutoff, which is applied afterwards (crflux nucleus_flux) at the site's
# cutoff. Nuclei are named by mass group ('H', 'He', 'CNO', 'MgAlSi', 'Fe'), by nucleus ('O16', 'Fe56') or by CORSIKA
# id; MCEq splits each into its nucleons (superposition). Transient fluxes (events) are the excess over the baseline.
# Time grids are fixed per model (log-spaced; keys t_min_yr, t_max_yr, n_times; 'tcut' = t_max_yr): the detector
# refines its own time grid around the template times.

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
        'description' (first sentence of the docstring), 'doc' (the whole docstring)}}.
    """
    out = {}
    for name, entry in _FLUX_MODELS.items():
        f = entry['function']
        text = ' '.join((inspect.getdoc(f) or '').split('\n\n')[0].split())
        first = re.split(r'(?<=[a-z0-9)])\. |: ', text, maxsplit=1)[0]          # first sentence or clause
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
    """An analytical flux function (see register_flux_model) at one time. Called once per evaluation, with the
    energies of every nucleus concatenated."""

    def __init__(self, function, t_yr, params):
        self.function, self.t_yr, self.params = function, t_yr, dict(params)
        probe = function(np.array([10.0]), t_yr, **self.params)
        self._names = {nucleus_id(k): k for k in probe}
        self.nucleus_ids = sorted(self._names)
        self.name = getattr(function, '__name__', 'flux model')

    def fluxes(self, energies):
        ids = [cid for cid in self.nucleus_ids if cid in energies]
        arrays = [np.atleast_1d(np.asarray(energies[cid], dtype=float)) for cid in ids]
        edges = np.cumsum([0] + [a.size for a in arrays])
        values = self.function(np.concatenate(arrays), self.t_yr, **self.params)
        out = {}
        for k, cid in enumerate(ids):
            v = np.broadcast_to(np.asarray(values[self._names[cid]], dtype=float), (edges[-1],))
            out[cid] = v[edges[k]:edges[k + 1]] * 1e4                              # cm^-2 -> m^-2 (crflux units)
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
    if 'flux' in spec and 'energy_gev' in spec:                         # older single-grid form
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
            raise ValueError(f"Tabulated: {name!r} should be a dict like {{'energy_gev': ..., 'flux_cm2_s_sr_gev': ...}}.")
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
    """A 'Tabulated' flux at one of its times (log-log in energy, 0 outside the table)."""

    def __init__(self, data, t_yr=None):
        row = 0 if t_yr is None or data['time_yr'] is None else int(np.argmin(np.abs(data['time_yr'] - t_yr)))
        self._tables = {cid: (np.log(E), np.log(np.clip(F[row] * 1e4, 1e-300, None)))     # crflux units, per m^2
                        for cid, (E, F) in data['nuclei'].items()}
        self.nucleus_ids = sorted(self._tables)
        self.name = 'Tabulated'

    def fluxes(self, energies):
        out = {}
        for cid, E in energies.items():
            E = np.atleast_1d(np.asarray(E, dtype=float))
            log_e, log_f = self._tables[cid]
            out[cid] = np.exp(np.interp(np.log(E), log_e, log_f, left=-np.inf, right=-np.inf))
        return out


# --- From a spec to a primary model -----------------------------------------------------------------------------------
def _spec_hash(values):
    """Short stable hash of a parameter dict, for template names: arrays by shape and content, functions by
    module and qualified name. Used by _primary_source (analytical specs) and _tabulated_data (tables)."""

    def encode(v):
        if isinstance(v, np.ndarray):
            v = np.ascontiguousarray(v, dtype=float)
            return f"{v.shape}:" + hashlib.sha1(v.tobytes()).hexdigest()
        if isinstance(v, dict):
            return {str(k): encode(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [encode(x) for x in v]
        if callable(v):
            return f"{getattr(v, '__module__', '')}.{getattr(v, '__qualname__', repr(v))}"
        return v
    return hashlib.sha1(json.dumps(encode(values), sort_keys=True, default=str).encode()).hexdigest()[:10]


def _primary_source(params):
    """(label, factory, times) of a spec without its site keys: factory(t) gives the PrimarySource at time t [yr]
    (None for a steady flux), times the template times [yr] or None."""
    params = dict(params)
    kind = params.pop('kind', 'Baseline')

    if kind == 'Baseline':
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
            import crflux.models as pm
            return _CrfluxSource(getattr(pm, cls_name)(tag, geomagnetic_cutoff=None))
        return label, factory, None

    if kind == 'Tabulated':
        data = _tabulated_data(params)
        label = f"Tabulated_{params.get('name', 'primary')}_{data['hash']}"
        return label, (lambda t: _TableSource(data, t)), data['time_yr']

    if kind == 'Analytical':
        function = params.pop('function')
        steady = bool(params.pop('steady', False))
        grid_default = None if steady else DEFAULT_TIME_GRID
        label = getattr(function, '__name__', 'analytical')
    elif kind in _FLUX_MODELS:
        function = _FLUX_MODELS[kind]['function']
        grid_default = _FLUX_MODELS[kind]['time_grid']
        label = kind
    else:
        raise ValueError(f"Unknown template kind {kind!r}: use 'Baseline', 'Tabulated', 'Analytical' or a "
                         f"registered model ({', '.join(repr(k) for k in _FLUX_MODELS)}).")
    if 'tcut' in params:                                                   # former name of t_max_yr
        params.setdefault('t_max_yr', params.pop('tcut'))
    grid = {k: params.pop(k) for k in ('t_min_yr', 't_max_yr', 'n_times') if k in params}
    physics = _check_parameters(function, params, label)
    label = f"{label}_{_spec_hash({'function': function, **physics, **grid})}"
    if grid_default is None and not grid:
        return label, (lambda t: _FunctionSource(function, None, physics)), None
    times = _time_grid({**(grid_default or DEFAULT_TIME_GRID), **grid})
    return label, (lambda t: _FunctionSource(function, t, physics)), times


MCEQ_SPECIES = {
    'mu+': ('total_mu+',),          # the trailing comma makes these tuples: ('x') is just the string 'x'
    'mu-': ('total_mu-',),
    'neutron': ('n0',),
    'proton': ('p+',),
    'nu_mu': ('total_numu',),
    'anti_nu_mu': ('total_antinumu',),
    'nu_e': ('total_nue',),
    'anti_nu_e': ('total_antinue',),
}


def _zenith_averaged_secondaries(mceq_run, angles=DEFAULT_ANGLES_DEG,
                                  weights=DEFAULT_ANGLE_WEIGHTS):
    """
    Zenith-averaged flux of every MCEQ_SPECIES entry, in MCEq's raw units, at `mceq_run`'s *current*
    configuration (interaction model, primary model/params, density model), using one `solve_fullsky` batch
    call and the solid-angle quadrature `weights` (see `_hemisphere_quadrature`).

    Returns:
        dict: species -> flux on mceq_run.e_grid (un-rescaled, un-clipped).
    """
    result = mceq_run.solve_fullsky(zenith_grid=angles)
    out = {key: np.zeros_like(mceq_run.e_grid) for key in MCEQ_SPECIES}
    for theta, w in zip(angles, weights):
        for key, names in MCEQ_SPECIES.items():
            for name in names:
                out[key] += w * result.get_solution(name, zenith=float(theta), mag=0)
    return out


# --- Low-energy nucleons ------------------------------------------------------------------------------------------
# MCEq is not meant for nucleons below a few GeV and its grid stops at e_grid[0]. Below ~1 GeV each nucleon flux is a
# fixed shape normalised to MCEq around e_match, so it follows every change of the MCEq solution (primary flux,
# events, cutoff) as a rescaling. Above e_match the MCEq solution is kept.
def gordon_neutron_spectrum(energy_gev, evaporation=True):
    """Sea-level neutron spectrum of Gordon et al., IEEE TNS 51 (2004) 3427 (fit adopted in JESD89A).

    New York City, outdoors, sea level: 12.7 n cm^-2 h^-1 above 10 MeV, 19.6 above 1 MeV. Two log-normal terms, the
    cascade peak (~90 MeV) and the evaporation peak (~1 MeV); fluence rate (all directions, ground albedo included),
    valid up to ~1 GeV (above, the log-normal tail falls too steeply).

    Args:
        energy_gev (array_like): Kinetic energy [GeV].
        evaporation (bool, optional): Include the evaporation term. Defaults to True.

    Returns:
        np.ndarray: dPhi/dE [n cm^-2 s^-1 GeV^-1].
    """
    ln_e = np.log(np.asarray(energy_gev, dtype=float) * 1e3)          # the fit uses E in MeV
    phi = 1.006e-6 * np.exp(-0.35 * ln_e ** 2 + 2.1451 * ln_e)
    if evaporation:
        phi = phi + 1.011e-3 * np.exp(-0.4106 * ln_e ** 2 - 0.667 * ln_e)
    return phi * 1e3                                                 # per MeV -> per GeV


def proton_equilibrium_shape(energy_gev, mean_excitation_ev=85.7):
    """Shape of the proton flux in ionisation equilibrium, dPhi/dE = Q(E) / S(E).

    A charged particle produced at rate q(E') per gram and slowed by ionisation has, in equilibrium,
    dPhi/dE = (1 / S(E)) * Q(E), with Q(E) = integral_E^inf q(E') dE' the production of protons above E. Above a few
    tens of MeV protons and neutrons come out of the same nuclear cascades in about equal numbers, so q is taken with
    the shape of the cascade term of the sea-level neutron spectrum (gordon_neutron_spectrum, which in equilibrium
    is proportional to the neutron production): Q is then an erfc in ln E. S is the Bethe stopping power (air,
    I = 85.7 eV, no shell or density corrections). Nuclear absorption of the protons and evaporation protons (low
    energy, Coulomb barrier) are neglected: the shape is meant below the matching energy only.

    Result: dPhi/dE ~ E^0.75 below ~10 MeV, maximum near 100 MeV, slope ~ -1.8 at 1 GeV; with the normalisation at
    1 GeV taken from MCEq, p/n ~ 1 % at 10 MeV, ~15 % at 100 MeV, ~60 % at 300 MeV (for p/n = 2 at 1 GeV).

    Args:
        energy_gev (array_like): Kinetic energy [GeV].
        mean_excitation_ev (float, optional): Mean excitation energy of the medium [eV]. Defaults to 85.7 (air).

    Returns:
        np.ndarray: Unnormalised shape (arbitrary units).
    """
    e = np.asarray(energy_gev, dtype=float)
    m_p, m_e = 0.93827, 0.51100e-3                                    # GeV
    gamma = 1.0 + e / m_p
    beta2 = 1.0 - 1.0 / gamma ** 2
    log_term = np.log(2.0 * m_e * 1e9 * beta2 * gamma ** 2 / mean_excitation_ev) - beta2
    inv_stopping = beta2 / np.clip(log_term, 1e-3, None)             # 1 / S up to a constant
    b, c = 0.35, 3.1451                                              # cascade term of the Gordon fit, in u = ln(E/MeV)
    production_above = erfc(np.sqrt(b) * (np.log(e * 1e3) - c / (2.0 * b)))   # Q(E) up to a constant
    return inv_stopping * production_above


# Template species -> low-energy shape and join. Between e_bridge and e_match the shape is continued by a power law
# E^-gamma (the shapes reach slope -gamma at e_bridge: Gordon at 1 GeV, the proton form at ~3 GeV), so that MCEq is
# matched at e_match, above the few-GeV region where its nucleon solution is not reliable.
LOW_E_SPLICE = {
    'neutron': {'shape': gordon_neutron_spectrum, 'e_bridge_gev': 1.0, 'gamma': 2.7,
                'e_match_gev': 7.0, 'blend_dex': 0.25},
    'proton': {'shape': proton_equilibrium_shape, 'e_bridge_gev': 3.0, 'gamma': 2.7,
               'e_match_gev': 7.0, 'blend_dex': 0.25},
}
# Neutrinos below e_match: MCEq is continued by a log-normal in E peaking at E_peak,
#     Phi(E) = Phi(E_m) * exp(-a [ln^2(E / E_peak) - ln^2(E_m / E_peak)]),   a = -slope / (2 ln(E_m / E_peak)),
# with slope the local MCEq log-slope around E_m, so flux and slope are continuous at E_m and the extension follows
# any change of the MCEq solution. The atmospheric neutrino spectrum peaks at a few tens of MeV, the scale of the
# decay energies at rest (29.8 MeV for pi -> mu nu, 52.8 MeV Michel endpoint), and falls below: the FLUKA calculation
# (Battistoni et al., Astropart. Phys. 23 (2005) 526) peaks at ~40 MeV for both flavours, with dN/dE ~ E^0.6 between
# 13 and 40 MeV and a peak ~3x the flux at 0.2 GeV; this form gives ~E^0.7 and ~2x for a slope of -1 at 0.2 GeV.
# Indicative only (no separate decay-at-rest Michel peak, no geomagnetic effects); use FLUKA for accurate fluxes.
NEUTRINO_LOW_E = {
    'species': ('nu_mu', 'anti_nu_mu', 'nu_e', 'anti_nu_e'),
    'e_match_gev': 0.2,      # just above the bottom of the MCEq grid
    'fit_dex': 0.15,         # half-width of the window where the MCEq slope is measured [decades]
    'e_peak_gev': 0.04,      # peak of dN/dE (FLUKA: ~40 MeV)
}
LOW_E_GRID = {'e_min_gev': 1e-4, 'n_per_decade': 10}   # template grid extended down to the lowest Geant4 energy
# Geomagnetic-cutoff scaling of the low-energy splice. The LOW_E_SPLICE shapes are normalised to MCEq at e_match
# (~7 GeV), where the cutoff has almost no effect, so on their own they do not respond to the cutoff. Their
# low-energy part is therefore multiplied by F(Rc, x) / F(Rc_ref, x), the nucleon-flux scaling for spallation of
# Desilets, Zreda & Prabu 2006 (EPSL 246, 265), Eqs. (5)-(7), at the observation depth x:
#     F(Rc, x) = f(Rc) f(x; Rc),
#     f(Rc)    = 1 - exp(-alpha Rc^-k)                       (Dorman function, sea level; f(0) = 1),
#     f(x; Rc) = exp[G(1033; Rc) - G(x; Rc)],                 G = integral of 1/Lambda(x; Rc) dx, with
#     1/Lambda = n / (1 + exp(-a Rc^-kappa)) + (a0 + a1 Rc + a2 Rc^2) x + (a3 + ...) x^2 + (a6 + ...) x^3.
# Table 2 coefficients; valid for x = 500-1033 g/cm^2 (0-5700 m) and Rc = 0-17.3 GV. The attenuation length grows
# with Rc, so the latitude effect grows with altitude. Rc_ref is the cutoff the uncorrected splice stands for: 5 GV,
# the default baseline cutoff, so templates built at 5 GV are unchanged. Set 'species' to () to switch it off.
CUTOFF_SCALING = {
    'species': ('neutron', 'proton'),
    'alpha': 10.275,
    'k': 0.9615,
    'reference_cutoff_gv': 5.0,
    'sea_level_depth_g_cm2': 1033.,
    'attenuation': {
        'n': 1.0177e-02, 'a': 1.0207e-01, 'kappa': -3.9527e-01,
        'a_coeffs': (8.5236e-06, -6.3670e-07, -7.0814e-09,
                     -9.9182e-09, 9.9250e-10, 2.4925e-11,
                     3.8615e-12, -4.8194e-13, -1.5371e-14),
    },
}


def atmospheric_depth_g_cm2(h_obs_km):
    """
    Vertical atmospheric depth [g/cm^2] at altitude `h_obs_km` (ICAO standard atmosphere, troposphere), with
    sea level at CUTOFF_SCALING['sea_level_depth_g_cm2'].

    Args:
        h_obs_km (float or None): Altitude [km]; None means sea level.

    Returns:
        float: Atmospheric depth [g/cm^2].
    """
    h_m = 0. if h_obs_km is None else 1.e3 * float(h_obs_km)
    return CUTOFF_SCALING['sea_level_depth_g_cm2'] * (1. - 2.25577e-5 * h_m) ** 5.25588


def nucleon_flux_scaling(geomagnetic_cutoff, depth_g_cm2=None):
    """
    Spallation-nucleon flux relative to sea level, high latitude: F(Rc, x) = f(Rc) f(x; Rc) of Desilets, Zreda &
    Prabu 2006, Eqs. (5)-(7) (see CUTOFF_SCALING).

    Args:
        geomagnetic_cutoff (float or None): Vertical cutoff rigidity [GV]; None or <= 0 means no cutoff.
        depth_g_cm2 (float, optional): Atmospheric depth [g/cm^2]. Defaults to None: sea level.

    Returns:
        float: F(Rc, x).
    """
    c = CUTOFF_SCALING
    rc = 0. if geomagnetic_cutoff is None else max(float(geomagnetic_cutoff), 0.)
    x0 = c['sea_level_depth_g_cm2']
    x = x0 if depth_g_cm2 is None else float(depth_g_cm2)

    f_rc = 1. if rc == 0. else 1. - np.exp(-c['alpha'] * rc ** (-c['k']))

    att = c['attenuation']
    a0, a1, a2, a3, a4, a5, a6, a7, a8 = att['a_coeffs']
    b0 = att['n'] / (1. + np.exp(-att['a'] * rc ** (-att['kappa'])))
    b1 = a0 + a1 * rc + a2 * rc ** 2
    b2 = a3 + a4 * rc + a5 * rc ** 2
    b3 = a6 + a7 * rc + a8 * rc ** 2

    def G(xx):
        return b0 * xx + b1 * xx ** 2 / 2. + b2 * xx ** 3 / 3. + b3 * xx ** 4 / 4.

    return f_rc * np.exp(G(x0) - G(x))


def cutoff_scaling_factor(geomagnetic_cutoff, h_obs_km=None, reference_cutoff=None):
    """
    F(Rc, x) / F(Rc_ref, x) at the observation altitude (see CUTOFF_SCALING): the factor applied to the
    low-energy nucleon splice of a template built at `geomagnetic_cutoff`.

    Args:
        geomagnetic_cutoff (float or None): Vertical cutoff rigidity [GV]; None or <= 0 means no cutoff.
        h_obs_km (float, optional): Observation altitude [km]. Defaults to None: sea level.
        reference_cutoff (float, optional): Rc_ref [GV]. Defaults to CUTOFF_SCALING['reference_cutoff_gv'].

    Returns:
        float: The scaling factor (> 1 below Rc_ref, < 1 above).
    """
    reference_cutoff = CUTOFF_SCALING['reference_cutoff_gv'] if reference_cutoff is None else reference_cutoff
    x = atmospheric_depth_g_cm2(h_obs_km)
    return nucleon_flux_scaling(geomagnetic_cutoff, x) / nucleon_flux_scaling(reference_cutoff, x)


def _postprocess_secondaries(e_grid, raw_fluxes, total_flux, geomagnetic_cutoff=None, h_obs_km=None):
    """
    Zenith-averaged MCEq fluxes -> template fluxes: the MCEq grid extended down to LOW_E_GRID['e_min_gev'], with the
    low-energy correction of every LOW_E_SPLICE species. Shared by both solvers (direct MCEq solve and response
    matrix), applied to every template (steady and, per time step, transient).

    For each spliced species the shape is continued above e_bridge by shape(e_bridge) (E / e_bridge)^-gamma;
    k = geometric mean of MCEq / shape over e_match * 10^(+-blend_dex), then a log-space tanh join,
        log Phi = s log Phi_MCEq + (1 - s) log(k shape),   s = (1 + tanh(log10(E / e_match) / blend_dex)) / 2
    so flux and slope are continuous. Muons and the primary are only extended as FluxTemplate would have
    extrapolated them (unchanged where they were queried before); neutrinos are continued below
    NEUTRINO_LOW_E['e_match_gev'] by a log-normal peaking at e_peak_gev. Fluxes are floored at 1e-44 for
    FluxTemplate's log-space interpolator. For the species in CUTOFF_SCALING['species'] the low-energy shape is
    multiplied by cutoff_scaling_factor(geomagnetic_cutoff, h_obs_km), so the splice follows the cutoff (MCEq above
    e_match is untouched). These steps are non-linear (log blend), so they come after the MCEq solution, never
    inside the response matrix.

    Args:
        e_grid (np.ndarray): MCEq energy grid [GeV].
        raw_fluxes (dict): species -> zenith-averaged MCEq flux on e_grid (`_zenith_averaged_secondaries` or
            `ResponseMatrix.apply`).
        total_flux (callable): E per nucleus [GeV] -> all-nucleus primary flux after the cutoff (the 'primary'
            species).
        geomagnetic_cutoff (float, optional): Cutoff [GV] the primary model was built with. Defaults to None:
            no cutoff scaling (factor 1).
        h_obs_km (float, optional): Observation altitude [km] for the cutoff scaling. Defaults to None: sea level.

    Returns:
        tuple: (e_ext, fluxes, primary): extended grid, dict species -> flux on e_ext, primary flux on e_ext.

    Raises:
        ValueError: If fewer than 2 MCEq grid points lie in a matching window.
    """
    e_grid = np.asarray(e_grid, dtype=float)
    n_low = max(int(np.ceil(np.log10(e_grid[0] / LOW_E_GRID['e_min_gev']) * LOW_E_GRID['n_per_decade'])), 0)
    e_ext = np.concatenate([np.logspace(np.log10(LOW_E_GRID['e_min_gev']), np.log10(e_grid[0]), n_low + 1)[:-1],
                            e_grid])

    fluxes = {}
    for key, flux in raw_fluxes.items():
        flux = np.clip(flux, 1e-300, None)                           # a decayed transient may be exactly 0
        mceq_ext = log_interp1d(e_grid, flux)(e_ext)
        if key in NEUTRINO_LOW_E['species']:
            c = NEUTRINO_LOW_E
            em, ep = c['e_match_gev'], c['e_peak_gev']
            win = np.abs(np.log10(e_grid / em)) <= c['fit_dex']
            if win.sum() < 2:
                raise ValueError(f"{key}: only {win.sum()} MCEq grid points around {em} GeV; "
                                 "change NEUTRINO_LOW_E e_match_gev or fit_dex.")
            slope = np.polyfit(np.log(e_grid[win]), np.log(flux[win]), 1)[0]
            a = max(-slope, 0.0) / (2.0 * np.log(em / ep))           # same local slope at E_m
            low = log_interp1d(e_grid, flux)(em) * np.exp(-a * (np.log(e_ext / ep) ** 2 - np.log(em / ep) ** 2))
            mceq_ext = np.where(e_ext < em, low, mceq_ext)
        if key in LOW_E_SPLICE:
            c = LOW_E_SPLICE[key]
            eb = c['e_bridge_gev']
            shape = lambda e: np.where(e <= eb, c['shape'](np.minimum(e, eb)), c['shape'](eb) * (e / eb) ** -c['gamma'])
            win = np.abs(np.log10(e_grid / c['e_match_gev'])) <= c['blend_dex']
            if win.sum() < 2:
                raise ValueError(f"{key}: only {win.sum()} MCEq grid points around {c['e_match_gev']} GeV; "
                                 "change LOW_E_SPLICE e_match_gev or blend_dex.")
            k = np.exp(np.mean(np.log(flux[win] / shape(e_grid[win]))))
            if geomagnetic_cutoff is not None and key in CUTOFF_SCALING['species']:
                k *= cutoff_scaling_factor(geomagnetic_cutoff, h_obs_km)
            s = 0.5 * (1.0 + np.tanh(np.log10(e_ext / c['e_match_gev']) / c['blend_dex']))
            mceq_ext = np.exp(s * np.log(mceq_ext) + (1.0 - s) * np.log(np.clip(k * shape(e_ext), 1e-300, None)))
        fluxes[key] = np.clip(mceq_ext, 1e-44, None)
    primary = log_interp1d(e_grid, np.clip(total_flux(e_grid), 1e-300, None))(e_ext)
    return e_ext, fluxes, primary


def _corrected_secondaries(mceq_run, model_instance, angles=DEFAULT_ANGLES_DEG, weights=DEFAULT_ANGLE_WEIGHTS,
                           geomagnetic_cutoff=None, h_obs_km=None):
    """
    Direct solver: `_zenith_averaged_secondaries` at mceq_run's current primary model, then
    `_postprocess_secondaries` (see there for the arguments and the return value).
    """
    raw = _zenith_averaged_secondaries(mceq_run, angles, weights)
    return _postprocess_secondaries(mceq_run.e_grid, raw, model_instance.total_flux,
                                    geomagnetic_cutoff=geomagnetic_cutoff,
                                    h_obs_km=h_obs_km)


def _set_h_obs(mceq_run, h_obs_cm):
    """
    Change observation height on an *existing* MCEqRun, in place.

    Three things verified directly against this real install before
    writing this (none worked the way the API alone suggests):
      1. `mceq_run.density_model.set_h_obs(h_obs_cm)` runs without error.
      2. On its own, that has NO effect on `solve()`'s output --
         `EarthsAtmosphere.set_h_obs` only calls
         `self.calculate_density_spline()` `if self.theta_deg:`, and
         0.0 (the vertical, used everywhere in this pipeline) is falsy
         in Python, so the spline recalculation is silently skipped.
      3. Even after forcing that recalculation directly
         (`density_model.set_theta(density_model.theta_deg)`, which
         recalculates unconditionally), `solve()` *still* reuses a
         stale result: `MCEqRun._calculate_integration_path` caches the
         integration path and only recomputes it if `int_grid`,
         `grid_var`, or the ETD2 step-size parameters change -- never
         checking whether the atmosphere itself changed. The fix is to
         invalidate that cache directly.
    Confirmed all three together produce a real, physically sensible
    change (~6.9x muon flux increase at 12 km vs sea level at ~1 GeV).
    """
    mceq_run.density_model.set_h_obs(h_obs_cm)
    mceq_run.density_model.set_theta(mceq_run.density_model.theta_deg)
    mceq_run.integration_path = None


# --- MCEq runs and the response matrix ----------------------------------------------------------------------------
# Database file forced on MCEq (None: MCEq's own default). Part of the response-matrix cache name.
MCEQ_DB_FNAME = "mceq_db_lext_dpm193_v142.h5"
# 'response': secondaries from a cached response matrix (MCEq runs once per primary energy cell, then every
# spectrum is a matrix product); 'direct': one full MCEq solve per spectrum (the previous behaviour).
DEFAULT_SOLVER = "response"
# Largest relative difference allowed between the response matrix and a direct solve of the baseline when the
# matrix is built (only where the flux is above 1e-10 of its maximum).
RESPONSE_VALIDATION_RTOL = 1e-3
# Write templates built with the response matrix to <template_dir>/<name>.npz (and read them back)? They are rebuilt
# from the cached matrix in a fraction of a second, so by default only the matrices are stored on disk.
SAVE_RESPONSE_TEMPLATES = False
_RESPONSE_CACHE = {}
_TEMPLATE_CACHE = {}


def _new_mceq_run(interaction_model=DEFAULT_INTERACTION_MODEL):
    """An MCEqRun (vertical, Hillas-Gaisser H3a) with the pipeline's settings: quiet, MCEQ_DB_FNAME database."""
    import crflux.models as pm
    from MCEq.core import MCEqRun
    import MCEq

    MCEq.config.debug_level = 0
    if MCEQ_DB_FNAME is not None:
        MCEq.config.mceq_db_fname = MCEQ_DB_FNAME
    return MCEqRun(interaction_model=interaction_model, primary_model=(pm.HillasGaisser2012, "H3a"), theta_deg=0.0)


class _NucleonInjection(PrimaryFlux):
    """
    Primary "model" used to build the response matrix. MCEq sets its initial condition from the model's
    p_and_n_flux, evaluated on its own grid of nucleon energies above config.minimal_primary_energy:
      * record mode (energies=None): stores that grid and returns zero flux;
      * injection mode: unit flux (crflux units) of protons (nucleon=0) or neutrons (nucleon=1) in the single cell
        `index` of the recorded grid, zero elsewhere.
    Going through p_and_n_flux, exactly as a real model does, keeps the matrix independent of MCEq internals.
    """

    def __init__(self, energies=None, index=None, nucleon=0):
        super().__init__(geomagnetic_cutoff=None)
        self.name = 'NucleonInjection'
        self.sname = 'Injection'
        self.nucleus_ids = [14]
        self.energies = None if energies is None else np.asarray(energies, dtype=float)
        self.index, self.nucleon = index, nucleon

    def _nucleus_flux(self, corsika_id, E):
        return np.zeros_like(np.atleast_1d(np.asarray(E, dtype=float)))

    def p_and_n_flux(self, E):
        E = np.atleast_1d(np.asarray(E, dtype=float))
        zeros = np.zeros_like(E)
        if self.energies is None:
            self.energies = E.copy()
            return zeros, zeros, zeros.copy()
        if E.shape != self.energies.shape or not np.allclose(E, self.energies, rtol=1e-12):
            raise RuntimeError("MCEq evaluated the primary model on a different energy grid than the one recorded "
                               "for the response matrix.")
        p, n = zeros.copy(), zeros.copy()
        (p if self.nucleon == 0 else n)[self.index] = 1.0
        return p, p, n                                  # proton fraction (unused by MCEq), protons, neutrons


class ResponseMatrix:
    """
    Zenith-averaged secondary fluxes per unit primary nucleon flux, for one atmosphere (interaction model,
    observation altitude, zenith quadrature, MCEq database):

        Phi_out[species](E_out) = sum_j  R[species][E_out, 0, j] * phi_p(E_j)  +  R[species][E_out, 1, j] * phi_n(E_j),

    with phi_p, phi_n the primary model's p_and_n_flux on MCEq's nucleon grid E_j (crflux units, as MCEq takes
    them). MCEq's cascade equations are linear in the initial condition, so this reproduces a full solve to
    numerical precision; it is checked against a direct solve of the baseline when built. Nuclei enter through
    p_and_n_flux (superposition, as in MCEq itself) after crflux has applied the per-nucleus cutoff Z * Rc, so
    composition, spectral shape and geomagnetic cutoff all live in the primary vector and need no new matrix.

    Cost: one MCEq solve (all zenith angles) per nucleon type and energy cell, ~2 x 100, once per atmosphere;
    cached in the template directory and in memory.
    """

    def __init__(self, e_grid, e_in, response, meta=None):
        self.e_grid = np.asarray(e_grid, dtype=float)
        self.e_in = np.asarray(e_in, dtype=float)
        self.response = {key: np.asarray(val, dtype=float) for key, val in response.items()}
        self.meta = dict(meta or {})

    def apply(self, model_instance):
        """Zenith-averaged MCEq fluxes (raw units, on e_grid) for a crflux primary model, as a matrix product."""
        _, p, n = model_instance.p_and_n_flux(self.e_in)
        return self.apply_nucleons(p, n)

    def apply_nucleons(self, p, n):
        """Zenith-averaged MCEq fluxes (raw units, on e_grid) for proton and neutron fluxes on e_in (crflux units)."""
        p = np.broadcast_to(np.asarray(p, dtype=float), self.e_in.shape)
        n = np.broadcast_to(np.asarray(n, dtype=float), self.e_in.shape)
        return {key: R[:, 0, :] @ p + R[:, 1, :] @ n for key, R in self.response.items()}

    @staticmethod
    def filename(interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, n_angles=len(DEFAULT_ANGLES_DEG)):
        db = "default" if MCEQ_DB_FNAME is None else os.path.splitext(MCEQ_DB_FNAME)[0]
        return f"Response_{interaction_model}_{h_obs_km or 0.0}km_{n_angles}ang_{db}.npz"

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {"e_grid": self.e_grid, "e_in": self.e_in}
        payload.update({f"R__{key}": val for key, val in self.response.items()})
        payload.update({f"meta__{key}": np.asarray(val) for key, val in self.meta.items()})
        np.savez_compressed(filepath, **payload)

    @classmethod
    def load(cls, filepath):
        with np.load(filepath, allow_pickle=False) as data:
            response = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith("R__")}
            meta = {k.split("__", 1)[1]: data[k].item() for k in data.files if k.startswith("meta__")}
            return cls(data["e_grid"], data["e_in"], response, meta)

    @classmethod
    def build(cls, interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, angles=DEFAULT_ANGLES_DEG,
              weights=DEFAULT_ANGLE_WEIGHTS, validate=True, verbose=True):
        """
        Runs MCEq once per primary nucleon type and energy cell.

        Raises:
            RuntimeError: If MCEq does not set its initial condition through p_and_n_flux, or if the matrix does
                not reproduce a direct solve of the baseline within RESPONSE_VALIDATION_RTOL (use solver='direct').
        """
        run = _new_mceq_run(interaction_model)
        h_obs_cm = (h_obs_km if h_obs_km is not None else 0.) * 1.0e5

        recorder = _NucleonInjection()
        run.set_primary_model(recorder)
        if recorder.energies is None:
            raise RuntimeError("MCEq did not evaluate p_and_n_flux in set_primary_model: the response matrix cannot "
                               "be built with this MCEq version; use solver='direct'.")
        e_in = recorder.energies
        n_in = e_in.size
        response = {key: np.zeros((run.e_grid.size, 2, n_in)) for key in MCEQ_SPECIES}
        if verbose:
            print(f"Building MCEq response matrix ({interaction_model}, h_obs={h_obs_km or 0.0} km): "
                  f"2 x {n_in} primary cells, {len(angles)} zenith angles each.")
        for j in range(n_in):
            for nucleon in (0, 1):
                run.set_primary_model(_NucleonInjection(e_in, j, nucleon))
                _set_h_obs(run, h_obs_cm)
                raw = _zenith_averaged_secondaries(run, angles, weights)
                for key in MCEQ_SPECIES:
                    response[key][:, nucleon, j] = raw[key]
            if verbose and (j + 1) % 20 == 0:
                print(f"  {j + 1}/{n_in} cells")

        matrix = cls(run.e_grid, e_in, response, meta={
            "interaction_model": interaction_model, "h_obs_km": float(h_obs_km or 0.0), "n_angles": len(angles),
            "mceq_db": str(MCEQ_DB_FNAME)})
        if validate:
            matrix._validate(run, h_obs_cm, angles, weights)
            if verbose:
                print(f"  checked against a direct solve: max relative difference "
                      f"{matrix.meta['validation_max_rel_diff']:.2e}")
        return matrix

    def _validate(self, run, h_obs_cm, angles, weights):
        """Compares apply() with a direct solve of the H3a baseline at 5 GV; stores the largest difference."""
        import crflux.models as pm

        model = pm.HillasGaisser2012("H3a", geomagnetic_cutoff=5.0)
        run.set_primary_model(model)
        _set_h_obs(run, h_obs_cm)
        direct = _zenith_averaged_secondaries(run, angles, weights)
        via = self.apply(model)
        worst, worst_key = 0.0, None
        for key in MCEQ_SPECIES:
            ref = direct[key]
            mask = ref > 1e-10 * ref.max() if ref.max() > 0 else np.zeros_like(ref, dtype=bool)
            if mask.any():
                diff = np.max(np.abs(via[key][mask] - ref[mask]) / ref[mask])
                if diff > worst:
                    worst, worst_key = diff, key
        self.meta["validation_max_rel_diff"] = float(worst)
        if worst > RESPONSE_VALIDATION_RTOL:
            raise RuntimeError(f"Response matrix differs from a direct MCEq solve by {worst:.2e} ({worst_key}); "
                               "not saved. Use solver='direct'.")

    @classmethod
    def get_or_build(cls, interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, template_dir=None,
                     force_rebuild=False, angles=DEFAULT_ANGLES_DEG, weights=DEFAULT_ANGLE_WEIGHTS):
        """The response matrix of one atmosphere: from memory, from <template_dir>/Response_*.npz, or built."""
        template_dir = config.flux_template_dir() if template_dir is None else template_dir
        path = os.path.join(template_dir, cls.filename(interaction_model, h_obs_km, len(angles)))
        if not force_rebuild:
            if path in _RESPONSE_CACHE:
                return _RESPONSE_CACHE[path]
            if os.path.exists(path):
                _RESPONSE_CACHE[path] = cls.load(path)
                return _RESPONSE_CACHE[path]
        matrix = cls.build(interaction_model, h_obs_km, angles, weights)
        matrix.save(path)
        _RESPONSE_CACHE[path] = matrix
        return matrix


def _build_template(name, model_factory, t_since_yr=None, interaction_model=DEFAULT_INTERACTION_MODEL,
                    h_obs_km=None, geomagnetic_cutoff=5.0, solver=None, template_dir=None):
    """
    Builds a FluxTemplate: the primary model model_factory(t) at each time of t_since_yr (model_factory(None) once
    for a steady flux) -> MCEq secondaries, zenith-averaged -> _postprocess_secondaries.

    Args:
        name (str): Template name.
        model_factory (callable): t [yr] or None -> PrimarySource (see _primary_source).
        t_since_yr (np.ndarray, optional): Template times since the event [yr]. None for a steady template.
        solver (str, optional): 'response' (ResponseMatrix of this atmosphere, built once and cached in
            template_dir) or 'direct' (one MCEq solve per spectrum). Defaults to None: DEFAULT_SOLVER.
        template_dir (str, optional): Where the response matrix is cached. Defaults to None:
            config.flux_template_dir().
    """
    solver = DEFAULT_SOLVER if solver is None else solver
    if solver not in ("response", "direct"):
        raise ValueError(f"Unknown solver {solver!r} (use 'response' or 'direct').")

    if solver == "response":
        matrix = ResponseMatrix.get_or_build(interaction_model, h_obs_km, template_dir)

        def solve(source):
            p, n = nucleon_spectra(source, matrix.e_in, geomagnetic_cutoff)
            return _postprocess_secondaries(matrix.e_grid, matrix.apply_nucleons(p, n),
                                            lambda E: total_nucleus_flux(source, E, geomagnetic_cutoff),
                                            geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)
    else:
        mceq_run = _new_mceq_run(interaction_model)
        reference_h_obs_cm = (h_obs_km if h_obs_km is not None else 0.) * 1.0e5

        def solve(source):
            model_instance = _SourceAsCrflux(source, geomagnetic_cutoff)
            mceq_run.set_primary_model(model_instance)
            _set_h_obs(mceq_run, reference_h_obs_cm)
            return _corrected_secondaries(mceq_run, model_instance, DEFAULT_ANGLES_DEG,
                                          geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)

    if t_since_yr is None:
        e_ext, fluxes, primary = solve(model_factory(None))
    else:
        steps, primary = [], []
        for t_yr in t_since_yr:
            e_ext, fluxes_t, primary_t = solve(model_factory(t_yr))
            steps.append(fluxes_t)
            primary.append(primary_t)
        fluxes = {key: np.array([f[key] for f in steps]) for key in MCEQ_SPECIES}

    return FluxTemplate(
        name, e_ext,
        {
            "primary": np.array(primary) * 1e-4,
            **{key: fluxes[key] for key in MCEQ_SPECIES},
        },
        t_since_yr=t_since_yr,
        meta={"h_obs_km": h_obs_km, "geomagnetic_cutoff": geomagnetic_cutoff, "interaction_model": interaction_model},
    )
