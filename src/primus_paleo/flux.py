import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import erfc, roots_jacobi

from . import config

def _hemisphere_gauss_legendre(n_angles):
    """
    Zenith angles (degrees) and weights for the Lambertian (cosine-
    projected) hemisphere flux integral,

        Phi_horiz = integral of I(theta) * cos(theta) dOmega
                  = 2*pi * integral_0^1 I(mu) * mu dmu,   mu = cos(theta)

    Uses Gauss-Jacobi quadrature (weight function mu on [0,1]) so the
    extra factor of mu is absorbed into the quadrature itself, keeping
    the same machine-precision exactness (up to degree 2*n_angles-1)
    the previous Gauss-Legendre scheme had. Verified against
    int_0^1 mu**k * mu dmu for k up to 21, and against the isotropic
    check (Phi_horiz = pi for constant I=1).

    Returns:
      (theta_deg, weights): weights sum to pi (not 1). Summing MCEq's
      raw solution (cm^-2 s^-1 sr^-1 GeV^-1) against these weights
      therefore gives the flux crossing a horizontal target directly,
      in cm^-2 s^-1 GeV^-1 -- no further sr^-1 or area rescale needed.
    """
    x, w_jacobi = roots_jacobi(n_angles, alpha=0, beta=1)
    mu = 0.5 * (x + 1.0)
    weights = 0.5 * np.pi * w_jacobi
    theta_deg = np.rad2deg(np.arccos(mu))
    return theta_deg, weights
 
 
DEFAULT_ANGLES_DEG, DEFAULT_ANGLE_WEIGHTS = _hemisphere_gauss_legendre(11)

DEFAULT_INTERACTION_MODEL = "EPOS-LHC-R"


class FluxTemplate:
    """
    A single precomputed (time-since-onset x energy) flux grid for one or
    more particle species, produced by one MCEq configuration.

    A template is either:
      * "steady" (`is_steady=True`): a single energy spectrum, constant in
        time. Used for baselines.
      * "transient": a 2D (t_since_kyr, energy_gev) grid representing the
        flux *excess* relative to a baseline, decaying back towards zero
        as t_since_kyr grows. Used for discrete events.

    Templates loaded from disk involve no MCEq / crflux import and are
    essentially instantaneous; only `get_or_build` (on a cache miss) pays
    the MCEq cost.
    """

    def __init__(self, name, energy_gev, species_grids, t_since_kyr=None):
        self.name = name
        self.energy_gev = np.asarray(energy_gev, dtype=float)
        self.t_since_kyr = None if t_since_kyr is None else np.asarray(t_since_kyr, dtype=float)
        self.is_steady = self.t_since_kyr is None

        self._grids = {}
        self._interp = {}
        for species, grid in species_grids.items():
            grid = np.asarray(grid, dtype=float)
            self._grids[species] = grid
            self._interp[species] = self._build_interpolator(grid)

    def _build_interpolator(self, grid):
        log_e = np.log10(self.energy_gev)
        log_flux = np.log10(np.clip(grid, 1e-300, None))
        if self.is_steady:
            return interp1d(
                log_e, log_flux, kind='linear',
                bounds_error=False, fill_value='extrapolate',
            )
        return RegularGridInterpolator(
            (self.t_since_kyr, log_e), log_flux,
            method='linear', bounds_error=False, fill_value=None,
        )

    @property
    def species(self):
        return list(self._grids.keys())

    def flux(self, species, energy_gev, t_since_kyr=None):
        """
        Evaluate this template for one species.

        Args:
            species: one of the species stored in this template.
            energy_gev: energies [GeV], array-like.
            t_since_kyr: required for transient templates (ignored for
                steady ones); broadcastable against energy_gev.
        """
        energy_gev = np.asarray(energy_gev, dtype=float)
        if species not in self._interp:
            return np.zeros_like(energy_gev)

        log_e = np.log10(energy_gev)
        if self.is_steady:
            return np.power(10.0, self._interp[species](log_e))

        if t_since_kyr is None:
            raise ValueError(f"Template '{self.name}' is transient: t_since_kyr is required.")

        t = np.asarray(t_since_kyr, dtype=float)

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
        if self.t_since_kyr is not None:
            payload['t_since_kyr'] = self.t_since_kyr
        for species, grid in self._grids.items():
            payload[f'species__{species}'] = grid
        np.savez_compressed(filepath, **payload)

    @classmethod
    def load(cls, filepath):
        """Load a template previously written with `.save()`. No MCEq needed."""
        with np.load(filepath, allow_pickle=False) as data:
            energy_gev = data['energy_gev']
            t_since_kyr = data['t_since_kyr'] if 't_since_kyr' in data else None
            species_grids = {
                key.split('__', 1)[1]: data[key]
                for key in data.files if key.startswith('species__')
            }
        name = os.path.splitext(os.path.basename(filepath))[0]
        return cls(name, energy_gev, species_grids, t_since_kyr=t_since_kyr)

    @classmethod
    def get_or_build(cls, params, template_dir=None, force_rebuild=False):
        """
        `params` is a small dict describing a physical scenario, e.g. `{"kind": "SN", "distance_pc": 200}`.

        If it's already on disk (and `force_rebuild` is False), loads and returns it. Otherwise, calls MCEq (lazily
        imported) to build it, saves it to `template_dir`, and returns it. Next time, this scenario is loaded.

        Kinds: 'Baseline' (Hillas-Gaisser H3a), 'SN' (SNHG12, params age-grid cut 'tcut' and 'distance_pc'),
        'Egal' (EgalHG12) and 'Tabulated' (a primary spectrum given as arrays, see `primary_spectrum_spec`).
        Common optional keys: 'geomagnetic_cutoff' [GV] (default 5.0), 'h_obs_km', 'interaction_model',
        'solver' ('response' or 'direct', default DEFAULT_SOLVER; not part of the cache name, both give the same
        template).

        Caching: templates built with the response matrix are rebuilt from it on demand (a fraction of a second)
        and kept only in memory for the session, unless SAVE_RESPONSE_TEMPLATES is True; only the matrices
        (Response_*.npz) live on disk. Templates of the 'direct' solver are saved to and loaded from
        <template_dir>/<name>.npz as before. A bare template name in FluxHistory still loads its .npz file.
        """
        template_dir = config.flux_template_dir() if template_dir is None else template_dir
        params = dict(params)
        kind = params.get("kind", "Baseline")
        solver = params.get("solver") or DEFAULT_SOLVER
        on_disk = solver == "direct" or SAVE_RESPONSE_TEMPLATES

        if on_disk and kind != "Tabulated":
            name = "_".join(str(val) for key, val in params.items() if key != "solver")
            path = os.path.join(template_dir, name + ".npz")
            if os.path.exists(path) and not force_rebuild:
                return cls.load(path)

        params.pop("kind", None)
        params.pop("solver", None)
        geomagnetic_cutoff = params.pop("geomagnetic_cutoff", 5.0)
        h_obs_km = params.pop("h_obs_km", None)
        interaction_model = params.pop("interaction_model", DEFAULT_INTERACTION_MODEL)

        if kind == "Tabulated":
            data = _tabulated_data(params)
            name = "_".join(["Tabulated", str(params.get("name", "primary")), data["hash"], f"{geomagnetic_cutoff}GV"])
        else:
            name = "_".join([kind, *(str(val) for val in params.values()), f"{geomagnetic_cutoff}GV"])
        if h_obs_km:
            name = name + "_" + str(h_obs_km) + "km"

        path = os.path.join(template_dir, name + ".npz")
        memory_key = (os.path.abspath(template_dir), name, interaction_model)
        if not force_rebuild:
            if memory_key in _TEMPLATE_CACHE:
                return _TEMPLATE_CACHE[memory_key]
            if on_disk and os.path.exists(path):
                return cls.load(path)

        if kind == "Tabulated":
            model, params_tuple, t_since_kyr = TabulatedPrimary, (data,), data["t_since_yr"]
        else:
            params_tuple = tuple(params.values())
            if kind == "Baseline":
                model = None
            elif kind == "SN":
                model = SNHG12
            elif kind == "Egal":
                model = EgalHG12
            else:
                raise ValueError(f"Unknown template kind: {kind}")

            tcut = params.get("tcut", 500.e3)
            nbins = min(250, int(tcut / 2.e2))
            t_since_kyr = np.linspace(1., tcut, nbins)

        template = _build_template(name=name, model=model, params=params_tuple, t_since_kyr=t_since_kyr,
                                   interaction_model=interaction_model, geomagnetic_cutoff=geomagnetic_cutoff,
                                   h_obs_km=h_obs_km, solver=solver, template_dir=template_dir)
        if on_disk:
            template.save(path)
        _TEMPLATE_CACHE[memory_key] = template
        return template

    @classmethod
    def from_primary_spectrum(cls, energy_gev, flux, composition=None, t_since_yr=None, name="primary",
                              template_dir=None, force_rebuild=False, **options):
        """
        Template whose secondaries MCEq computes from a primary spectrum given as arrays: shorthand for
        `get_or_build(primary_spectrum_spec(...))`. See `primary_spectrum_spec` for the arguments; `options` are
        the usual 'geomagnetic_cutoff', 'h_obs_km', 'interaction_model', 'solver'. To have the template respond to
        geomagnetic excursions, pass the spec dict to FluxHistory instead of this pre-built template.
        """
        spec = primary_spectrum_spec(energy_gev, flux, composition=composition, t_since_yr=t_since_yr, name=name,
                                     **options)
        return cls.get_or_build(spec, template_dir=template_dir, force_rebuild=force_rebuild)


class FluxHistory:
    """
    Baseline template + a list of discrete events (plain dicts), summed on
    demand. Structurally the flux-side analogue of `overburden_history`:
    a small spec goes in, one object answers every downstream query.

    Timeline: see module docstring -- 0 = present, negative = past.
    """

    def __init__(self, baseline={"kind": "Baseline"}, events=None, excursions=None, template_dir=None):
        """
        Args:
            baseline: a FluxTemplate, a cached template name (str), or a
                params dict (built on demand) -- e.g. {"kind": "baseline"}.
            events: list of dicts, each with 'start_time_kyr' (or
                'time_kyr') and 'template' (FluxTemplate / name / params
                dict), and optionally 'weight' / 'label'.
            excursions: list of dicts, each with 'start_time_kyr',
                'end_time_kyr', 'geomagnetic_cutoff', and optionally
                'label' -- see `add_excursion`.
            template_dir: directory used to load/save templates. Defaults to None: <data root>/flux_data
                (config.flux_template_dir()).
            name: optional human label; if omitted, `.signature` derives
                one from the actual content.
        """
        self.template_dir = config.flux_template_dir() if template_dir is None else template_dir
        self._baseline_spec = baseline
        self._baseline_cutoff_variants = {}
        self.baseline = self._resolve(baseline)
        self.events = []
        for event in (events or []):
            start_time = event.get("start_time_kyr")
            self.add_event(start_time, event["template"], weight=event.get("weight", 1.0), label=event.get("label"))
        self.excursions = []
        for excursion in (excursions or []):
            self.add_excursion(
                excursion["start_time_kyr"], excursion["end_time_kyr"],
                excursion["geomagnetic_cutoff"], label=excursion.get("label"),
            )

    def _resolve(self, template):
        if isinstance(template, FluxTemplate):
            return template
        if isinstance(template, dict):
            return FluxTemplate.get_or_build(template, self.template_dir)
        if isinstance(template, str):
            path = os.path.join(self.template_dir, f"{template}.npz")
            if os.path.exists(path):
                return FluxTemplate.load(path)
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
        resolved = self._resolve(template)
        self.events.append({
            "template": resolved, "spec": template, "start_time_kyr": float(start_time_kyr),
            "weight": weight, "label": label, "_cutoff_variants": {},
        })
        self._name = None 
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
            t_since = np.asarray(t_kyr, dtype=float) - event["start_time_kyr"]
            t_since *= 1.e3
            total = total + event["weight"] * template.flux(species, energy_gev, t_since_kyr=t_since)
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


class SNHG12(PrimaryFlux):

    def __init__(self, model, geomagnetic_cutoff=5.0):
        super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
        self.name = 'Mod'
        self.sname = model
        self.t = model[0] * 3.154e7
        self.d = model[1] * 3.086e18
        self.params = {}
        self.SNfrac = {}
        self.rid_cutoff = {}

        mass_comp = [14, 402, 1206, 2814, 5426]
        for mcomp in mass_comp:
            self.params[mcomp] = {}
            self.SNfrac[mcomp] = 0.0

        self.rid_cutoff[1] = 4e6
        self.rid_cutoff[2] = 30e6
        self.rid_cutoff[3] = 2e9

        self.SNfrac[14] = 0.90    # H
        self.SNfrac[402] = 0.08   # He
        self.SNfrac[1206] = 0.01  # CNO
        self.SNfrac[2814] = 0.005  # MgAlSi
        self.SNfrac[5426] = 0.005  # Fe

        self.params[14][1] = (1. * 7860, 1.66, 1)    # H
        self.params[402][1] = (1. * 3550, 1.58, 2)   # He
        self.params[1206][1] = (1. * 2200, 1.63, 6)  # CNO
        self.params[2814][1] = (1. * 1430, 1.67, 14)  # MgAlSi
        self.params[5426][1] = (1. * 2120, 1.63, 26)  # Fe

        self.params[14][2] = (1 * 20, 1.4, 1)      # H
        self.params[402][2] = (1 * 20, 1.4, 2)     # He
        self.params[1206][2] = (1 * 13.4, 1.4, 6)  # CNO
        self.params[2814][2] = (1 * 13.4, 1.4, 14)  # MgAlSi
        self.params[5426][2] = (1 * 13.4, 1.4, 26)  # Fe

        self.params[14][3] = (1 * 1.7, 1.4, 1)      # H
        self.params[402][3] = (1 * 1.7, 1.4, 2)     # He
        self.params[1206][3] = (1 * 1.14, 1.4, 6)   # CNO
        self.params[2814][3] = (1 * 1.14, 1.4, 14)  # MgAlSi
        self.params[5426][3] = (1 * 1.14, 1.4, 26)  # Fe

        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0

        D0 = 2e28
        Q0 = 4.e52 * self.SNfrac[corsika_id]
        Ecut = 1e6
        D = D0 * (E**(1 / 3))
        Q = Q0 * (E**-2.2) * np.exp(-E / self.params[corsika_id][3][2] / Ecut)
        flux += ((Q) / (np.pi**1.5 * np.sqrt(4 * D * self.t)**3) *
                 np.exp(-self.d**2 / (4 * D * self.t))) * 1e4 * 3e10 / 4 * np.pi

        return flux

class EgalHG12(PrimaryFlux):

    def __init__(self, model, geomagnetic_cutoff=0.0):
        super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
        self.name = 'Mod'
        self.sname = model
        self.t = model[0] * 3.154e7
        self.egfactor = model[1]
        self.egcut = model[2]
        self.tcut = model[3] * 3.154e7
        self.params = {}
        self.SNfrac = {}
        self.rid_cutoff = {}

        mass_comp = [14, 402, 1206, 2814, 5426]
        for mcomp in mass_comp:
            self.params[mcomp] = {}

        self.rid_cutoff[1] = 4e6
        self.rid_cutoff[2] = 30e6
        self.rid_cutoff[3] = 2e9

        self.params[14][1] = (1. * 7860, 1.66, 1)    # H
        self.params[402][1] = (1. * 3550, 1.58, 2)   # He
        self.params[1206][1] = (1. * 2200, 1.63, 6)  # CNO
        self.params[2814][1] = (1. * 1430, 1.67, 14)  # MgAlSi
        self.params[5426][1] = (1. * 2120, 1.63, 26)  # Fe

        self.params[14][2] = (1 * 20, 1.4, 1)      # H
        self.params[402][2] = (1 * 20, 1.4, 2)     # He
        self.params[1206][2] = (1 * 13.4, 1.4, 6)  # CNO
        self.params[2814][2] = (1 * 13.4, 1.4, 14)  # MgAlSi
        self.params[5426][2] = (1 * 13.4, 1.4, 26)  # Fe

        self.params[14][3] = (1 * 1.7, 1.4, 1)      # H
        self.params[402][3] = (1 * 1.7, 1.4, 2)     # He
        self.params[1206][3] = (1 * 1.14, 1.4, 6)   # CNO
        self.params[2814][3] = (1 * 1.14, 1.4, 14)  # MgAlSi
        self.params[5426][3] = (1 * 1.14, 1.4, 26)  # Fe

        self.nucleus_ids = list(self.params.keys())

    def _nucleus_flux(self, corsika_id, E):
        corsika_id = self._find_nearby_id(corsika_id)

        flux = 0.0

        tdecay = np.exp(-self.t / self.tcut) if self.t > self.tcut else 1.0

        flux += self.egfactor * self.params[corsika_id][3][0] * E ** (-self.params[corsika_id][3][1] - 1.0) * \
            np.exp(-E / self.params[corsika_id][3][2] / self.egcut) * tdecay

        return flux


# --- Primary spectra given as arrays --------------------------------------------------------------------------------
# Mass groups of the Hillas-Gaisser models (CORSIKA ids: 100 A + Z, protons 14).
MASS_GROUP_IDS = {'H': 14, 'p': 14, 'proton': 14, 'He': 402, 'CNO': 1206, 'MgAlSi': 2814, 'Fe': 5426}


def primary_spectrum_spec(energy_gev, flux, composition=None, t_since_yr=None, name="primary", **options):
    """
    Template spec (a params dict for FluxTemplate.get_or_build / FluxHistory) for a primary cosmic-ray spectrum
    given as arrays. MCEq then computes every secondary species from it, so each species responds to the shape of
    the spectrum (not a common ratio).

    Units are those of the template's own 'primary' species: energy per nucleus [GeV] (crflux convention, total
    energy) and flux per nucleus [cm^-2 s^-1 sr^-1 GeV^-1] at the top of the atmosphere, before the geomagnetic
    cutoff (applied by MCEq's primary model at 'geomagnetic_cutoff', default 5 GV, as for every other kind).
    Between the given energies the flux is interpolated in log-log; outside them it is 0, so cover the range that
    matters (MCEq uses primaries above ~3 GeV per nucleon; ~1e7 GeV is enough for everything but deep muons).

    Args:
        energy_gev (array_like): Energies per nucleus [GeV], increasing, shape (n_E,).
        flux (array_like or dict): Per-nucleus flux, either one array (all nuclei, split with `composition`) or a
            dict mass group -> array, with groups 'H' (or 'p'), 'He', 'CNO', 'MgAlSi', 'Fe' or CORSIKA ids.
            Shape (n_E,) for a steady spectrum, (n_t, n_E) for a transient one.
        composition (dict, optional): Fractions per mass group of a single `flux` array, at the same energy per
            nucleus (normalised to sum 1). Defaults to None: all protons.
        t_since_yr (array_like, optional): Time since the event onset of each row of a transient `flux`, in YEARS
            (the time axis of all transient templates is in years: FluxHistory multiplies its kyr by 1e3).
            Defaults to None (steady).
        name (str, optional): Label in the cache file name. Defaults to "primary".
        **options: 'geomagnetic_cutoff', 'h_obs_km', 'interaction_model', 'solver'.

    Returns:
        dict: {'kind': 'Tabulated', ...}, usable as a FluxHistory baseline or event template (and with excursions).

    Notes:
        A transient event template is the flux *excess* over the baseline, as for 'SN': give the excess primary
        flux. MCEq is linear, so the secondaries of the excess are exactly the excess of the secondaries.
    """
    spec = {"kind": "Tabulated", "name": name, "energy_gev": np.asarray(energy_gev, dtype=float)}
    if isinstance(flux, dict):
        spec["flux"] = {k: np.asarray(v, dtype=float) for k, v in flux.items()}
    else:
        spec["flux"] = np.asarray(flux, dtype=float)
    if composition is not None:
        spec["composition"] = dict(composition)
    if t_since_yr is not None:
        spec["t_since_yr"] = np.asarray(t_since_yr, dtype=float)
    spec.update(options)
    return spec


def _tabulated_data(params):
    """
    Normalises the arrays of a 'Tabulated' spec: energies, per-group 2D flux grids (n_t or 1, n_E) keyed by CORSIKA
    id, the time axis (None if steady) and a short hash of all of them for the cache name.
    """
    import hashlib

    energy = np.asarray(params["energy_gev"], dtype=float)
    if energy.ndim != 1 or energy.size < 2 or np.any(np.diff(energy) <= 0) or energy[0] <= 0:
        raise ValueError("Tabulated: 'energy_gev' must be 1D, positive and increasing, with at least 2 values.")
    t_since = params.get("t_since_yr")
    t_since = None if t_since is None else np.asarray(t_since, dtype=float)

    flux = params["flux"]
    if isinstance(flux, dict):
        if params.get("composition") is not None:
            raise ValueError("Tabulated: give 'composition' only with a single flux array, not with a dict.")
        raw = flux
    else:
        composition = params.get("composition") or {"H": 1.0}
        total = sum(composition.values())
        raw = {group: np.asarray(flux, dtype=float) * frac / total for group, frac in composition.items()}

    groups = {}
    for group, grid in raw.items():
        cid = MASS_GROUP_IDS.get(group, group)
        if not isinstance(cid, (int, np.integer)):
            raise ValueError(f"Tabulated: unknown mass group {group!r} (use {sorted(MASS_GROUP_IDS)} or CORSIKA ids).")
        grid = np.atleast_2d(np.asarray(grid, dtype=float))
        if grid.shape[-1] != energy.size:
            raise ValueError(f"Tabulated: flux of {group!r} has {grid.shape[-1]} energies, energy_gev has {energy.size}.")
        if np.any(grid < 0):
            raise ValueError(f"Tabulated: negative flux in {group!r}.")
        groups[int(cid)] = groups.get(int(cid), 0.) + grid

    n_rows = {g.shape[0] for g in groups.values()}
    if len(n_rows) != 1:
        raise ValueError("Tabulated: all mass groups need the same number of time rows.")
    n_rows = n_rows.pop()
    if t_since is None and n_rows != 1:
        raise ValueError("Tabulated: a 2D flux needs 't_since_yr' (one time per row).")
    if t_since is not None and (t_since.ndim != 1 or t_since.size != n_rows or np.any(np.diff(t_since) <= 0)):
        raise ValueError("Tabulated: 't_since_yr' must be increasing, one value per flux row.")

    h = hashlib.sha1(energy.tobytes())
    for cid in sorted(groups):
        h.update(str(cid).encode())
        h.update(np.ascontiguousarray(groups[cid]).tobytes())
    if t_since is not None:
        h.update(t_since.tobytes())
    return {"energy_gev": energy, "groups": groups, "t_since_yr": t_since, "hash": h.hexdigest()[:10]}


class TabulatedPrimary(PrimaryFlux):
    """
    crflux primary model built from arrays (see primary_spectrum_spec). Called like the other models:
    `TabulatedPrimary((data,), geomagnetic_cutoff)` for a steady spectrum, `TabulatedPrimary((t_since, data), ...)`
    for the row of a transient one at time t_since (one of data['t_since_yr']). Per-nucleus flux interpolated in
    log-log, 0 outside the tabulated energies; the cutoff is applied by crflux's nucleus_flux as for every model.
    """

    def __init__(self, model, geomagnetic_cutoff=5.0):
        super().__init__(geomagnetic_cutoff=geomagnetic_cutoff)
        if isinstance(model[0], dict):
            data, row = model[0], 0
        else:
            data = model[1]
            row = int(np.argmin(np.abs(data["t_since_yr"] - model[0])))
        self.name = 'Tabulated'
        self.sname = 'Tabulated'
        self._log_e = np.log(data["energy_gev"])
        # template units (cm^-2) -> crflux units (m^-2)
        self._log_flux = {cid: np.log(np.clip(grid[row] * 1e4, 1e-300, None)) for cid, grid in data["groups"].items()}
        self.nucleus_ids = sorted(self._log_flux)

    def _nucleus_flux(self, corsika_id, E):
        E = np.atleast_1d(np.asarray(E, dtype=float))
        log_flux = self._log_flux.get(corsika_id)
        if log_flux is None:
            return np.zeros_like(E)
        return np.exp(np.interp(np.log(E), self._log_e, log_flux, left=-np.inf, right=-np.inf))


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
    call and the solid-angle quadrature `weights` (see `_hemisphere_gauss_legendre`).

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


def _loglog(x_new, x, y):
    """Log-log linear interpolation, extrapolating linearly: what FluxTemplate does outside its grid."""
    f = interp1d(np.log10(x), np.log10(np.clip(y, 1e-300, None)), kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return np.power(10.0, f(np.log10(x_new)))


def _postprocess_secondaries(e_grid, raw_fluxes, model_instance, geomagnetic_cutoff=None, h_obs_km=None):
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
        model_instance: The crflux primary model (for the 'primary' species).
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
        mceq_ext = _loglog(e_ext, e_grid, flux)
        if key in NEUTRINO_LOW_E['species']:
            c = NEUTRINO_LOW_E
            em, ep = c['e_match_gev'], c['e_peak_gev']
            win = np.abs(np.log10(e_grid / em)) <= c['fit_dex']
            if win.sum() < 2:
                raise ValueError(f"{key}: only {win.sum()} MCEq grid points around {em} GeV; "
                                 "change NEUTRINO_LOW_E e_match_gev or fit_dex.")
            slope = np.polyfit(np.log(e_grid[win]), np.log(flux[win]), 1)[0]
            a = max(-slope, 0.0) / (2.0 * np.log(em / ep))           # same local slope at E_m
            low = _loglog(em, e_grid, flux) * np.exp(-a * (np.log(e_ext / ep) ** 2 - np.log(em / ep) ** 2))
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
    primary = _loglog(e_ext, e_grid, np.clip(model_instance.total_flux(e_grid), 1e-300, None))
    return e_ext, fluxes, primary


def _corrected_secondaries(mceq_run, model_instance, angles=DEFAULT_ANGLES_DEG, weights=DEFAULT_ANGLE_WEIGHTS,
                           geomagnetic_cutoff=None, h_obs_km=None):
    """
    Direct solver: `_zenith_averaged_secondaries` at mceq_run's current primary model, then
    `_postprocess_secondaries` (see there for the arguments and the return value).
    """
    raw = _zenith_averaged_secondaries(mceq_run, angles, weights)
    return _postprocess_secondaries(mceq_run.e_grid, raw, model_instance, geomagnetic_cutoff=geomagnetic_cutoff,
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


def _build_template(name=None, model=None, params=None, t_since_kyr=None, interaction_model=DEFAULT_INTERACTION_MODEL,
                    h_obs_km=None, geomagnetic_cutoff=5.0, solver=None, template_dir=None):
    """
    Builds a FluxTemplate: the primary model (class `model`, called as model(params, geomagnetic_cutoff=...), or
    model((t, *params), ...) at each time t of t_since_kyr for a transient) -> MCEq secondaries, zenith-averaged
    -> _postprocess_secondaries. model=None is the Hillas-Gaisser H3a baseline (steady).

    Args:
        solver (str, optional): 'response' (ResponseMatrix of this atmosphere, built once and cached in
            template_dir) or 'direct' (one MCEq solve per spectrum). Defaults to None: DEFAULT_SOLVER.
        template_dir (str, optional): Where the response matrix is cached. Defaults to None:
            config.flux_template_dir().

    Note: the time axis of transient templates (`t_since_kyr`) is in years (the models take model[0] in years and
    FluxHistory converts its kyr with *1e3); the name is historical.
    """
    import crflux.models as pm

    solver = DEFAULT_SOLVER if solver is None else solver
    if solver not in ("response", "direct"):
        raise ValueError(f"Unknown solver {solver!r} (use 'response' or 'direct').")

    if model is None:
        model = pm.HillasGaisser2012
        params = "H3a"
        name = "Baseline"
        t_since_kyr = None

    if name is None:
        name = f"{model[0].__name__}_{model[1]}"

    if params is None:
        params = ()

    if solver == "response":
        matrix = ResponseMatrix.get_or_build(interaction_model, h_obs_km, template_dir)

        def solve(model_instance):
            return _postprocess_secondaries(matrix.e_grid, matrix.apply(model_instance), model_instance,
                                            geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)
    else:
        mceq_run = _new_mceq_run(interaction_model)
        reference_h_obs_cm = (h_obs_km if h_obs_km is not None else 0.) * 1.0e5

        def solve(model_instance):
            mceq_run.set_primary_model(model_instance)
            _set_h_obs(mceq_run, reference_h_obs_cm)
            return _corrected_secondaries(mceq_run, model_instance, DEFAULT_ANGLES_DEG,
                                          geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)

    if t_since_kyr is None:
        e_ext, fluxes, primary = solve(model(params, geomagnetic_cutoff=geomagnetic_cutoff))
    else:
        steps, primary = [], []
        for age_kyr in t_since_kyr:
            e_ext, fluxes_t, primary_t = solve(model((age_kyr, ) + tuple(params), geomagnetic_cutoff=geomagnetic_cutoff))
            steps.append(fluxes_t)
            primary.append(primary_t)
        fluxes = {key: np.array([f[key] for f in steps]) for key in MCEQ_SPECIES}

    return FluxTemplate(
        name, e_ext,
        {
            "primary": np.array(primary) * 1e-4,
            **{key: fluxes[key] for key in MCEQ_SPECIES},
        },
        t_since_kyr=t_since_kyr,
    )
