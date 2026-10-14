import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.special import erfc, roots_jacobi

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
    def get_or_build(cls, params, template_dir="Data/flux_data", force_rebuild=False):
        """
        `params` is a small dict describing a
        physical scenario, e.g. `{"kind": "SN", "distance_pc": 200}`.

        If it's already on disk (and `force_rebuild` is False), loads
        and returns it. Otherwise, calls MCEq (lazily imported) to build it, saves it
        to `template_dir`, and returns it. Next time, this scenario is loaded.
        """
        name = "_".join(str(val) for val in params.values())
        path = os.path.join(template_dir, name + ".npz")
        if os.path.exists(path) and not force_rebuild:
            return cls.load(path)
        
        params = dict(params)
        kind = params.pop("kind", "Baseline")
        geomagnetic_cutoff = params.pop("geomagnetic_cutoff", 5.0)
        h_obs_km = params.pop("h_obs_km", None)
        interaction_model = params.pop("interaction_model", DEFAULT_INTERACTION_MODEL)

        name = "_".join([kind, *(str(val) for val in params.values()), f"{geomagnetic_cutoff}GV"])
        if h_obs_km: 
            name = name + "_" + str(h_obs_km) + "km"

        path = os.path.join(template_dir, name + ".npz")
        if os.path.exists(path) and not force_rebuild:
            return cls.load(path)

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

        template = _build_template(name=name, model=model, params=params_tuple, t_since_kyr=t_since_kyr, interaction_model=interaction_model, geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)
        template.save(path)
        return template

class FluxHistory:
    """
    Baseline template + a list of discrete events (plain dicts), summed on
    demand. Structurally the flux-side analogue of `overburden_history`:
    a small spec goes in, one object answers every downstream query.

    Timeline: see module docstring -- 0 = present, negative = past.
    """

    def __init__(self, baseline={"kind": "Baseline"}, events=None, excursions=None, template_dir="Data/flux_data"):
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
            template_dir: directory used to load/save templates.
            name: optional human label; if omitted, `.signature` derives
                one from the actual content.
        """
        self.template_dir = template_dir
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


def _loglog(x_new, x, y):
    """Log-log linear interpolation, extrapolating linearly: what FluxTemplate does outside its grid."""
    f = interp1d(np.log10(x), np.log10(np.clip(y, 1e-300, None)), kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return np.power(10.0, f(np.log10(x_new)))


def _corrected_secondaries(mceq_run, model_instance, angles=DEFAULT_ANGLES_DEG, weights=DEFAULT_ANGLE_WEIGHTS):
    """
    `_zenith_averaged_secondaries` and the primary flux on the MCEq grid extended down to LOW_E_GRID['e_min_gev'],
    with the low-energy correction of every LOW_E_SPLICE species. Used by every template (steady and, per time
    step, transient).

    For each spliced species the shape is continued above e_bridge by shape(e_bridge) (E / e_bridge)^-gamma;
    k = geometric mean of MCEq / shape over e_match * 10^(+-blend_dex), then a log-space tanh join,
        log Phi = s log Phi_MCEq + (1 - s) log(k shape),   s = (1 + tanh(log10(E / e_match) / blend_dex)) / 2
    so flux and slope are continuous. Muons and the primary are only extended as FluxTemplate would have
    extrapolated them (unchanged where they were queried before); neutrinos are continued below
    NEUTRINO_LOW_E['e_match_gev'] by a log-normal peaking at e_peak_gev. Fluxes are floored at 1e-44 for FluxTemplate's log-space interpolator.

    Returns:
        tuple: (e_ext, fluxes, primary): extended grid, dict species -> flux on e_ext, primary flux on e_ext.

    Raises:
        ValueError: If fewer than 2 MCEq grid points lie in a matching window.
    """
    e_grid = np.asarray(mceq_run.e_grid, dtype=float)
    n_low = max(int(np.ceil(np.log10(e_grid[0] / LOW_E_GRID['e_min_gev']) * LOW_E_GRID['n_per_decade'])), 0)
    e_ext = np.concatenate([np.logspace(np.log10(LOW_E_GRID['e_min_gev']), np.log10(e_grid[0]), n_low + 1)[:-1],
                            e_grid])

    fluxes = {}
    for key, flux in _zenith_averaged_secondaries(mceq_run, angles, weights).items():
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
            s = 0.5 * (1.0 + np.tanh(np.log10(e_ext / c['e_match_gev']) / c['blend_dex']))
            mceq_ext = np.exp(s * np.log(mceq_ext) + (1.0 - s) * np.log(np.clip(k * shape(e_ext), 1e-300, None)))
        fluxes[key] = np.clip(mceq_ext, 1e-44, None)
    primary = _loglog(e_ext, e_grid, model_instance.total_flux(e_grid))
    return e_ext, fluxes, primary


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


def _build_template(name=None, model=None, params=None, t_since_kyr=None, interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, geomagnetic_cutoff=5.0):

    import crflux.models as pm
    from MCEq.core import MCEqRun
    import MCEq

    MCEq.config.debug_level = 0

    if model is None:
        model = pm.HillasGaisser2012
        params = "H3a"
        name = "Baseline"
        t_since_kyr = None

    if name is None:
        name = f"{model[0].__name__}_{model[1]}"

    mceq_run = MCEqRun(
    interaction_model=interaction_model,
    primary_model=(pm.HillasGaisser2012, "H3a"),
    theta_deg=0.0
    )

    angles = DEFAULT_ANGLES_DEG

    if params is None:
        params = ()

    reference_h_obs_cm = (h_obs_km if h_obs_km is not None else 0.) * 1.0e5

    if t_since_kyr is None:

        model_instance = model(params, geomagnetic_cutoff=geomagnetic_cutoff)

        mceq_run.set_primary_model(model_instance)

        _set_h_obs(mceq_run, reference_h_obs_cm)

        e_ext, fluxes, primary = _corrected_secondaries(mceq_run, model_instance, angles)

    else:
        steps, primary = [], []

        for age_kyr in t_since_kyr:

            age_params = (age_kyr, ) + params

            model_instance = model(age_params, geomagnetic_cutoff=geomagnetic_cutoff)
            mceq_run.set_primary_model(model_instance)

            _set_h_obs(mceq_run, reference_h_obs_cm)

            e_ext, fluxes_t, primary_t = _corrected_secondaries(mceq_run, model_instance, angles)
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