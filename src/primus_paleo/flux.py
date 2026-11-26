"""Cosmic-ray fluxes at the site of a sample: primary fluxes, the atmospheric secondaries MCEq makes from them
(FluxTemplate) and their history over time (FluxHistory), as read by Paleodetector. The primary fluxes themselves
(spec kinds, model registry, built-in and tabulated models, nucleon split) are in primaries.

Timeline. FluxHistory works in absolute time [kyr]: 0 is the present, negative values the past (Paleodetector maps
its local time, 0 = start of exposure, onto it). A baseline holds at all times; an event (a supernova, a boost)
adds, from its start_time_kyr on, the flux of a transient template, whose own time axis t_since_yr is the time since
the event in years; a geomagnetic excursion swaps the cutoff of the baseline and events over [start, end).

Specs. A template is described by a spec dict of one of three kinds (details in primaries):
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

Zenith distribution. MCEq solves each zenith angle of the quadrature; besides their weighted sum, a template keeps,
per species, the angular shape g_i(E) = I_i(E) / sum_j w_j I_j(E) at the quadrature nodes mu_i = cos(theta_i)
(sum_i w_i g_i = 1), so that the intensity is I_i = flux * g_i and the flux crossing a horizontal plane per unit
cos(zenith) is 2 pi mu g(mu) * flux (FluxTemplate.zenith_shape, FluxHistory.get_zenith_map; used by Paleodetector
for the slanted paths through the overburden and for the neutrinos from all directions). Where MCEq is not used, the
nucleon shape is blended into a measured low-energy one (LOW_E_ZENITH) and the other shapes are held at their value
at the matching energy (_zenith_shapes).

Caching (_cached). Templates and response matrices are kept in memory for the session; the matrices are saved to
<template_dir>/Response_*.npz, and templates of the 'direct' solver (or all of them, with SAVE_RESPONSE_TEMPLATES) to
<template_dir>/<name>.npz (default config.flux_template_dir()).

Layout: settings; MCEq runs; response matrix; low-energy extension; building a template; FluxTemplate; FluxHistory.
"""

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import erfc, roots_jacobi

from . import config
from ._core import log_interp1d
from .primaries import (PrimaryFlux, _SourceAsCrflux, _primary_source, nucleon_spectra, primary_spectrum_spec,
                        total_nucleus_flux)

__all__ = ['FluxHistory', 'FluxTemplate', 'MCEQ_SPECIES']


# --- Settings: zenith quadrature, species, solvers ------------------------------------------------------------------
def _hemisphere_quadrature(n_angles):
    """Zenith angles and weights for the flux crossing a horizontal plane, from a zenith-dependent intensity.

        Phi_horizontal = integral I(theta) cos(theta) dOmega = 2 pi integral_0^1 I(mu) mu dmu,   mu = cos(theta)

    Gauss-Jacobi quadrature with weight function mu on [0, 1], so the factor mu is part of the rule (exact for
    polynomials in mu up to degree 2 n_angles - 1; a constant I = 1 gives pi). Summing MCEq's solutions
    [cm^-2 s^-1 sr^-1 GeV^-1] with these weights gives the flux crossing a horizontal plane [cm^-2 s^-1 GeV^-1].
    Used for DEFAULT_ANGLES_DEG / DEFAULT_ANGLE_WEIGHTS (_zenith_sum).

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


# Template species -> the MCEq solution names summed into it (one-element tuples: ('x',), not ('x')).
MCEQ_SPECIES = {
    'mu+': ('total_mu+',),
    'mu-': ('total_mu-',),
    'neutron': ('n0',),
    'proton': ('p+',),
    'nu_mu': ('total_numu',),
    'anti_nu_mu': ('total_antinumu',),
    'nu_e': ('total_nue',),
    'anti_nu_e': ('total_antinue',),
}


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
# In-memory caches of the session (see _cached): response matrices by file path, templates by (directory, name,
# interaction model).
_RESPONSE_CACHE = {}
_TEMPLATE_CACHE = {}


def _cached(memory, key, path, load, build, use_disk=True, force_rebuild=False):
    """Memory -> disk -> build lookup shared by FluxTemplate.get_or_build and ResponseMatrix.get_or_build.

    Args:
        memory (dict): Session cache.
        key: Key of the object in memory.
        path (str): File of the object on disk.
        load (callable): path -> object, or None if the file is outdated (then built again).
        build (callable): () -> object (then saved to path if use_disk).
        use_disk (bool, optional): Read and write path. Defaults to True.
        force_rebuild (bool, optional): Build even if cached. Defaults to False.

    Returns:
        The object (also stored in memory).
    """
    if not force_rebuild:
        if key in memory:
            return memory[key]
        if use_disk and os.path.exists(path):
            obj = load(path)
            if obj is not None:
                memory[key] = obj
                return obj
    obj = build()
    if use_disk:
        obj.save(path)
    memory[key] = obj
    return obj


def _h_obs_cm(h_obs_km):
    """Observation altitude [cm] for MCEq from h_obs_km [km] (None: sea level)."""
    return (h_obs_km if h_obs_km is not None else 0.) * 1.0e5


# --- MCEq runs --------------------------------------------------------------------------------------------------------


def _zenith_resolved_secondaries(mceq_run, angles=DEFAULT_ANGLES_DEG):
    """MCEq intensity of every MCEQ_SPECIES entry at each zenith angle, at mceq_run's current configuration
    (interaction model, primary model, atmosphere), from one solve_fullsky call. Called by ResponseMatrix.build and
    _validate and by the direct solver of _build_template.

    Returns:
        dict: species -> intensity [raw MCEq units, per sr], shape (n_angles, n_e) on mceq_run.e_grid.
    """
    result = mceq_run.solve_fullsky(zenith_grid=angles)
    return {key: np.array([sum(result.get_solution(name, zenith=float(theta), mag=0) for name in names)
                           for theta in angles])
            for key, names in MCEQ_SPECIES.items()}


def _zenith_sum(intensity, weights=DEFAULT_ANGLE_WEIGHTS):
    """Flux crossing a horizontal plane from intensities at the quadrature angles: sum_i w_i I_i over the first
    axis (_hemisphere_quadrature), added in angle order. Called wherever the per-angle MCEq results are integrated."""
    out = np.zeros(np.shape(intensity)[1:])
    for w, row in zip(weights, intensity):
        out += w * row
    return out


def _set_h_obs(mceq_run, h_obs_cm):
    """Sets the observation altitude of an existing MCEqRun, in place. Called before every solve that needs it
    (ResponseMatrix.build and _validate, the direct solver of _build_template).

    MCEq's own set_h_obs is not enough: it recomputes the density spline only if theta_deg is truthy (not for the
    vertical, 0.0), so the spline is recomputed here with set_theta; and MCEqRun caches the integration path without
    checking the atmosphere, so that cache is cleared. Checked: ~6.9x more muons at 1 GeV at 12 km than at sea level.

    Args:
        mceq_run (MCEqRun): The run.
        h_obs_cm (float): Altitude [cm].
    """
    mceq_run.density_model.set_h_obs(h_obs_cm)
    mceq_run.density_model.set_theta(mceq_run.density_model.theta_deg)
    mceq_run.integration_path = None


def _new_mceq_run(interaction_model=DEFAULT_INTERACTION_MODEL):
    """An MCEqRun (vertical, Hillas-Gaisser H3a) with the pipeline's settings: quiet, MCEQ_DB_FNAME database. Called
    by ResponseMatrix.build and the direct solver of _build_template."""
    import crflux.models as pm
    from MCEq.core import MCEqRun
    import MCEq

    MCEq.config.debug_level = 0
    if MCEQ_DB_FNAME is not None:
        MCEq.config.mceq_db_fname = MCEQ_DB_FNAME
    return MCEqRun(interaction_model=interaction_model, primary_model=(pm.HillasGaisser2012, "H3a"), theta_deg=0.0)


# --- Response matrix --------------------------------------------------------------------------------------------------

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
        """crflux's per-nucleus hook: unused (the injection goes through p_and_n_flux), zero."""
        return np.zeros_like(np.atleast_1d(np.asarray(E, dtype=float)))

    def p_and_n_flux(self, E):
        """crflux's hook read by MCEq: (proton fraction, protons, neutrons); the first (unused by MCEq) is the
        proton array again. Records the grid (record mode) or returns the unit injection."""
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
        return p, p, n


class ResponseMatrix:
    """
    Secondary intensities at each zenith angle per unit primary nucleon flux, for one atmosphere (interaction model,
    observation altitude, zenith quadrature, MCEq database):

        I[species](theta_a, E) = sum_j  R[species][a, E, 0, j] * phi_p(E_j)  +  R[species][a, E, 1, j] * phi_n(E_j),

    with phi_p, phi_n the primary model's p_and_n_flux on MCEq's nucleon grid E_j (crflux units, as MCEq takes
    them). MCEq's cascade equations are linear in the initial condition, so this reproduces a full solve to
    numerical precision; it is checked against a direct solve of the baseline when built. Nuclei enter through
    p_and_n_flux (superposition, as in MCEq itself) after crflux has applied the per-nucleus cutoff Z * Rc, so
    composition, spectral shape and geomagnetic cutoff all live in the primary vector and need no new matrix.

    The angles are kept (not only their weighted sum) for the zenith distribution of the templates; the flux
    crossing a horizontal plane is _zenith_sum of the intensities.

    Cost: one MCEq solve (all zenith angles) per nucleon type and energy cell, ~2 x 100, once per atmosphere;
    cached in the template directory and in memory. Files of the earlier, zenith-summed form have another name
    (no "_zen") and are not read.

    Attributes:
        e_grid (np.ndarray): MCEq energy grid of the secondaries [GeV].
        e_in (np.ndarray): MCEq nucleon grid of the primaries [GeV].
        response (dict): species -> R, shape (n_angles, n_e, 2, n_in).
        angles_deg, weights (np.ndarray): Zenith angles [deg] and their quadrature weights.
        meta (dict): Atmosphere and validation.
    """

    def __init__(self, e_grid, e_in, response, meta=None, angles_deg=DEFAULT_ANGLES_DEG,
                 weights=DEFAULT_ANGLE_WEIGHTS):
        self.e_grid = np.asarray(e_grid, dtype=float)
        self.e_in = np.asarray(e_in, dtype=float)
        self.response = {key: np.asarray(val, dtype=float) for key, val in response.items()}
        self.angles_deg = np.asarray(angles_deg, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.meta = dict(meta or {})

    def apply(self, model_instance):
        """MCEq intensities per zenith angle (raw units, on e_grid) for a crflux primary model, as a matrix
        product (see apply_nucleons)."""
        _, p, n = model_instance.p_and_n_flux(self.e_in)
        return self.apply_nucleons(p, n)

    def apply_nucleons(self, p, n):
        """MCEq intensities for proton and neutron fluxes on e_in (crflux units): species -> (n_angles, n_e), raw
        units, on e_grid."""
        p = np.broadcast_to(np.asarray(p, dtype=float), self.e_in.shape)
        n = np.broadcast_to(np.asarray(n, dtype=float), self.e_in.shape)
        return {key: R[..., 0, :] @ p + R[..., 1, :] @ n for key, R in self.response.items()}

    @staticmethod
    def filename(interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, n_angles=len(DEFAULT_ANGLES_DEG)):
        """Cache file name of the matrix of one atmosphere: interaction model, altitude, zenith angles, database."""
        db = "default" if MCEQ_DB_FNAME is None else os.path.splitext(MCEQ_DB_FNAME)[0]
        return f"Response_{interaction_model}_{h_obs_km or 0.0}km_{n_angles}ang_zen_{db}.npz"

    def save(self, filepath):
        """Writes the matrix to an .npz file (arrays, R__<species> blocks, meta__<key> entries)."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {"e_grid": self.e_grid, "e_in": self.e_in, "angles_deg": self.angles_deg,
                   "angle_weights": self.weights}
        payload.update({f"R__{key}": val for key, val in self.response.items()})
        payload.update({f"meta__{key}": np.asarray(val) for key, val in self.meta.items()})
        np.savez_compressed(filepath, **payload)

    @classmethod
    def load(cls, filepath):
        """Reads a matrix written by save."""
        with np.load(filepath, allow_pickle=False) as data:
            response = {k.split("__", 1)[1]: data[k] for k in data.files if k.startswith("R__")}
            meta = {k.split("__", 1)[1]: data[k].item() for k in data.files if k.startswith("meta__")}
            return cls(data["e_grid"], data["e_in"], response, meta, data["angles_deg"], data["angle_weights"])

    @classmethod
    def build(cls, interaction_model=DEFAULT_INTERACTION_MODEL, h_obs_km=None, angles=DEFAULT_ANGLES_DEG,
              weights=DEFAULT_ANGLE_WEIGHTS, validate=True, verbose=True):
        """Runs MCEq once per primary nucleon type and energy cell (all zenith angles each).

        Raises:
            RuntimeError: If MCEq does not set its initial condition through p_and_n_flux, or if the matrix does
                not reproduce a direct solve of the baseline within RESPONSE_VALIDATION_RTOL (use solver='direct').
        """
        run = _new_mceq_run(interaction_model)
        h_obs_cm = _h_obs_cm(h_obs_km)

        recorder = _NucleonInjection()
        run.set_primary_model(recorder)
        if recorder.energies is None:
            raise RuntimeError("MCEq did not evaluate p_and_n_flux in set_primary_model: the response matrix cannot "
                               "be built with this MCEq version; use solver='direct'.")
        e_in = recorder.energies
        n_in = e_in.size
        response = {key: np.zeros((len(angles), run.e_grid.size, 2, n_in)) for key in MCEQ_SPECIES}
        if verbose:
            print(f"Building MCEq response matrix ({interaction_model}, h_obs={h_obs_km or 0.0} km): "
                  f"2 x {n_in} primary cells, {len(angles)} zenith angles each.")
        for j in range(n_in):
            for nucleon in (0, 1):
                run.set_primary_model(_NucleonInjection(e_in, j, nucleon))
                _set_h_obs(run, h_obs_cm)
                intensity = _zenith_resolved_secondaries(run, angles)
                for key in MCEQ_SPECIES:
                    response[key][:, :, nucleon, j] = intensity[key]
            if verbose and (j + 1) % 20 == 0:
                print(f"  {j + 1}/{n_in} cells")

        matrix = cls(run.e_grid, e_in, response, meta={
            "interaction_model": interaction_model, "h_obs_km": float(h_obs_km or 0.0), "n_angles": len(angles),
            "mceq_db": str(MCEQ_DB_FNAME)}, angles_deg=angles, weights=weights)
        if validate:
            matrix._validate(run, h_obs_cm, angles, weights)
            if verbose:
                print(f"  checked against a direct solve: max relative difference "
                      f"{matrix.meta['validation_max_rel_diff']:.2e}")
        return matrix

    def _validate(self, run, h_obs_cm, angles, weights):
        """Compares apply() with a direct solve of the H3a baseline at 5 GV, angle by angle; stores the largest
        difference."""
        import crflux.models as pm

        model = pm.HillasGaisser2012("H3a", geomagnetic_cutoff=5.0)
        run.set_primary_model(model)
        _set_h_obs(run, h_obs_cm)
        direct = _zenith_resolved_secondaries(run, angles)
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
        """The response matrix of one atmosphere: from memory, from <template_dir>/Response_*.npz, or built and saved
        there (_cached). Called by _build_template (solver 'response')."""
        template_dir = config.flux_template_dir() if template_dir is None else template_dir
        path = os.path.join(template_dir, cls.filename(interaction_model, h_obs_km, len(angles)))
        return _cached(_RESPONSE_CACHE, path, path, cls.load,
                       lambda: cls.build(interaction_model, h_obs_km, angles, weights), force_rebuild=force_rebuild)


# --- Low-energy extension of the templates ----------------------------------------------------------------------------

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
        np.ndarray: dPhi/dE [n cm^-2 s^-1 GeV^-1] (the fit is in MeV: E converted, result per MeV x 1e3).
    """
    ln_e = np.log(np.asarray(energy_gev, dtype=float) * 1e3)
    phi = 1.006e-6 * np.exp(-0.35 * ln_e ** 2 + 2.1451 * ln_e)
    if evaporation:
        phi = phi + 1.011e-3 * np.exp(-0.4106 * ln_e ** 2 - 0.667 * ln_e)
    return phi * 1e3


def proton_equilibrium_shape(energy_gev, mean_excitation_ev=85.7):
    """Shape of the proton flux in ionisation equilibrium, dPhi/dE = Q(E) / S(E).

    A charged particle produced at rate q(E') per gram and slowed by ionisation has, in equilibrium,
    dPhi/dE = (1 / S(E)) * Q(E), with Q(E) = integral_E^inf q(E') dE' the production of protons above E. Above a few
    tens of MeV protons and neutrons come out of the same nuclear cascades in about equal numbers, so q is taken with
    the shape of the cascade term of the sea-level neutron spectrum (gordon_neutron_spectrum, which in equilibrium
    is proportional to the neutron production): Q is then an erfc in ln E. S is the Bethe stopping power (air,
    I = 85.7 eV, no shell or density corrections). Nuclear absorption of the protons and evaporation protons (low
    energy, Coulomb barrier) are neglected: the shape is meant below the matching energy only.

    In the code: masses in GeV; inv_stopping is 1 / S up to a constant; the cascade term of the Gordon fit is
    exp(-b u^2 + c u) with u = ln(E / MeV), b = 0.35, c = 3.1451, so Q is an erfc in u (up to a constant).

    Result: dPhi/dE ~ E^0.75 below ~10 MeV, maximum near 100 MeV, slope ~ -1.8 at 1 GeV; with the normalisation at
    1 GeV taken from MCEq, p/n ~ 1 % at 10 MeV, ~15 % at 100 MeV, ~60 % at 300 MeV (for p/n = 2 at 1 GeV).

    Args:
        energy_gev (array_like): Kinetic energy [GeV].
        mean_excitation_ev (float, optional): Mean excitation energy of the medium [eV]. Defaults to 85.7 (air).

    Returns:
        np.ndarray: Unnormalised shape (arbitrary units).
    """
    e = np.asarray(energy_gev, dtype=float)
    m_p, m_e = 0.93827, 0.51100e-3
    gamma = 1.0 + e / m_p
    beta2 = 1.0 - 1.0 / gamma ** 2
    log_term = np.log(2.0 * m_e * 1e9 * beta2 * gamma ** 2 / mean_excitation_ev) - beta2
    inv_stopping = beta2 / np.clip(log_term, 1e-3, None)
    b, c = 0.35, 3.1451
    production_above = erfc(np.sqrt(b) * (np.log(e * 1e3) - c / (2.0 * b)))
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
# Zenith shape of the nucleons where MCEq is not used: MCEq has no angular deflection and is not meant for nucleons
# below a few GeV. Below e_low_gev the intensity follows I ~ cos^cos_power(theta), measured at ground level for
# 10-170 MeV neutrons: n = 2.6 +- 0.2 (Moser et al., ICRC 2005, SH 3.5; a neutron scatter camera found cos^2.7 at
# 0.2-10 MeV, OSTI 1147482); taken for protons too (same cascades). Above e_high_gev (the LOW_E_SPLICE e_match_gev,
# where the spectrum is MCEq's) the shape is MCEq's; in between the two normalised shapes are mixed with a weight
# rising smoothly in log E (_zenith_shapes). MCEq gives ~cos^12 for nucleons above 10 GeV (slant absorption).
LOW_E_ZENITH = {
    'neutron': {'cos_power': 2.6, 'e_low_gev': 0.17, 'e_high_gev': 7.0},
    'proton': {'cos_power': 2.6, 'e_low_gev': 0.17, 'e_high_gev': 7.0},
}

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
        """Integral of the polynomial exponent b0 + b1 x + b2 x^2 + b3 x^3."""
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
    Zenith-integrated MCEq fluxes -> template fluxes: the MCEq grid extended down to LOW_E_GRID['e_min_gev'], with the
    low-energy correction of every LOW_E_SPLICE species. Shared by both solvers (direct MCEq solve and response
    matrix), applied to every template (steady and, per time step, transient).

    For each spliced species the shape is continued above e_bridge by shape(e_bridge) (E / e_bridge)^-gamma;
    k = geometric mean of MCEq / shape over e_match * 10^(+-blend_dex), then a log-space tanh join,
        log Phi = s log Phi_MCEq + (1 - s) log(k shape),   s = (1 + tanh(log10(E / e_match) / blend_dex)) / 2
    so flux and slope are continuous. Muons and the primary are only extended as FluxTemplate would have
    extrapolated them (unchanged where they were queried before); neutrinos are continued below
    NEUTRINO_LOW_E['e_match_gev'] by a log-normal peaking at e_peak_gev. Fluxes are floored at 1e-44 for
    FluxTemplate's log-space interpolator (and the MCEq fluxes at 1e-300 before the logs: a decayed transient may be
    exactly 0). For the species in CUTOFF_SCALING['species'] the low-energy shape is
    multiplied by cutoff_scaling_factor(geomagnetic_cutoff, h_obs_km), so the splice follows the cutoff (MCEq above
    e_match is untouched). These steps are non-linear (log blend), so they come after the MCEq solution, never
    inside the response matrix.

    Args:
        e_grid (np.ndarray): MCEq energy grid [GeV].
        raw_fluxes (dict): species -> MCEq flux crossing a horizontal plane on e_grid (_zenith_sum of the
            intensities of _zenith_resolved_secondaries or ResponseMatrix.apply_nucleons).
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
        flux = np.clip(flux, 1e-300, None)
        mceq_ext = log_interp1d(e_grid, flux)(e_ext)
        if key in NEUTRINO_LOW_E['species']:
            c = NEUTRINO_LOW_E
            em, ep = c['e_match_gev'], c['e_peak_gev']
            win = np.abs(np.log10(e_grid / em)) <= c['fit_dex']
            if win.sum() < 2:
                raise ValueError(f"{key}: only {win.sum()} MCEq grid points around {em} GeV; "
                                 "change NEUTRINO_LOW_E e_match_gev or fit_dex.")
            slope = np.polyfit(np.log(e_grid[win]), np.log(flux[win]), 1)[0]
            a = max(-slope, 0.0) / (2.0 * np.log(em / ep))
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


def _zenith_shapes(e_grid, intensities, weights, cos_zenith, e_ext):
    """Angular shape of each species on the template energy grid: g_i(E) = I_i(E) / sum_j w_j I_j(E) at the
    quadrature angles (see the module docstring). Called by _build_template, next to _postprocess_secondaries.

    MCEq's shape is used where MCEq is: for the LOW_E_ZENITH species (nucleons) it is mixed with the measured
    low-energy shape cos^n (normalised the same way), g = s g_MCEq + (1 - s) g_low, with s = 3u^2 - 2u^3,
    u = log(E / e_low) / log(e_high / e_low) clipped to [0, 1]; the mixture is normalised too. For the other species
    the shape is held below their matching energy (NEUTRINO_LOW_E['e_match_gev'] for neutrinos, the bottom of the
    MCEq grid for muons). Energies where MCEq gives no flux take its shape at the nearest energy that has one (an
    isotropic intensity if none has).

    Args:
        e_grid (np.ndarray): MCEq energy grid [GeV].
        intensities (dict): species -> intensities (n_angles, len(e_grid)) (_zenith_resolved_secondaries or
            ResponseMatrix.apply_nucleons).
        weights (np.ndarray): Quadrature weights of the angles.
        cos_zenith (np.ndarray): cos of the quadrature angles.
        e_ext (np.ndarray): Template grid (_postprocess_secondaries): e_grid extended downwards.

    Returns:
        dict: species -> g, shape (n_angles, len(e_ext)).
    """
    e_grid = np.asarray(e_grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    cos_zenith = np.asarray(cos_zenith, dtype=float)
    on_grid = np.clip(np.searchsorted(e_grid, e_ext), 0, e_grid.size - 1)
    index = np.arange(e_grid.size)
    shapes = {}
    for key, intensity in intensities.items():
        intensity = np.clip(np.asarray(intensity, dtype=float), 0.0, None)
        total = _zenith_sum(intensity, weights)
        valid = np.nonzero(total > 0)[0]
        if valid.size == 0:
            g_mceq = np.full((weights.size, e_ext.size), 1.0 / np.sum(weights))
        else:
            above = valid[np.clip(np.searchsorted(valid, index), 0, valid.size - 1)]
            below = valid[np.clip(np.searchsorted(valid, index) - 1, 0, valid.size - 1)]
            nearest = np.where(np.abs(below - index) < np.abs(above - index), below, above)
            g = intensity[:, nearest] / total[nearest]
            e_hold = NEUTRINO_LOW_E['e_match_gev'] if key in NEUTRINO_LOW_E['species'] else e_grid[0]
            j_hold = 0 if key in LOW_E_ZENITH else min(int(np.searchsorted(e_grid, e_hold)), e_grid.size - 1)
            g_mceq = g[:, np.maximum(on_grid, j_hold)]
        if key not in LOW_E_ZENITH:
            shapes[key] = g_mceq
            continue
        c = LOW_E_ZENITH[key]
        g_low = cos_zenith ** c['cos_power'] / np.sum(weights * cos_zenith ** c['cos_power'])
        u = np.clip(np.log(e_ext / c['e_low_gev']) / np.log(c['e_high_gev'] / c['e_low_gev']), 0.0, 1.0)
        s = u * u * (3.0 - 2.0 * u)
        mixed = s * g_mceq + (1.0 - s) * g_low[:, None]
        shapes[key] = mixed / (weights @ mixed)
    return shapes


# --- Building a template ----------------------------------------------------------------------------------------------

def _build_template(name, model_factory, t_since_yr=None, interaction_model=DEFAULT_INTERACTION_MODEL,
                    h_obs_km=None, geomagnetic_cutoff=5.0, solver=None, template_dir=None):
    """
    Builds a FluxTemplate: the primary model model_factory(t) at each time of t_since_yr (model_factory(None) once
    for a steady flux) -> MCEq intensities per zenith angle -> their sum (_zenith_sum) through
    _postprocess_secondaries, and their shape (_zenith_shapes).

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

        e_grid, angles, weights = matrix.e_grid, matrix.angles_deg, matrix.weights

        def intensities(source):
            """Intensities per angle of one PrimarySource, from the response matrix, and its primary flux."""
            p, n = nucleon_spectra(source, matrix.e_in, geomagnetic_cutoff)
            return matrix.apply_nucleons(p, n), lambda E: total_nucleus_flux(source, E, geomagnetic_cutoff)
    else:
        mceq_run = _new_mceq_run(interaction_model)
        reference_h_obs_cm = _h_obs_cm(h_obs_km)
        e_grid, angles, weights = mceq_run.e_grid, DEFAULT_ANGLES_DEG, DEFAULT_ANGLE_WEIGHTS

        def intensities(source):
            """Intensities per angle of one PrimarySource, from a full MCEq solve, and its primary flux."""
            model_instance = _SourceAsCrflux(source, geomagnetic_cutoff)
            mceq_run.set_primary_model(model_instance)
            _set_h_obs(mceq_run, reference_h_obs_cm)
            return _zenith_resolved_secondaries(mceq_run, angles), model_instance.total_flux

    def solve(source):
        """Template fluxes, primary flux and zenith shapes of one PrimarySource."""
        per_angle, total_flux = intensities(source)
        e_ext, fluxes, primary = _postprocess_secondaries(
            e_grid, {key: _zenith_sum(v, weights) for key, v in per_angle.items()}, total_flux,
            geomagnetic_cutoff=geomagnetic_cutoff, h_obs_km=h_obs_km)
        return e_ext, fluxes, primary, _zenith_shapes(e_grid, per_angle, weights, np.cos(np.deg2rad(angles)), e_ext)

    if t_since_yr is None:
        e_ext, fluxes, primary, shapes = solve(model_factory(None))
    else:
        steps, primary, step_shapes = [], [], []
        for t_yr in t_since_yr:
            e_ext, fluxes_t, primary_t, shapes_t = solve(model_factory(t_yr))
            steps.append(fluxes_t)
            primary.append(primary_t)
            step_shapes.append(shapes_t)
        fluxes = {key: np.array([f[key] for f in steps]) for key in MCEQ_SPECIES}
        shapes = {key: np.array([g[key] for g in step_shapes]) for key in MCEQ_SPECIES}

    return FluxTemplate(
        name, e_ext,
        {
            "primary": np.array(primary) * 1e-4,
            **{key: fluxes[key] for key in MCEQ_SPECIES},
        },
        t_since_yr=t_since_yr,
        meta={"h_obs_km": h_obs_km, "geomagnetic_cutoff": geomagnetic_cutoff, "interaction_model": interaction_model},
        zenith={"cos_zenith": np.cos(np.deg2rad(angles)), "weights": weights, "shapes": shapes},
    )


# --- FluxTemplate -----------------------------------------------------------------------------------------------------


class FluxTemplate:
    """The fluxes of every species at the site for one primary flux and one site: an energy spectrum per species
    (steady, for baselines) or a (time since the event, energy) grid (transient, for events: the excess over the
    baseline, decaying as t_since_yr grows).

    Made by get_or_build from a spec (built with MCEq on a cache miss, see _build_template), or loaded from a file.
    Evaluating one (flux) needs neither MCEq nor crflux. Units: see the module docstring.

    Attributes:
        name (str): Template name (the spec's label, cutoff and altitude).
        energy_gev (np.ndarray): Energy grid [GeV].
        t_since_yr (np.ndarray or None): Time axis [yr since the event]; None for a steady template.
        is_steady (bool): No time axis.
        meta (dict): The site it was built for: 'h_obs_km', 'geomagnetic_cutoff', 'interaction_model' (checked by
            FluxHistory against its own site).
        species (list[str]): Species stored.
        cos_zenith, zenith_weights (np.ndarray or None): Quadrature nodes mu_i = cos(theta_i) and weights of the
            zenith shapes (None without zenith data).
    """

    def __init__(self, name, energy_gev, species_grids, t_since_yr=None, meta=None, zenith=None):
        """
        Args:
            name (str): Template name.
            energy_gev (array_like): Energy grid [GeV], shape (n_e,).
            species_grids (dict): species -> flux, shape (n_e,) (steady) or (n_t, n_e) (transient).
            t_since_yr (array_like, optional): Time axis [yr since the event]. Defaults to None (steady).
            meta (dict, optional): Site of the template. Defaults to None.
            zenith (dict, optional): Zenith distribution: 'cos_zenith' (n_angles,), 'weights' (n_angles,) and
                'shapes' {species: g, shape (n_angles, n_e) or (n_t, n_angles, n_e), sum_i w_i g_i = 1} (see the
                module docstring). Made by _build_template; needed by Paleodetector for the cosmic-ray species.
                Defaults to None.
        """
        self.name = name
        self.meta = dict(meta or {})
        self.energy_gev = np.asarray(energy_gev, dtype=float)
        self.t_since_yr = None if t_since_yr is None else np.asarray(t_since_yr, dtype=float)
        self.is_steady = self.t_since_yr is None

        self._grids = {}
        self._interp = {}
        for species, grid in species_grids.items():
            grid = np.asarray(grid, dtype=float)
            self._grids[species] = grid
            self._interp[species] = self._build_interpolator(grid)

        self.cos_zenith = self.zenith_weights = None
        self._zenith_shapes = {}
        if zenith is not None:
            self.cos_zenith = np.asarray(zenith['cos_zenith'], dtype=float)
            self.zenith_weights = np.asarray(zenith['weights'], dtype=float)
            n_angles = self.cos_zenith.size
            expected = (n_angles, self.energy_gev.size) if self.is_steady else \
                (self.t_since_yr.size, n_angles, self.energy_gev.size)
            for species, g in zenith['shapes'].items():
                g = np.asarray(g, dtype=float)
                if g.shape != expected:
                    raise ValueError(f"Template '{name}': zenith shape of {species} has shape {g.shape}, expected "
                                     f"{expected}.")
                self._zenith_shapes[species] = g

    def _build_interpolator(self, grid):
        """Interpolator of one species: log flux linear in log energy (steady: _core.log_interp1d) or in (time, log
        energy) (transient: RegularGridInterpolator); both extrapolate beyond the grid."""
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
        """Species stored in the template."""
        return list(self._grids.keys())

    @property
    def has_zenith(self):
        """Whether the template has zenith shapes (templates built before they were kept do not)."""
        return self.cos_zenith is not None

    def zenith_shape(self, species, energy_gev, t_since_yr=None):
        """Zenith shape g_i of one species at the quadrature angles (see the module docstring): the intensity at
        angle i is flux * g_i. Linear in log energy (and time) between the grid points, held at the edges.

        Args:
            species (str): A species with a zenith shape.
            energy_gev (array_like): Energies [GeV].
            t_since_yr (array_like, optional): Times since the event [yr], broadcastable against energy_gev;
                required for a transient template.

        Returns:
            np.ndarray: g, shape (*broadcast shape of the inputs, n_angles).

        Raises:
            ValueError: If the template has no zenith shape for the species, or is transient and t_since_yr is
                missing.
        """
        if species not in self._zenith_shapes:
            raise ValueError(f"Template '{self.name}' has no zenith distribution for {species}: it was built before "
                             "templates kept it, or by hand. Rebuild it (FluxTemplate.get_or_build(spec, "
                             "force_rebuild=True), or delete its .npz file) or pass zenith= to FluxTemplate.")
        g = self._zenith_shapes[species]
        log_grid = np.log10(self.energy_gev)
        log_e = np.clip(np.log10(np.asarray(energy_gev, dtype=float)), log_grid[0], log_grid[-1])
        if self.is_steady:
            j = np.clip(np.searchsorted(log_grid, log_e) - 1, 0, log_grid.size - 2)
            f = ((log_e - log_grid[j]) / (log_grid[j + 1] - log_grid[j]))[..., None]
            return np.moveaxis(g[:, j], 0, -1) * (1.0 - f) + np.moveaxis(g[:, j + 1], 0, -1) * f
        if t_since_yr is None:
            raise ValueError(f"Template '{self.name}' is transient: t_since_yr is required.")
        t_b, e_b = np.broadcast_arrays(np.clip(np.asarray(t_since_yr, dtype=float), self.t_since_yr[0],
                                               self.t_since_yr[-1]), log_e)
        interp = RegularGridInterpolator((self.t_since_yr, log_grid), np.moveaxis(g, 1, -1), method='linear')
        return interp(np.stack([t_b.ravel(), e_b.ravel()], axis=-1)).reshape(*t_b.shape, -1)

    def flux(self, species, energy_gev, t_since_yr=None):
        """Flux of one species at energies (and, for a transient template, times since the event).

        Args:
            species (str): A species of the template (an absent one gives zeros).
            energy_gev (array_like): Energies [GeV].
            t_since_yr (array_like, optional): Times since the event [yr], broadcastable against energy_gev; required
                for a transient template, ignored for a steady one. Negative times (before the event) give 0.

        Returns:
            np.ndarray: Flux, shape of the broadcast inputs.

        Raises:
            ValueError: If the template is transient and t_since_yr is missing.
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
        """Writes the template to an .npz file (energy_gev, t_since_yr, species__<name> grids, meta__<key>, and the
        zenith data: zenith__cos, zenith__weights, zshape__<name>)."""
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
        if self.has_zenith:
            payload['zenith__cos'] = self.cos_zenith
            payload['zenith__weights'] = self.zenith_weights
            for species, g in self._zenith_shapes.items():
                payload[f'zshape__{species}'] = g
        np.savez_compressed(filepath, **payload)

    @classmethod
    def load(cls, filepath):
        """Reads a template written by save (no MCEq needed); files of the old code, with the time axis under
        't_since_kyr' (already in years), are read too."""
        with np.load(filepath, allow_pickle=False) as data:
            energy_gev = data['energy_gev']
            key = next((k for k in ('t_since_yr', 't_since_kyr') if k in data), None)
            t_since_yr = data[key] if key else None
            species_grids = {
                key.split('__', 1)[1]: data[key]
                for key in data.files if key.startswith('species__')
            }
            meta = {key.split('__', 1)[1]: data[key].item() for key in data.files if key.startswith('meta__')}
            zenith = None
            if 'zenith__cos' in data.files:
                zenith = {'cos_zenith': data['zenith__cos'], 'weights': data['zenith__weights'],
                          'shapes': {key.split('__', 1)[1]: data[key] for key in data.files
                                     if key.startswith('zshape__')}}
        meta = {k: (None if v == '' else v) for k, v in meta.items()}
        name = os.path.splitext(os.path.basename(filepath))[0]
        return cls(name, energy_gev, species_grids, t_since_yr=t_since_yr, meta=meta, zenith=zenith)

    @classmethod
    def get_or_build(cls, params, template_dir=None, force_rebuild=False):
        """
        The template of a spec dict: a 'Baseline' (default, {} = H3a), an analytical model ({"kind": "SN",
        "distance_pc": 100}, see register_flux_model) or a 'Tabulated' flux (see the primary-flux section), plus the
        site keys 'h_obs_km' (default sea level), 'geomagnetic_cutoff' [GV] (default 5.0), 'interaction_model' and
        'solver' ('response' or 'direct', default DEFAULT_SOLVER; not part of the name: both give the same template).
        In a FluxHistory, give the site once to the history instead.

        Caching (_cached): templates are kept in memory for the session. Templates built with the response matrix
        are rebuilt from it on demand (a fraction of a second), so only the matrices (Response_*.npz) live on disk,
        unless SAVE_RESPONSE_TEMPLATES is True; 'direct' templates are saved to and loaded from
        <template_dir>/<name>.npz (a saved template without zenith data is built again). Called by FluxHistory for
        its baseline, events and excursion variants.
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

        def load(path):
            """The saved template, or None (rebuild) if it has no zenith data."""
            template = cls.load(path)
            return template if template.has_zenith else None

        return _cached(
            _TEMPLATE_CACHE, (os.path.abspath(template_dir), name, interaction_model),
            os.path.join(template_dir, name + ".npz"), load,
            lambda: _build_template(name, factory, t_since_yr=times, interaction_model=interaction_model,
                                    h_obs_km=h_obs_km, geomagnetic_cutoff=geomagnetic_cutoff, solver=solver,
                                    template_dir=template_dir),
            use_disk=on_disk, force_rebuild=force_rebuild)

    @classmethod
    def from_primary_spectrum(cls, energy_gev, flux, composition=None, t_since_yr=None, name="primary",
                              template_dir=None, force_rebuild=False, **site):
        """Shorthand for get_or_build(primary_spectrum_spec(...)): a 'Tabulated' template from one energy grid."""
        spec = primary_spectrum_spec(energy_gev, flux, composition=composition, t_since_yr=t_since_yr, name=name)
        return cls.get_or_build({**spec, **site}, template_dir=template_dir, force_rebuild=force_rebuild)


# Keys of a spec that describe the site, not the primary flux (FluxHistory gives them once to every template).
SITE_KEYS = ("h_obs_km", "geomagnetic_cutoff", "interaction_model")


# --- FluxHistory ------------------------------------------------------------------------------------------------------

class FluxHistory:
    """The cosmic-ray flux at the site over time: a baseline template, discrete events and geomagnetic excursions,
    summed on demand. The flux-side analogue of the overburden history: a small spec goes in, one object answers
    every query of the detector (flux, get_map).

    Timeline: absolute time in kyr, 0 = present, negative = past (see the module docstring). Every template gets the
    site of the history (h_obs_km, geomagnetic_cutoff, interaction_model).

    Attributes:
        site (dict): The site given to every template.
        baseline (FluxTemplate): The baseline.
        events (list[dict]): 'template', 'spec', 'start_time_kyr', 'weight', 'label' of each event.
        excursions (list[dict]): 'start_time_kyr', 'end_time_kyr', 'geomagnetic_cutoff', 'label' of each excursion.
        template_dir (str): Where templates and response matrices are cached.
    """

    def __init__(self, baseline=None, events=None, excursions=None, template_dir=None, h_obs_km=None,
                 geomagnetic_cutoff=None, interaction_model=None):
        """
        Args:
            baseline (dict, FluxTemplate or str, optional): A spec (built on demand), a FluxTemplate or the name of a
                saved template. Defaults to None (Hillas-Gaisser H3a, see FluxTemplate.get_or_build).
            events (list[dict], optional): Each with 'start_time_kyr' and 'template' (as in add_event), optionally
                'weight' and 'label'. Defaults to None.
            excursions (list[dict], optional): Each with 'start_time_kyr', 'end_time_kyr', 'geomagnetic_cutoff',
                optionally 'label' (see add_excursion). Defaults to None.
            template_dir (str, optional): Where templates are loaded and saved. Defaults to None
                (config.flux_template_dir(): <data root>/flux_data).
            h_obs_km, geomagnetic_cutoff, interaction_model (optional): The site, given once and applied to the
                baseline, every event and the excursion variants. One not given here is taken from the baseline spec
                if it has it (older notebooks), else the get_or_build default. An entry with a different value raises.

        Raises:
            KeyError: If an event lacks 'start_time_kyr' or 'template'.
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
        """The spec with the site of the history; raises if it states a different site. Called by __init__ and
        add_event."""
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
        """Raises if a built template's site (its meta) differs from the history's. Called by _with_site and
        _resolve."""
        for key, value in self.site.items():
            if key in template.meta and template.meta[key] != value:
                raise ValueError(f"{label}: template {template.name!r} was built with {key} = "
                                 f"{template.meta[key]!r}, the flux history has {value!r}.")

    def _resolve(self, template):
        """The FluxTemplate of a baseline or event: a FluxTemplate as it is, a spec dict built or taken from the cache
        (FluxTemplate.get_or_build), or the name of a template saved in template_dir.

        Raises:
            FileNotFoundError: If a name has no saved template.
            TypeError: For any other kind of argument.
        """
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
        """Adds a flux event: from start_time_kyr on, weight times the transient template is added to the flux (events
        superpose, even the same template at nearby times).

        Args:
            start_time_kyr (float): Start of the event [kyr, 0 = present, negative = past].
            template (FluxTemplate, str or dict): A built template, the name of a saved one, or a spec dict (e.g.
                {"kind": "SN", "distance_pc": 200}, built on first use). Only a spec dict can follow excursions.
            weight (float, optional): Multiplies the template. Defaults to 1.
            label (str, optional): Name used in messages. Defaults to None.

        Returns:
            FluxTemplate: The template of the event.
        """
        template = self._with_site(template, label or f"event at {start_time_kyr} kyr")
        resolved = self._resolve(template)
        self.events.append({
            "template": resolved, "spec": template, "start_time_kyr": float(start_time_kyr),
            "weight": weight, "label": label, "_cutoff_variants": {},
        })
        return resolved

    def add_excursion(self, start_time_kyr, end_time_kyr, geomagnetic_cutoff, label=None):
        """Adds a geomagnetic excursion: over [start_time_kyr, end_time_kyr) the baseline and every event are read from
        templates built at geomagnetic_cutoff instead of the site's cutoff (a change of the existing flux, not a new
        source). MCEq has no time-dependent cutoff, so each affected template gets an independent variant at the
        excursion cutoff (_cutoff_variant, cached by name like any template); flux picks, per time, the normal or the
        excursion stack. Only a baseline or event given as a spec dict can be rebuilt at another cutoff; a pre-built
        template or a name raises, when a query falls inside the window. Where excursions overlap, the one added later
        wins.

        Args:
            start_time_kyr, end_time_kyr (float): Window [kyr, same timeline].
            geomagnetic_cutoff (float): Cutoff during the excursion [GV].
            label (str, optional): Name used in messages. Defaults to None.

        Returns:
            dict: The excursion entry.

        Raises:
            ValueError: If end_time_kyr <= start_time_kyr.
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
        """The template of a baseline or event spec rebuilt at another cutoff (the rest of the spec unchanged), kept in
        cache, the dict of that baseline or event. Called by _flux_stack during excursions.

        Raises:
            TypeError: If the baseline or event was not given as a spec dict.
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

    def _flux_stack(self, species, energy_gev, t_kyr, geomagnetic_cutoff=None, zenith=False):
        """Baseline plus every event (each at its time since the start, in years) at absolute times t_kyr, with the
        site's templates (geomagnetic_cutoff None) or their variants at an excursion cutoff. With zenith, the
        intensities at the quadrature angles (flux * zenith_shape of each template, last axis) instead of the flux.
        Called by flux and angular_flux.
        """
        if geomagnetic_cutoff is None:
            baseline = self.baseline
        else:
            baseline = self._cutoff_variant(self._baseline_spec, self._baseline_cutoff_variants, geomagnetic_cutoff)

        def term(template, t_since_yr=None):
            """Flux (or intensities, with zenith) of one template."""
            flux = template.flux(species, energy_gev, t_since_yr=t_since_yr)
            if not zenith:
                return flux
            self._check_zenith(template)
            return flux[..., None] * template.zenith_shape(species, energy_gev, t_since_yr=t_since_yr)

        shape = np.broadcast_shapes(np.shape(t_kyr), np.shape(energy_gev))
        total = term(baseline)
        total = np.broadcast_to(total, shape + total.shape[len(np.shape(energy_gev)):]).copy()
        for event in self.events:
            if geomagnetic_cutoff is None:
                template = event["template"]
            else:
                template = self._cutoff_variant(event["spec"], event["_cutoff_variants"], geomagnetic_cutoff)
            t_since_yr = (np.asarray(t_kyr, dtype=float) - event["start_time_kyr"]) * 1.e3
            total = total + event["weight"] * term(template, t_since_yr)
        return total

    def _check_zenith(self, template):
        """Raises unless the template has zenith shapes on the baseline's angles. Called by _flux_stack."""
        if not template.has_zenith:
            raise ValueError(f"Template '{template.name}' has no zenith distribution: it was built before templates "
                             "kept it, or by hand. Rebuild it (FluxTemplate.get_or_build(spec, force_rebuild=True), "
                             "or delete its .npz file) or pass zenith= to FluxTemplate.")
        if not (template.cos_zenith.shape == self.baseline.cos_zenith.shape
                and np.allclose(template.cos_zenith, self.baseline.cos_zenith)):
            raise ValueError(f"Template '{template.name}' has its zenith shapes on other angles than the baseline "
                             f"'{self.baseline.name}'; build them with the same zenith quadrature.")

    def flux(self, species, energy_gev, t_kyr, zenith=False):
        """Total flux of one species (baseline and active events, excursions swapped in over their windows).

        Args:
            species (str): Template species (MCEQ_SPECIES key or 'primary').
            energy_gev (array_like): Energies [GeV].
            t_kyr (array_like): Absolute times [kyr, 0 = present], broadcastable against energy_gev.

            zenith (bool, optional): Return the intensities at the zenith quadrature angles instead (last axis; see
                get_zenith_map). Defaults to False.

        Returns:
            np.ndarray: Flux (units: see the module docstring), shape of the broadcast inputs (with zenith, plus
            the angle axis).
        """
        total = self._flux_stack(species, energy_gev, t_kyr, zenith=zenith)

        if not self.excursions:
            return total

        t_arr = np.asarray(t_kyr, dtype=float)
        shape = np.broadcast_shapes(t_arr.shape, np.shape(energy_gev))
        t_b = np.broadcast_to(t_arr, shape)
        if zenith:
            shape, t_b = shape + total.shape[-1:], t_b[..., None]
        total = np.broadcast_to(total, shape).copy()

        for excursion in self.excursions:
            mask = (t_b >= excursion["start_time_kyr"]) & (t_b < excursion["end_time_kyr"])
            if not np.any(mask):
                continue
            excursion_total = self._flux_stack(
                species, energy_gev, t_kyr, geomagnetic_cutoff=excursion["geomagnetic_cutoff"], zenith=zenith
            )
            excursion_total = np.broadcast_to(excursion_total, shape)
            total = np.where(mask, excursion_total, total)
        return total

    def get_map(self, species, t_grid_kyr, e_grid_gev=None):
        """Flux of one species on a (time, energy) grid: the form Paleodetector reads.

        Args:
            species (str): Template species.
            t_grid_kyr (array_like): Absolute times [kyr].
            e_grid_gev (array_like, optional): Energies [GeV], any shape. Defaults to None (the baseline's grid).

        Returns:
            tuple: (t_grid_kyr, e_grid_gev, flux_map), flux_map of shape (len(t_grid_kyr), *e_grid_gev.shape); for a 1D
            energy grid ready for plt.pcolormesh(e_grid, t_grid, flux_map.T).
        """
        if e_grid_gev is None:
            e_grid_gev = self.baseline.energy_gev
        t_grid_kyr = np.asarray(t_grid_kyr, dtype=float)
        e_grid_gev = np.asarray(e_grid_gev, dtype=float)
        tt, ee = np.meshgrid(t_grid_kyr, e_grid_gev, indexing='ij')
        return t_grid_kyr, e_grid_gev, self.flux(species, ee, tt)

    def get_zenith_map(self, species, t_grid_kyr, e_grid_gev):
        """Intensities of one species at the zenith quadrature angles on a (time, energy) grid: what Paleodetector
        reads for the slanted paths through the overburden (with excursions, as flux). sum_i w_i I_i is get_map's
        flux; the flux crossing a horizontal plane per unit cos(zenith) is 2 pi mu I(mu).

        Args:
            species (str): Template species with zenith shapes (MCEQ_SPECIES key).
            t_grid_kyr (array_like): Absolute times [kyr].
            e_grid_gev (array_like): Energies [GeV], any shape.

        Returns:
            tuple: (cos_zenith, weights, intensity): the quadrature nodes mu_i and weights (n_angles,), and the
            intensities [cm^-2 s^-1 sr^-1 GeV^-1], shape (len(t_grid_kyr), *e_grid_gev.shape, n_angles).

        Raises:
            ValueError: If a template of the history has no zenith data (rebuild it) or other angles.
        """
        t_grid_kyr = np.asarray(t_grid_kyr, dtype=float)
        e_grid_gev = np.asarray(e_grid_gev, dtype=float)
        tt = t_grid_kyr.reshape(-1, *([1] * e_grid_gev.ndim))
        intensity = self.flux(species, e_grid_gev[None], tt, zenith=True)
        return self.baseline.cos_zenith, self.baseline.zenith_weights, intensity
