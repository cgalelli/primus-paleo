"""The Paleodetector class: expected track-length (or recoil-energy) spectra of one mineral sample from cosmic-ray
particles (primaries and atmospheric secondaries), atmospheric neutrinos and the radiogenic sources (spontaneous
fission and radiogenic neutrons), given its overburden and flux histories."""

import json
import os
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

from . import config
from .flux import FluxHistory, MCEQ_SPECIES
from ._core import (
    B0_U238, CAPTURE_SPECTRUM_MAX_E0_GEV, COMPUTABLE_SPECIES, DEFAULT_TRACK_SUM, DEPTH_FLOOR_MWE,
    DERIVED_SPECIES, FISSIONS_U238_G_KYR, SECONDS_PER_KYR, MIN_DEPTH_SAMPLES, MUON_SPECIES,
    NEUTRINO_MAX_GENERATIONS, NEUTRINO_MAX_INTERACTION_FRACTION, NEUTRINO_PRODUCTS, NEUTRINO_SECONDARY_PREFIX,
    NEUTRINO_SOLID_ANGLE_FACTOR, NEUTRINO_SPECIES, NEUTRON_MASS_MEV, NEUTRON_P_NODES, NU_BAR_U238_SF,
    PROTON_MASS_MEV, RECOIL_ENERGY_BINS_MEV, SECONDARY_SOURCES, SECONDARY_WINDOW_BINS,
    SECONDARY_WINDOW_SURVIVAL, STOPPING_P_NODES, TIME_GRID_ATTENUATION_MWE, TIME_GRID_BACKBONE,
    TIME_GRID_FLUX_PROBES, TIME_GRID_MAX_PASSES, TIME_GRID_MAX_POINTS, TIME_GRID_RTOL, TOTAL_ALL_FRAGMENTS,
    TOTAL_NO_LIGHT_IONS, TRACK_GROUPS, TRACK_GROUPS_KEY, TRACK_LENGTH_BINS_NM, TRACK_SUMS, TRUNCATION_FRACTION,
    U238_ABUNDANCE, WATT_A_MEV, ELEMENT_SYMBOLS, fragment_za, predict_track_groups, track_group,
    WATT_B_PER_MEV, _ENERGY_EDGE_RTOL, OBSERVABLE_TRACKS, OBSERVABLE_RECOIL_ENERGY, OBSERVABLES,
    RADIOGENIC_SOURCES, FISSION_KEYS, ENERGY_CELL_NODES, COS_ZENITH_NODES, _depth_cdf_rows,
    GEANT4_DEFAULT_TARGET_LENGTH_M, GEANT4_HARDCODED_DENSITIES_G_CM3,
    _geant4_energy_edges, _list_geant4_energy_files, _loglog_interp_rows,
    _rebin_density, _resolve_x_bins, _restrict_edges,
    is_light_ion, normalize_fragment_names,
)

__all__ = [
    'Paleodetector',
]


# --- Speed helpers: caches and a compiled loop. None of them changes a number of the results. ------------------------
# numba is optional: without it _secondary_cascade uses its numpy loop instead of _window_source_numba.

def _element_symbol(z):
    """Element symbol of an atomic number (names the fission fragments, see _fission_fragment_recoils)."""
    return ELEMENT_SYMBOLS[int(z) - 1]


# Process-wide cache of what is read from the Geant4 files: per-fragment recoil histograms of the mineral runs, the
# StdRock production rows, the range tables. Keys contain each file's modification time and size, so an entry is
# never reused after a file changes. Shared by every Paleodetector of the session (e.g. one per sample age) and sent
# to the integrate_all_particles workers, which send back what they read. Least recently used entries are dropped
# beyond GEANT4_FILE_CACHE_MAX_BYTES; 0 disables the cache.
GEANT4_FILE_CACHE_MAX_BYTES = 2_000_000_000
_FILE_CACHE = OrderedDict()
_FILE_CACHE_SIZE = [0]


def _nbytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_nbytes(v) for v in value.values()) + 64 * len(value)
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(v) for v in value) + 8 * len(value)
    return 64


def _file_stamp(path):
    """(path, mtime_ns, size) of a file, or None if it does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (path, st.st_mtime_ns, st.st_size)


def _cache_get(key):
    if key in _FILE_CACHE:
        _FILE_CACHE.move_to_end(key)
        return True, _FILE_CACHE[key]
    return False, None


def _cache_put(key, value):
    if GEANT4_FILE_CACHE_MAX_BYTES <= 0 or key in _FILE_CACHE:
        return
    size = _nbytes(value)
    if size > GEANT4_FILE_CACHE_MAX_BYTES:
        return
    _FILE_CACHE[key] = value
    _FILE_CACHE_SIZE[0] += size
    while _FILE_CACHE_SIZE[0] > GEANT4_FILE_CACHE_MAX_BYTES and _FILE_CACHE:
        _, old = _FILE_CACHE.popitem(last=False)
        _FILE_CACHE_SIZE[0] -= _nbytes(old)


def clear_geant4_file_cache():
    """Empties the process-wide cache of Geant4 file contents (e.g. to free memory)."""
    _FILE_CACHE.clear()
    _FILE_CACHE_SIZE[0] = 0


def _mineral_run_histograms(filepath):
    """Per-fragment recoil-energy histograms of one mineral run file, cached.

    Same reading and the same np.histogram calls as before (rows of each fragment in file order), so the counts are
    identical; they depend only on the file, while the flux weights they multiply change from call to call.

    Returns:
        dict or None: None if the file is missing or empty; else {'all': {fragment: counts}, 'peak': {fragment:
        counts of the rows with remaining energy 0, or None if there are none}, 'elastic': {same, other rows}}.
        Fragments are the non-empty normalised names (normalize_fragment_names).
    """
    stamp = _file_stamp(filepath)
    if stamp is None:
        return None
    key = ('mineral_hist', stamp, RECOIL_ENERGY_BINS_MEV.size, float(RECOIL_ENERGY_BINS_MEV[0]),
           float(RECOIL_ENERGY_BINS_MEV[-1]))
    hit, value = _cache_get(key)
    if hit:
        return value
    value = None
    if stamp[2] > 0:
        try:
            df = pd.read_csv(
                filepath, sep=r'\s+', header=None, usecols=[0, 2, 5],
                names=['name', 'rec_e', 'rem_e'],
                dtype={'name': str, 'rec_e': float, 'rem_e': float},
            )
        except pd.errors.EmptyDataError:
            df = None
        if df is not None:
            names = normalize_fragment_names(df['name'].values)
            rec_energies = df['rec_e'].values
            rem_energies = df['rem_e'].values
            keep = names != ''
            names, rec_energies, rem_energies = names[keep], rec_energies[keep], rem_energies[keep]
            value = {'all': {}, 'peak': {}, 'elastic': {}}
            for frag in np.unique(names):
                frag_mask = (names == frag)
                rec = rec_energies[frag_mask]
                rem = rem_energies[frag_mask]
                value['all'][frag] = np.histogram(rec, bins=RECOIL_ENERGY_BINS_MEV)[0]
                mask_peak = (rem == 0.0)
                rec_peak, rec_elastic = rec[mask_peak], rec[~mask_peak]
                value['peak'][frag] = (np.histogram(rec_peak, bins=RECOIL_ENERGY_BINS_MEV)[0]
                                       if len(rec_peak) > 0 else None)
                value['elastic'][frag] = (np.histogram(rec_elastic, bins=RECOIL_ENERGY_BINS_MEV)[0]
                                          if len(rec_elastic) > 0 else None)
    _cache_put(key, value)
    return value


def _range_table(filepath):
    """Columns of a DEDX_Z<z>_A<a>.txt range table (np.loadtxt, as before), cached."""
    stamp = _file_stamp(filepath)
    key = ('range', stamp)
    hit, value = _cache_get(key)
    if hit:
        return value
    value = np.loadtxt(filepath, skiprows=1, unpack=True)
    _cache_put(key, value)
    return value


try:
    import numba as _numba
except ImportError:
    _numba = None

if _numba is not None:
    @_numba.njit(cache=True, parallel=True)
    def _window_source_numba(A, depth, primary_cell, cell, lo_idx, hi_idx, beyond, window_top, phi_cell, weight,
                             delta, n_b, n_q, k0, active):
        """Compiled form of the (t, c) loop of _secondary_cascade: same operations on each row, same order of
        summation within each bin (np.bincount adds the weights in row order from 0, then the sum is added to A).
        The times are independent (each writes only A[t]) and run in parallel threads; only those with active[t]."""
        n_t, n_c = lo_idx.shape
        for t in _numba.prange(n_t):
            if not active[t]:
                continue
            buf = np.zeros((n_b, n_q))
            for c in range(n_c):
                lo, hi = lo_idx[t, c], hi_idx[t, c]
                if beyond[t, c] or hi <= lo:
                    continue
                buf[:, :] = 0.0
                top = window_top[t, c]
                for r in range(lo, hi):
                    b = int((depth[r] - top) / delta)
                    if b > n_b - 1:
                        b = n_b - 1
                    buf[b, cell[r]] += phi_cell[t, primary_cell[r]] * weight
                for b in range(n_b):
                    for k in range(n_q):
                        A[t, c, b, k0 + k] += buf[b, k]

def _combine_track_results(results):
    """Sum of several track results: arrays, or dicts of arrays with the same keys (nucleus='groups' or 'all';
    a key missing from some results counts as 0)."""
    results = list(results)
    if not results:
        return 0.0
    if not isinstance(results[0], dict):
        return sum(results)
    keys = list(dict.fromkeys(k for r in results for k in r))
    return {k: sum(r[k] for r in results if k in r) for k in keys}


# --- Recoil spectra, observables and quadratures (the constants are in _core: OBSERVABLES, ENERGY_CELL_NODES, ...) ----


def _output_axes(observable, x_bins, x_grid):
    """Output bin edges and fine grid of an observable; called at the start of every integrate_* function.

    tracks: x_bins in nm (default: the module default, _resolve_x_bins) and the densities on x_grid.
    recoil_energy: x_bins in MeV (default RECOIL_ENERGY_BINS_MEV) and the densities on RECOIL_ENERGY_BINS_MEV, the
    grid the recoils are histogrammed on (x_grid is ignored).

    Args:
        observable (str): 'tracks' or 'recoil_energy'.
        x_bins (array_like or None): Output bin edges.
        x_grid (array_like): Fine track-length grid [nm].

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_bins, grid).

    Raises:
        ValueError: If observable is unknown.
    """
    if observable == OBSERVABLE_TRACKS:
        return _resolve_x_bins(x_bins), np.asarray(x_grid, dtype=float)
    if observable == OBSERVABLE_RECOIL_ENERGY:
        return (RECOIL_ENERGY_BINS_MEV if x_bins is None else np.asarray(x_bins, dtype=float)), RECOIL_ENERGY_BINS_MEV
    raise ValueError(f"observable must be one of {OBSERVABLES}, got {observable!r}.")


def _recoil_spectra(fragments, t_kyr=None):
    """The recoil spectra every source produces: the common input of Paleodetector._recoil_density.

    Args:
        fragments (dict): {fragment key: recoils per g of mineral per kyr in each RECOIL_ENERGY_BINS_MEV bin (counts
            per bin, not per MeV)}, shape (T, n_recoil_bins) for a time-dependent source or (n_recoil_bins,) for a
            constant one. A 'neutron' entry is dropped (not a recoiling nucleus).
        t_kyr (np.ndarray, optional): Local times of the rows [kyr]; None for a constant source.

    Returns:
        dict: {'t_kyr': t_kyr, 'fragments': {fragment: array}}.
    """
    return {'t_kyr': t_kyr, 'fragments': {k: v for k, v in fragments.items() if k != 'neutron'}}


def _energy_cell_nodes(edges_gev, n_nodes=ENERGY_CELL_NODES):
    """Nodes and trapezoidal weights for integrals over each energy cell.

    Used by every integral over the energy cells of the Geant4 runs: Paleodetector._flux_cell_integral (bare flux:
    secondary cascade, neutrinos), Paleodetector._primary_recoils (flux times slab probability) and
    Paleodetector._radiogenic_neutron_recoils (emission spectrum). The integral of f over cell i is
    sum_j f(e[i, j]) * w[i, j].

    Args:
        edges_gev (array_like): Cell edges [GeV], shape (M,).
        n_nodes (int, optional): Linear points per cell, edges included. Defaults to ENERGY_CELL_NODES.

    Returns:
        tuple[np.ndarray, np.ndarray]: Nodes e [GeV] and weights w [GeV], both of shape (M - 1, n_nodes).
    """
    edges_gev = np.asarray(edges_gev, dtype=float)
    e_lo, e_hi = edges_gev[:-1], edges_gev[1:]
    nodes = e_lo[:, None] + (e_hi - e_lo)[:, None] * np.linspace(0.0, 1.0, n_nodes)[None, :]
    weights = np.repeat(((e_hi - e_lo) / (n_nodes - 1))[:, None], n_nodes, axis=1)
    weights[:, [0, -1]] *= 0.5
    return nodes, weights


def _product_axis(product_bins_gev):
    """Layout of the products of a cascade on one flat axis: product p occupies its energy cells at offsets[p].

    Shared by the two cascades that follow neutrons, protons (and, for neutrinos, muons) through all their
    generations: Paleodetector._secondary_cascade (atmospheric secondaries) and the neutrino path
    (Paleodetector._neutrino_run_pass, _neutrino_product_operator, integrate_atmospheric_neutrinos).

    Args:
        product_bins_gev (dict): {product: energy cell edges [GeV]}, in axis order.

    Returns:
        tuple[dict, int]: offsets {product: slice} and the total number of cells K.
    """
    offsets, k = {}, 0
    for p, edges in product_bins_gev.items():
        offsets[p] = slice(k, k + len(edges) - 1)
        k += len(edges) - 1
    return offsets, k


def _accumulate_into(store, key, value):
    """store[key] += value, creating the entry at the first value (the readers list fragments as they meet them)."""
    store[key] = store[key] + value if key in store else value


# --- Main Paleodetector Class ---
class Paleodetector:
    """Expected tracks (or recoils) in one mineral sample, from every source, given its overburden and flux histories.

    Combines the Geant4 recoil populations of the mineral (mineral runs), the depth distributions of the StdRock
    runs, the overburden history and the cosmic-ray flux history (FluxHistory). Entry points, all returning
    (bins, {source: counts per mg, ..., 'total'}) with the same keys and options:

    * integrate_all_particles: cosmic-ray primaries (mu-, mu+, neutron, proton) and the atmospheric secondary
      neutrons and protons made above the sample, over an exposure window (in parallel; integrate_particle_signal_
      spectrum for a single species);
    * integrate_atmospheric_neutrinos: neutrino interactions in the sample and in the surrounding rock;
    * integrate_radiogenic: U-238 spontaneous-fission fragments and the radiogenic (SF and (alpha,n)) neutrons,
      over the whole age.

    Each can return track lengths (observable='tracks') or the kinetic energies of the recoiling nuclei
    (observable='recoil_energy'), as one sum, the track groups (nucleus='groups', see set_track_groups) or every
    recoiling nuclide (nucleus='all'). Every source goes through the same three steps: recoil spectra
    (_recoil_spectra), density of the observable (_recoil_density), counts (_density_to_counts).

    Attributes:
        config (dict): Mineral configuration (name, composition, density, uranium concentration, (alpha,n)
            coefficient, optional track_groups).
        name (str): Mineral name, used to locate its Geant4 data.
        data_path (str): Root of the data directory.
        total_age_kyr (float): Exposure age of the sample [kyr].
        flux_history (FluxHistory): Cosmic-ray flux history (change it with set_flux_history).
        energy_bins (dict): {species: energy cell edges [GeV]} of the mineral runs found.
        track_groups (dict): Z thresholds of the track groups (set_track_groups).
        sf_neutron_spectrum, alpha_n_spectrum (callable): Radiogenic neutron emission spectra [per g per s per GeV].
        verbose (int): 0 silent, 1 warnings and summaries (default), 2 progress details (see _log).
    """

    def __init__(self, mineral_config, total_age_kyr, overburden_history=None, flux_history=None, data_dir=None):
        """Initialises the detector and builds the depth-probability tables.

        Args:
            mineral_config (dict): Mineral properties. Required keys: 'name',
                'density_g_cm3', 'uranium_concentration_g_g', 'an_coeff_n_nalpha'.
                Or the name of a material of the mineral library (config.mineral_config), e.g. "Halite".
            total_age_kyr (float): Exposure age of the sample [kyr].
            overburden_history (dict or float, optional): Overburden history (see
                _interpolate_overburden_history), or a constant overburden [m.w.e.].
                Defaults to None (no overburden).
            flux_history (FluxHistory, optional): Cosmic-ray flux history. Its
                timeline runs from 0 (present) to negative values (past): local time
                0 (start of exposure) corresponds to -total_age_kyr there, and local
                time total_age_kyr (present) to 0. If None, a baseline-only history is
                used. To change it later use set_flux_history (the only other way to
                set it). Defaults to None.
            data_dir (str, optional): Root of the data directory. Defaults to None (config.data_path():
                set_data_path, PRIMUS_DATA, <repository>/Data or ./Data).
        """
        if isinstance(mineral_config, str):
            mineral_config = dict(config.mineral_config(mineral_config, data_dir), name=mineral_config)
        self.config = mineral_config
        self.name = mineral_config['name']
        self.data_path = config.data_path(data_dir)

        self.verbose = 1

        self.track_groups = None
        self.set_track_groups()

        self.sf_neutron_spectrum = self._sf_neutron_spectrum()

        self.alpha_n_spectrum = self._alpha_n_spectrum()

        self.total_age_kyr = total_age_kyr

        self._overburden_interpolator = self._interpolate_overburden_history(overburden_history)

        self.flux_history = None
        self._time_grid_cache = None
        self.set_flux_history(flux_history)

        self._nuclear_data_cache = {}
        self._geometry_cache = {}
        self._depth_tables = {}

        self.energy_bins = self._scan_geant4_species()

        for species in COMPUTABLE_SPECIES:
            if species in self.energy_bins:
                try:
                    self._depth_table(species)
                except (FileNotFoundError, ValueError) as err:
                    self._log(f"Warning: {species} cannot be computed, no usable StdRock depth table ({err}).")

        if self.verbose > 0:
            self._print_available_species()


    def _log(self, message, level=1):
        """Prints a message if verbose >= level: the class's single way to print a warning or a progress note.

        Levels: 1 (default) warnings and the creation and time-grid summaries; 2 progress details. Multi-line
        summaries that need extra work to build (_print_available_species, the neutrino run report and missing
        products) stay behind their own `if self.verbose` check.

        Args:
            message (str): Text to print.
            level (int, optional): Smallest verbose that prints it. Defaults to 1.
        """
        if self.verbose >= level:
            print(message)


    def set_track_groups(self, heavy_min_z=None, medium_min_z=None):
        """Sets the Z thresholds of the track groups (TRACK_GROUPS, used by nucleus='groups').

        Each threshold is, in order of priority: the argument, the 'track_groups' entry of the mineral
        configuration ({'heavy_min_z': ..., 'medium_min_z': ...}, either key optional), or the prediction from the
        composition ('nuclei' and 'stoich' of the configuration, see predict_track_groups). Without a composition
        both must be given here or in the configuration.

        Args:
            heavy_min_z (int, optional): Smallest Z of the 'heavy' group.
            medium_min_z (int, optional): Smallest Z of the 'medium_heavy' group (below it: 'medium_light', down
                to Z = 3).

        Returns:
            dict: The thresholds in use, with 'source' telling where each came from.

        Raises:
            ValueError: If a threshold cannot be set, or medium_min_z > heavy_min_z.
        """
        predicted = {}
        if self.config.get('nuclei') and self.config.get('stoich'):
            predicted = predict_track_groups(self.config['nuclei'], self.config['stoich'])
        configured = dict(self.config.get('track_groups') or {})
        out, source = {}, {}
        for key, given in (('heavy_min_z', heavy_min_z), ('medium_min_z', medium_min_z)):
            for origin, value in (('argument', given), ('configuration', configured.get(key)),
                                  ('composition', predicted.get(key))):
                if value is not None:
                    out[key], source[key] = int(value), origin
                    break
            else:
                raise ValueError(f"{self.name}: no '{key}' for the track groups: give it to set_track_groups, in a "
                                 "'track_groups' entry of the mineral configuration, or give 'nuclei' and 'stoich'.")
        if out['medium_min_z'] > out['heavy_min_z']:
            raise ValueError(f"medium_min_z ({out['medium_min_z']}) > heavy_min_z ({out['heavy_min_z']}).")
        out['source'] = source
        if 'majors' in predicted:
            out['majors'] = predicted['majors']
        self.track_groups = out
        return out


    def fragment_track_group(self, fragment):
        """Track group ('heavy', 'medium_heavy', 'medium_light' or 'light') of a fragment key, e.g. "Na23".

        Raises:
            ValueError: If the key is not a nucleus (fragment_za).
        """
        za = fragment_za(fragment)
        if za is None:
            raise ValueError(f"{fragment!r} is not a nucleus.")
        return track_group(za[0], self.track_groups)


    def describe_track_groups(self):
        """Prints the track-group thresholds of the mineral and the Z range of each group."""
        th = self.track_groups
        h, m = th['heavy_min_z'], th['medium_min_z']
        sym = lambda z: f"{_element_symbol(z)}({z})"
        print(f"Track groups of {self.name} (heavy_min_z from the {th['source']['heavy_min_z']}, medium_min_z from "
              f"the {th['source']['medium_min_z']}"
              + (f"; major elements {', '.join(th['majors'])}" if 'majors' in th else "") + "):")
        print(f"  heavy         Z >= {sym(h)} (and spontaneous-fission fragments)")
        print(f"  medium_heavy  {sym(m)} <= Z < {sym(h)}" if m < h else "  medium_heavy  (empty)")
        print(f"  medium_light  Li(3) <= Z < {sym(m)}" if m > 3 else "  medium_light  (empty)")
        print("  light         Z <= 2 (H, He)")


    def set_flux_history(self, flux_history):
        """Sets the cosmic-ray flux history (the only way to change it after creation) and rebuilds the time grid.

        Called by __init__ and whenever the flux history changes. The master time grid depends on the overburden
        (fixed at creation) and on the flux history, so it is rebuilt here, at once, and its size printed.

        Args:
            flux_history (FluxHistory or None): New flux history; None means a baseline-only history.
        """
        if flux_history is None:
            flux_history = FluxHistory(
                baseline={"kind": "Baseline"}, template_dir=os.path.join(self.data_path, "flux_data")
            )
        self.flux_history = flux_history
        self._time_grid_cache = None
        grid = self._master_time_grid()
        if self.verbose > 0:
            info = self._time_grid_info
            print(f"Time grid: {grid.size} nodes over 0-{self.total_age_kyr:g} kyr ({info['start']} uniform and "
                  f"breakpoint nodes, {info['refined']} added where the overburden or flux change; "
                  f"{len(self.flux_history.events)} flux events, {len(self.flux_history.excursions)} excursions)")


    def _scan_geant4_species(self):
        """Finds the Geant4 runs of the mineral: one folder per MCEQ_SPECIES entry in Data/Geant4_data/<name>.

        Returns:
            dict: species -> primary-energy bin edges [GeV] built from the file names (_geant4_energy_edges).
            Species with fewer than 2 files are left out (with a warning).
        """
        self._geant4_runs = {}
        mineral_dir = os.path.join(self.data_path, "Geant4_data", self.name)
        for species in MCEQ_SPECIES:
            info = _geant4_energy_edges(os.path.join(mineral_dir, species))
            if info is None:
                if _list_geant4_energy_files(os.path.join(mineral_dir, species)):
                    self._log(f"Warning: {self.name}/{species}: a single Geant4 file, at least 2 are needed; ignored.")
                continue
            self._geant4_runs[species] = info
        return {species: info['edges'] for species, info in self._geant4_runs.items()}


    def _print_available_species(self):
        """Prints the species found by _scan_geant4_species, with their energy range."""
        if not self._geant4_runs:
            print(f"Created paleo-detector {self.name}: no Geant4 runs found in "
                  f"{os.path.join(self.data_path, 'Geant4_data', self.name)} (only fission is available).")
            return
        lines = [f"Created paleo-detector {self.name}, species available for computation:"]
        for species, info in self._geant4_runs.items():
            e = info['edges']
            note = ("" if self._has_depth_table(species) else
                    ("  (integrate_atmospheric_neutrinos; secondaries made in "
                     + ("StdRock)" if _list_geant4_energy_files(os.path.join(self.data_path, "Geant4_data", "StdRock",
                                                                             species)) else "the mineral: no StdRock runs)"))
                    if species in NEUTRINO_SPECIES else
                    "  (no StdRock depth table: not computable)" if species in COMPUTABLE_SPECIES else
                    "  (runs found, calculation not implemented yet)")
            lines.append(f"  {species:<10} energy from {e[0]:.4g} to {e[-2]:.4g} GeV in {info['n_files']} steps{note}")
            if self.verbose > 1 and info['gaps']:    
                for lo, hi in info['gaps']:
                    lines.append(f"  {'':<10} warning: no file between {lo:.4g} and {hi:.4g} GeV "
                                f"(the {lo:.4g} GeV run is used for the whole step)")
        for derived, product in DERIVED_SPECIES.items():
            if derived in self.available_species():
                sources = [src for src in SECONDARY_SOURCES if self._stdrock_dir(src) is not None]
                lines.append(f"  {derived:<10} {product}s produced by {', '.join(sources)} above the sample "
                             f"(StdRock runs, all generations of the n + p cascade)")
        print("\n".join(lines))


    def available_species(self):
        """Species that integrate_all_particles can compute with the Geant4 runs present.

        Returns:
            list[str]: Computable species with mineral runs and a StdRock depth table, plus the secondary species
            whose product is among them.
        """
        out = [s for s in COMPUTABLE_SPECIES if s in self.energy_bins and self._has_depth_table(s)]
        out += [d for d, src in DERIVED_SPECIES.items() if src in out]
        return out


    def _energy_edges(self, species, e_range=None, edges=None):
        """Primary-energy bin edges of a species: the given edges, or its Geant4 run grid restricted to a range.

        The single place where the energy bins of a computation are decided; used by every integration path
        (cosmic-ray primaries and secondaries, radiogenic neutrons, neutrinos) and by the recoil converters to list
        the fragments.

        Args:
            species (str): Species ('mu-', 'mu+', 'neutron', 'secondary_neutron', ...).
            e_range (tuple[float, float], optional): (E_min, E_max) [GeV]. The bins lying entirely inside it are
                kept. Ignored if edges is given. Defaults to None (all the files of the species).
            edges (array_like, optional): Explicit bin edges [GeV], returned as they are. Defaults to None.

        Returns:
            np.ndarray: Bin edges [GeV].

        Raises:
            ValueError: If the species has no runs, or e_range is outside the available range or contains no bin.
        """
        if edges is not None:
            return np.asarray(edges, dtype=float)
        source = DERIVED_SPECIES.get(species, species)
        if source not in self.energy_bins:
            raise ValueError(f"No Geant4 runs of '{source}' for {self.name}; available: {list(self.energy_bins)}.")
        return _restrict_edges(self.energy_bins[source], e_range, species)


    def _flux_breakpoints(self, nodes=True):
        """Times [kyr, local] where the flux history changes abruptly (event starts and excursion edges, each with
        a point just before) and, if nodes, the template time nodes on which an event's flux is interpolated (the
        templates' t_since_yr axis is in years, hence the factor 1e-3). Called by _master_time_grid."""
        if self.flux_history is None:
            return np.zeros(0)
        eps = 1e-9 * self.total_age_kyr
        to_local = self.total_age_kyr
        points = []
        for event in self.flux_history.events:
            start = event["start_time_kyr"] + to_local
            points.append([start - eps, start])
            t_since = getattr(event["template"], "t_since_yr", None)
            if nodes and t_since is not None:
                points.append(start + np.asarray(t_since, dtype=float) * 1e-3)
        for exc in self.flux_history.excursions:
            for edge in (exc["start_time_kyr"], exc["end_time_kyr"]):
                points.append([edge + to_local - eps, edge + to_local])
        return np.concatenate([np.asarray(q, dtype=float).ravel() for q in points]) if points else np.zeros(0)


    def _grid_probes(self, t_kyr):
        """Probe functions of time used to refine the master grid, shape (n_probes, len(t_kyr))."""
        depth = np.asarray(self._overburden_interpolator(t_kyr), dtype=float)
        probes = [depth] + [np.exp(-depth / lam) for lam in TIME_GRID_ATTENUATION_MWE]
        if self.flux_history is not None:
            t_flux = self._flux_time_kyr(t_kyr)
            for species, energy in TIME_GRID_FLUX_PROBES:
                probes.append(np.asarray(self.flux_history.flux(species, np.full(t_kyr.shape, energy), t_flux),
                                         dtype=float))
        return np.array(probes)


    def _master_time_grid(self):
        """The detector's time grid over [0, total_age_kyr] [kyr, local], built adaptively and cached.

        Built by set_flux_history (at creation and on every change of the flux history); later calls return the
        cached grid.

        Start: TIME_GRID_BACKBONE uniform points, the breakpoints of the overburden history (phase limits,
        discrete deposits) and of the flux history (_flux_breakpoints). Refinement: every interval whose
        trapezoid error on any probe (_grid_probes), estimated at its midpoint, exceeds TIME_GRID_RTOL times that
        probe's integral over the segment between the abrupt breakpoints around it is split in two (down to
        1e-6 of the age, so that a genuine jump is not refined forever); repeated until no interval needs it, TIME_GRID_MAX_PASSES,
        or TIME_GRID_MAX_POINTS. Steps in either history are therefore resolved where they matter for the rates,
        and nowhere else. Rebuilt after set_flux_history.

        Returns:
            np.ndarray: Sorted time points [kyr].
        """
        if self._time_grid_cache is not None:
            return self._time_grid_cache
        t_now = float(self.total_age_kyr)
        t = np.concatenate([np.linspace(0.0, t_now, TIME_GRID_BACKBONE), self._overburden_breakpoints,
                            self._flux_breakpoints()])
        t = np.unique(t[(t >= 0.0) & (t <= t_now)])
        n_start = t.size
        seg_edges = np.unique(np.concatenate([[0.0, t_now], self._overburden_breakpoints,
                                              self._flux_breakpoints(nodes=False)]))
        min_dt = 1e-6 * t_now
        p_t = self._grid_probes(t)
        for _ in range(TIME_GRID_MAX_PASSES):
            mid = 0.5 * (t[1:] + t[:-1])
            dt = np.diff(t)
            p_mid = self._grid_probes(mid)
            trap = 0.5 * np.abs(p_t[:, 1:] + p_t[:, :-1]) * dt
            seg = np.clip(np.searchsorted(seg_edges, mid) - 1, 0, seg_edges.size - 2)
            seg_total = np.zeros((trap.shape[0], seg_edges.size - 1))
            np.add.at(seg_total.T, seg, trap.T)
            err = np.abs(p_mid - 0.5 * (p_t[:, 1:] + p_t[:, :-1])) * dt
            split = np.any(err > TIME_GRID_RTOL * np.maximum(seg_total[:, seg], 1e-300), axis=0) & (dt > min_dt)
            if not split.any() or t.size + split.sum() > TIME_GRID_MAX_POINTS:
                if split.any():
                    self._log(f"Warning: time grid stopped at {t.size} points (TIME_GRID_MAX_POINTS).")
                break
            t = np.concatenate([t, mid[split]])
            p_t = np.concatenate([p_t, p_mid[:, split]], axis=1)
            order = np.argsort(t)
            t, p_t = t[order], p_t[:, order]
        self._time_grid_cache = t
        self._time_grid_info = {'start': n_start, 'refined': t.size - n_start}
        return t


    def _window_grid(self, window_kyr):
        """The master grid restricted to an exposure window, with the window ends added [kyr]."""
        t0, t1 = window_kyr
        grid = self._master_time_grid()
        return np.unique(np.concatenate([[t0, t1], grid[(grid > t0) & (grid < t1)]]))


    def _exposure_window(self, exposure_window_kyr):
        """[start, end] of an exposure window [kyr]: None means the whole age, a number w means [0, w]."""
        if exposure_window_kyr is None:
            exposure_window_kyr = self.total_age_kyr
        if isinstance(exposure_window_kyr, (int, float)):
            exposure_window_kyr = [0, exposure_window_kyr]
        return [float(exposure_window_kyr[0]), float(exposure_window_kyr[1])]


    def _interpolate_overburden_history(self, overburden_history=None):
        """Builds the overburden [m.w.e.] as a function of local time.

        The overburden starts at initial_depth * initial_density_g_cm3 and grows
        through continuous deposition phases (rate * density over each time window)
        and discrete deposition events (thickness * density from a given time on).

        The times where the overburden changes abruptly (phase limits, discrete deposits) are kept in
        _overburden_breakpoints, for the master time grid (_master_time_grid).

        Args:
            overburden_history (dict or float, optional): Either a constant overburden [m.w.e.], or a dict with the
                optional keys (defaults in brackets) 'initial_depth' [m] (0.3), 'initial_density_g_cm3' (1.0),
                'start_time_continuous_kyr' (0), 'end_time_continuous_kyr' (total_age_kyr), 'rate_continuous'
                [m/kyr, negative for erosion] (0), 'density_continuous_g_cm3' (1.0), 'time_discrete_kyr' (0),
                'overburden_discrete' [m, negative for removal] (0), 'density_discrete_g_cm3' (1.0). The continuous
                and discrete keys take a number (the same for every phase or deposit) or one value per phase or
                deposit. Defaults to None (0 m.w.e.).

        Returns:
            scipy.interpolate.interp1d: Local time [kyr] -> overburden [m.w.e.].

        Raises:
            ValueError: If the dict has an unknown key.
        """
        self._overburden_breakpoints = np.array([0.0, float(self.total_age_kyr)])
        if isinstance(overburden_history, (int, float)):
            const = float(overburden_history)
            return interp1d([0.0, self.total_age_kyr], [const, const],
                             bounds_error=False, fill_value=const)
        elif overburden_history is None:
            return interp1d([0.0, self.total_age_kyr], [0.0, 0.0],
                             bounds_error=False, fill_value=0.0)

        defaults = {
            'initial_depth': 0.3, 'initial_density_g_cm3': 1.0,
            'start_time_continuous_kyr': 0.0, 'end_time_continuous_kyr': self.total_age_kyr, 'rate_continuous': 0.0,
            'density_continuous_g_cm3': 1.0,
            'time_discrete_kyr': 0.0, 'overburden_discrete': 0.0, 'density_discrete_g_cm3': 1.0,
        }
        unknown = set(overburden_history) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown overburden_history keys {sorted(unknown)} (use {list(defaults)}).")
        h = {**defaults, **overburden_history}
        initial_depth, initial_density_g_cm3 = h['initial_depth'], h['initial_density_g_cm3']
        continuous = ('start_time_continuous_kyr', 'end_time_continuous_kyr', 'rate_continuous',
                      'density_continuous_g_cm3')
        discrete = ('time_discrete_kyr', 'overburden_discrete', 'density_discrete_g_cm3')
        try:
            start_time_continuous_kyr, end_time_continuous_kyr, rate_continuous, density_continuous_g_cm3 = (
                np.broadcast_arrays(*(np.atleast_1d(np.asarray(h[k], dtype=float)) for k in continuous)))
            time_discrete, overburden_discrete, density_discrete_g_cm3 = (
                np.broadcast_arrays(*(np.atleast_1d(np.asarray(h[k], dtype=float)) for k in discrete)))
        except ValueError:
            raise ValueError(f"overburden_history: the keys {continuous} and the keys {discrete} must each be "
                             "numbers or lists of the same length (one value per phase or deposit).") from None

        eps = 1e-9 * self.total_age_kyr
        breaks = [0.0, self.total_age_kyr, *start_time_continuous_kyr, *end_time_continuous_kyr]
        steps_at = [t for t, o in zip(time_discrete, overburden_discrete) if o != 0.0]
        breaks += list(steps_at) + [t - eps for t in steps_at]
        breaks = np.clip(np.asarray(breaks, dtype=float), 0.0, self.total_age_kyr)
        self._overburden_breakpoints = np.unique(breaks)
        times = np.union1d(np.linspace(0, self.total_age_kyr, 10000), self._overburden_breakpoints)
        overburdens = np.zeros_like(times) + initial_depth * initial_density_g_cm3

        for start, end, rate, density in zip(start_time_continuous_kyr, end_time_continuous_kyr, rate_continuous, density_continuous_g_cm3):
            mask = (times >= start) & (times <= end)
            overburdens[mask] += density * rate * (times[mask] - start)
            mask_after_end = (times > end)
            overburdens[mask_after_end] += density * rate * (end - start)

        for t, o, d in zip(time_discrete, overburden_discrete, density_discrete_g_cm3):
            mask = (times >= t)
            overburdens[mask] += d * o

        if np.any(overburdens < 0.0):
            self._log(f"Warning: the overburden history erodes below the surface (down to {overburdens.min():.3g} "
                      "m.w.e.); it is set to 0 there.")
            overburdens = np.maximum(overburdens, 0.0)

        self._log("Interpolating overburden history...", level=2)

        return interp1d(times, overburdens, bounds_error=True)


    def _load_nuclear_data(self, filename):
        """Loads and caches a nuclear data file.

        Args:
            filename (str): 'U238.dat' (fission fragments, columns Z, A, ...) or
                'BindingEne.txt' (Z, A, binding energy per nucleon [MeV]).

        Returns:
            tuple[np.ndarray, ...]: The three used columns, unpacked.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file name is not a known format.
        """
        if filename in self._nuclear_data_cache:
            return self._nuclear_data_cache[filename]

        filepath = os.path.join(self.data_path,"nuclear_data", filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Nuclear data file not found: {filepath}")

        if "BindingEne.txt" in filename:
            cols_to_use = (0, 1, 2)
        elif "U238.dat" in filename:
            cols_to_use = (1, 2, 3)
        else:
            raise ValueError(f"Unknown nuclear data file format: {filename}")

        self._nuclear_data_cache[filename] = np.loadtxt(filepath, usecols=cols_to_use, unpack=True)
        return self._nuclear_data_cache[filename]


    def _fission_rate_per_g_kyr(self):
        """Spontaneous-fission rate of the mineral.

        Single source for both fission branches: the fragment tracks
        (_fission_track_density, _fission_fragment_recoils) and the SF neutrons (_sf_neutron_spectrum).

        Returns:
            float: Fissions per g of mineral per kyr,
            uranium_concentration_g_g * U238_ABUNDANCE * FISSIONS_U238_G_KYR.
        """
        return self.config["uranium_concentration_g_g"] * U238_ABUNDANCE * FISSIONS_U238_G_KYR


    def _sf_neutron_spectrum(self):
        """Neutron emission spectrum from U-238 spontaneous fission.

        Watt spectrum exp(-E/a) sinh(sqrt(b E)) with a = WATT_A_MEV and
        b = WATT_B_PER_MEV (Cranberg et al., Phys. Rev. 103, 662 (1956)),
        normalised to the fission rate of _fission_rate_per_g_kyr times
        NU_BAR_U238_SF neutrons per fission, so the neutron yield and the
        fragment tracks come from the same fission rate.

        Built once by __init__ (attribute sf_neutron_spectrum); used by _radiogenic_neutron_recoils ('fission_n').

        Returns:
            scipy.interpolate.interp1d: Energy [GeV] -> neutrons per g per s per GeV.
        """
        sf_yield_per_g_s = self._fission_rate_per_g_kyr() * NU_BAR_U238_SF / SECONDS_PER_KYR
        energies_gev = np.logspace(-6, -1, 10000)
        energies_mev = energies_gev * 1e3
        watt_shape = np.exp(-energies_mev / WATT_A_MEV) * np.sinh(np.sqrt(WATT_B_PER_MEV * energies_mev))
        sf_flux = watt_shape * (sf_yield_per_g_s / np.trapezoid(watt_shape, energies_gev))
        return interp1d(energies_gev, sf_flux, bounds_error=False, fill_value='extrapolate')


    def _alpha_n_spectrum(self):
        """Neutron emission spectrum from (alpha,n) reactions.

        Shape E^2 exp(-beta E) with a smooth cut-off, normalised to the (alpha,n)
        yield of the U-238 chain (8 alphas per chain) times the mineral coefficient
        an_coeff_n_nalpha (Kudryavtsev et al., SciPost Phys. Proc. 12 (2023), SOURCES4).

        Built once by __init__ (attribute alpha_n_spectrum); used by _radiogenic_neutron_recoils ('alpha_n').

        Returns:
            scipy.interpolate.interp1d: Energy [GeV] -> neutrons per g per s per GeV.
        """
        an_yield_per_gu_s = self.config["an_coeff_n_nalpha"] * 8 * 1.245 * 1e-5 * 1e9 / 1e6

        an_yield_per_g_s = an_yield_per_gu_s * self.config['uranium_concentration_g_g']

        alpha, beta_gev, cutoff_e_gev = 2, 900, 0.0065

        energies_gev = np.logspace(-4, -0.5, 10000)

        an_shape_gev = (energies_gev**alpha)* np.exp(-beta_gev*energies_gev)
        an_shape_gev *= 1.0 / (1.0 + np.exp((energies_gev - cutoff_e_gev) / 0.0005))

        an_flux_gev = an_shape_gev * (an_yield_per_g_s / np.trapezoid(an_shape_gev, energies_gev))

        interpolator = interp1d(energies_gev, an_flux_gev, bounds_error=False, fill_value='extrapolate')

        return interpolator


    def _radiogenic_neutron_recoils(self, source, energy_bins_gev=None, total_simulated_particles=1e4):
        """Recoils made in the mineral by the neutrons of one radiogenic source, constant in time.

        The neutrons are emitted inside the mineral, so every recoil they cause is counted (infinite homogeneous
        medium): the recoils per neutron of the mineral's neutron run at the lower edge of each energy cell are
        weighted by the neutrons emitted in the cell (emission spectrum integrated with _energy_cell_nodes). Called
        by integrate_radiogenic.

        Args:
            source (str): 'fission_n' (U-238 spontaneous-fission neutrons, _sf_neutron_spectrum) or 'alpha_n'
                ((alpha,n) neutrons, _alpha_n_spectrum).
            energy_bins_gev (np.ndarray, optional): Neutron energy cell edges [GeV]. Defaults to None (all the
                neutron runs of the mineral).
            total_simulated_particles (float, optional): Neutrons per Geant4 run. Defaults to 1e4.

        Returns:
            dict: Recoil spectra (_recoil_spectra), constant in time.

        Raises:
            ValueError: If source is not a neutron source.
        """
        emission = {'fission_n': self.sf_neutron_spectrum, 'alpha_n': self.alpha_n_spectrum}.get(source)
        if emission is None:
            raise ValueError(f"Unknown radiogenic neutron source '{source}' (expected 'fission_n' or 'alpha_n').")
        edges = self._energy_edges('neutron', edges=energy_bins_gev)
        run_dir = os.path.join(self.data_path, "Geant4_data", self.name, "neutron")
        nodes, weights = _energy_cell_nodes(edges)
        emitted_per_g_s = np.sum(emission(nodes) * weights, axis=-1)
        fragments = {}
        for e_lo, weight in zip(edges[:-1], emitted_per_g_s):
            hist = _mineral_run_histograms(os.path.join(run_dir, f"outNuclei_{e_lo:.6f}.txt"))
            if hist is None:
                continue
            for frag, counts in sorted(hist['all'].items()):
                _accumulate_into(fragments, frag, counts * weight)
        per_kyr = SECONDS_PER_KYR / total_simulated_particles
        return _recoil_spectra({frag: spectrum * per_kyr for frag, spectrum in fragments.items()})


    def _fission_fragments(self):
        """Kinetic energies of the two fragments of each U-238 spontaneous-fission event of U238.dat.

        Two-body kinematics with masses from BindingEne.txt, fragments taken before neutron emission (A1 + A2 = 238;
        a warning is printed otherwise). Events with a fragment missing from BindingEne.txt are skipped (warning).
        Called by _fission_track_density and _fission_fragment_recoils, so both observables use the same events.

        Returns:
            list[tuple]: (z1, a1, Ek1_MeV, z2, a2, Ek2_MeV) of each event used.

        Raises:
            ValueError: If no event has binding energies for both fragments.
        """
        Z_fission, A_fission, _ = self._load_nuclear_data("U238.dat")
        Z_bind, A_bind, B_bind = self._load_nuclear_data("BindingEne.txt")
        binding_map = {(int(z), int(a)): b for z, a, b in zip(Z_bind, A_bind, B_bind)}
        M0 = 92 * PROTON_MASS_MEV + (238 - 92) * NEUTRON_MASS_MEV - B0_U238 * 238

        events = []
        num_events = len(Z_fission) // 3
        n_skipped = 0
        n_not_238 = 0
        for i in range(num_events):
            z1, a1 = int(Z_fission[3*i + 1]), int(A_fission[3*i + 1])
            z2, a2 = int(Z_fission[3*i + 2]), int(A_fission[3*i + 2])

            if a1 + a2 != 238:
                n_not_238 += 1

            b1 = binding_map.get((z1, a1), 0)
            b2 = binding_map.get((z2, a2), 0)
            if b1 == 0 or b2 == 0:
                n_skipped += 1
                continue

            m1 = z1 * PROTON_MASS_MEV + (a1 - z1) * NEUTRON_MASS_MEV - b1 * a1
            m2 = z2 * PROTON_MASS_MEV + (a2 - z2) * NEUTRON_MASS_MEV - b2 * a2

            Ek1_MeV = (M0**2 + m1**2 - m2**2) / (2 * M0) - m1
            Ek2_MeV = (M0**2 + m2**2 - m1**2) / (2 * M0) - m2
            events.append((z1, a1, Ek1_MeV, z2, a2, Ek2_MeV))

        if not events:
            raise ValueError("No fission event with binding energies for both fragments.")
        if n_not_238:
            self._log(f"Warning: {n_not_238}/{num_events} fission events with A1 + A2 != 238; "
                      "the two-body kinematics assume fragments before neutron emission.")
        if n_skipped:
            self._log(f"Warning: {n_skipped}/{num_events} fission events skipped "
                      "(fragment missing from BindingEne.txt).")
        return events


    def _fission_track_density(self, x_grid, fission_key='heavier'):
        """Track-length density of U-238 spontaneous fission: one track per fission, filed under one key.

        The two fragments fly apart back to back from the same point, so a fission makes a single track whose length
        is the sum of the two fragment ranges (kinetic energies from _fission_fragments, CSDA ranges from the
        mineral's range tables). This is why fission tracks do not go through _recoils_to_track_density, which makes
        one track per recoil (it would give two half-length tracks per fission). The rate is _fission_rate_per_g_kyr,
        the same as for the SF neutrons. Called by integrate_radiogenic (observable='tracks').

        Args:
            x_grid (np.ndarray): Fine track-length grid [nm].
            fission_key (str, optional): Key of each track: 'heavier' (fragment with the larger A; the first of
                U238.dat if equal), 'lighter', or 'pair' ('<heavier>+<lighter>', e.g. 'Sn133+Mo105'). Defaults to
                'heavier'.

        Returns:
            dict: {key: dR/dx [tracks per mg per kyr per nm] on x_grid}.

        Raises:
            ValueError: If fission_key is unknown.
        """
        if fission_key not in FISSION_KEYS:
            raise ValueError(f"fission_key must be one of {FISSION_KEYS}, got {fission_key!r}.")
        events = self._fission_fragments()
        range_dir = os.path.join(self.data_path, "Geant4_data", self.name, "Range")

        range_of = {}

        def range_nm(z, a, e_mev):
            if (z, a) not in range_of:
                e_tab, _, _, length_um = _range_table(os.path.join(range_dir, f"DEDX_Z{z}_A{a}.txt"))
                range_of[(z, a)] = interp1d(e_tab, length_um * 1e3, bounds_error=False, fill_value=0.0)
            return float(range_of[(z, a)](e_mev))

        lengths, keys = [], []
        for z1, a1, e1, z2, a2, e2 in events:
            lengths.append(range_nm(z1, a1, e1) + range_nm(z2, a2, e2))
            heavy, light = ((z1, a1), (z2, a2)) if a1 >= a2 else ((z2, a2), (z1, a1))
            heavy, light = f"{_element_symbol(heavy[0])}{heavy[1]}", f"{_element_symbol(light[0])}{light[1]}"
            keys.append({'heavier': heavy, 'lighter': light, 'pair': f"{heavy}+{light}"}[fission_key])
        lengths, keys = np.asarray(lengths), np.asarray(keys)
        per_fission_mg_kyr = self._fission_rate_per_g_kyr() / len(events) * 1e-3
        widths = np.diff(x_grid)
        return {str(k): np.histogram(lengths[keys == k], bins=x_grid)[0] * per_fission_mg_kyr / widths
                for k in np.unique(keys)}


    def _fission_fragment_recoils(self):
        """Recoils of the U-238 spontaneous-fission fragments: two per fission, each under its own nuclide.

        Kinetic energies from _fission_fragments, at the rate _fission_rate_per_g_kyr per fission. Called by
        integrate_radiogenic (observable='recoil_energy'; for tracks see _fission_track_density).

        Returns:
            dict: Recoil spectra (_recoil_spectra), constant in time.
        """
        events = self._fission_fragments()
        per_fragment = self._fission_rate_per_g_kyr() / len(events)
        energies = {}
        for z1, a1, e1, z2, a2, e2 in events:
            for z, a, e in ((z1, a1, e1), (z2, a2, e2)):
                energies.setdefault(f"{_element_symbol(z)}{a}", []).append(e)
        return _recoil_spectra({name: np.histogram(e, bins=RECOIL_ENERGY_BINS_MEV)[0] * per_fragment
                                for name, e in sorted(energies.items())})


    def integrate_radiogenic(
        self, x_bins=None, sources=RADIOGENIC_SOURCES, energy_bins_gev=None, total_simulated_particles=1e4,
        x_grid=TRACK_LENGTH_BINS_NM, nucleus=DEFAULT_TRACK_SUM, observable=OBSERVABLE_TRACKS,
        fission_key='heavier', fission_group='heavy',
    ):
        """Track counts (or recoil counts per recoil-energy bin) from the radiogenic sources over the sample age.

        The backgrounds of the uranium in the mineral, all constant in time (accumulated over the whole age, with no
        exposure window), one entry each, as the species of integrate_all_particles:

        * 'fission_fragments': U-238 spontaneous fission. Tracks: one per fission, the two fragment ranges summed
          (_fission_track_density), filed under fission_key. Recoil energies: each fragment is a recoil (two per
          fission) under its own nuclide (_fission_fragment_recoils). Every key is in fission_group.
        * 'fission_n': recoils made by the spontaneous-fission neutrons (Watt spectrum, same fission rate).
        * 'alpha_n': recoils made by the (alpha,n) neutrons.

        Args:
            x_bins (np.ndarray, optional): Output bin edges: track lengths [nm] (default DEFAULT_INTEGRATION_X_BINS_NM)
                or recoil energies [MeV] (default RECOIL_ENERGY_BINS_MEV).
            sources (tuple[str], optional): Sources to compute, from RADIOGENIC_SOURCES. Defaults to all three.
            energy_bins_gev (np.ndarray, optional): Neutron energy cell edges [GeV] of the neutron sources. Defaults
                to None (all the neutron runs).
            total_simulated_particles (float, optional): Neutrons per Geant4 run. Defaults to 1e4.
            x_grid (np.ndarray, optional): Fine track-length grid [nm] (ignored for recoil energies). Defaults to
                TRACK_LENGTH_BINS_NM.
            nucleus (str, optional): As in integrate_particle_signal_spectrum. A fragment key missing from a source
                gives zeros for that source (an error only if no source has it). Defaults to 'total_no_light_ions'.
            observable (str, optional): 'tracks' or 'recoil_energy'. Defaults to 'tracks'.
            fission_key (str, optional): Key of the fission tracks: 'heavier', 'lighter' or 'pair' (see
                _fission_track_density). Not used for recoil energies. Defaults to 'heavier'.
            fission_group (str, optional): Track group of every fission fragment. Defaults to 'heavy'.

        Returns:
            tuple[np.ndarray, dict]: x_bins, and {source: tracks (or recoils) per mg in each bin, ..., 'total': sum}.

        Raises:
            ValueError: If a source, fission_key or fission_group is unknown, or nucleus is in no source.
        """
        x_bins, grid = _output_axes(observable, x_bins, x_grid)
        unknown = [src for src in sources if src not in RADIOGENIC_SOURCES]
        if unknown:
            raise ValueError(f"Unknown radiogenic sources {unknown} (use {RADIOGENIC_SOURCES}).")
        if fission_group not in TRACK_GROUPS:
            raise ValueError(f"fission_group must be one of {TRACK_GROUPS}, got {fission_group!r}.")

        results = {}
        for source in sources:
            if source == 'fission_fragments':
                density = (self._fission_track_density(grid, fission_key) if observable == OBSERVABLE_TRACKS else
                           self._recoil_density(self._fission_fragment_recoils(), observable, grid, source))
                group = fission_group
            else:
                spectra = self._radiogenic_neutron_recoils(source, energy_bins_gev, total_simulated_particles)
                density, group = self._recoil_density(spectra, observable, grid, source), None
            results[source] = self._density_to_counts(density, grid, x_bins, None, 'all', source, group)
        results['total'] = _combine_track_results(results.values())

        self._select_nucleus(results['total'], nucleus, 'the radiogenic sources')
        zeros = np.zeros(len(x_bins) - 1)
        return x_bins, {k: (self._select_nucleus(r, nucleus, k)
                            if nucleus in r or nucleus in ('all', TRACK_GROUPS_KEY) else zeros)
                        for k, r in results.items()}


    def _geant4_geometry(self, data_dir):
        """Geometry of a Geant4 output directory.

        Read from <data_dir>/geometry.json, written by geant4_runs.run_geant4_simulation; without it, the defaults of
        _core (a StdRock target of GEANT4_DEFAULT_TARGET_LENGTH_M shot from its top face).

        Args:
            data_dir (str): Geant4 output directory.

        Returns:
            dict: 'depth_reference_z_m' (z of the target top face [m]),
            'target_length_m' [m], 'density_g_cm3' (None if unknown) and 'neutrino_bias' (neutrino runs only: the
            cross-section bias factor, a number or {"<E0 as %.6f>": factor}; None if not written).
        """
        if data_dir in self._geometry_cache:
            return self._geometry_cache[data_dir]

        geo = {
            "depth_reference_z_m": GEANT4_DEFAULT_TARGET_LENGTH_M / 2.0,
            "target_length_m": GEANT4_DEFAULT_TARGET_LENGTH_M,
            "density_g_cm3": GEANT4_HARDCODED_DENSITIES_G_CM3["StdRock"],
            "neutrino_bias": None,
        }
        meta_path = os.path.join(data_dir, "geometry.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            geo["depth_reference_z_m"] = float(meta["depth_reference_z_m"])
            geo["target_length_m"] = float(meta.get("target_length_m", 2.0 * geo["depth_reference_z_m"]))
            if meta.get("density_g_cm3") is not None:
                geo["density_g_cm3"] = float(meta["density_g_cm3"])
            geo["neutrino_bias"] = meta.get("neutrino_bias")

        self._geometry_cache[data_dir] = geo
        return geo


    def _depth_table(self, species):
        """Depth table of a species from its StdRock runs, built on first use and cached in _depth_tables.

        Muons (mu- and mu+ share the mu- table) use the empirical distribution of the stopping depth R (rows with
        remaining primary energy 0); hadrons (StdRock/neutron, StdRock/proton) the distribution of all interaction
        depths up to the NEUTRON_TAIL_P quantile, with an exponential tail beyond it (maximum-likelihood length:
        mean excess beyond the last node). For protons the interactions stop where the proton stops, which the
        empirical distribution contains. Every outNuclei_<E>.txt in the StdRock directory is used, so a denser
        energy grid is picked up automatically. Energies with fewer than MIN_DEPTH_SAMPLES depths, or whose depths
        reach TRUNCATION_FRACTION of the StdRock target (the particle may have left it), are skipped with a warning.

        Built for every computable species by __init__ (a species whose table cannot be built is not computable,
        see available_species); read by _depth_rows, _slab_interaction_probabilities and _secondary_window_mwe.

        Args:
            species (str): 'mu-', 'mu+', 'neutron' or 'proton'.

        Returns:
            dict: 'energies' [GeV], 'p' (node probabilities), 'q' (quantile nodes [m.w.e.], shape (n_energies,
            n_nodes)) and, for hadrons, 'tail_lambda' (exponential-tail length per energy [m.w.e.]).

        Raises:
            FileNotFoundError: If the StdRock directory has no output files.
            ValueError: If fewer than two energies are usable.
        """
        tab_species = 'mu-' if species in MUON_SPECIES else species
        if tab_species in self._depth_tables:
            return self._depth_tables[tab_species]

        data_dir = os.path.join(self.data_path, "Geant4_data", f"StdRock/{tab_species}")
        is_muon = tab_species == 'mu-'

        geo = self._geant4_geometry(data_dir)
        z_ref_m = geo["depth_reference_z_m"]
        rho = geo["density_g_cm3"] if geo["density_g_cm3"] is not None else 2.65
        target_mwe = geo["target_length_m"] * rho

        files = _list_geant4_energy_files(data_dir)
        if not files:
            raise FileNotFoundError(f"No StdRock Geant4 files found in {data_dir}")

        p_nodes = STOPPING_P_NODES if is_muon else NEUTRON_P_NODES
        energies, rows, tails, skipped, truncated = [], [], [], [], []
        for e0, filepath in files:
            if os.path.getsize(filepath) == 0:
                skipped.append(e0)
                continue
            df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[3, 5],
                             names=['z_mm', 'rem_e'], dtype=float)
            depth_mwe = np.clip((z_ref_m - df['z_mm'].values * 1e-3) * rho, 0.0, None)

            sample = depth_mwe[df['rem_e'].values == 0.0] if is_muon else depth_mwe
            if sample.size < MIN_DEPTH_SAMPLES:
                skipped.append(e0)
                continue
            if sample.max() > TRUNCATION_FRACTION * target_mwe:
                truncated.append(e0)
                continue
            q = np.quantile(sample, p_nodes)
            if not is_muon:
                excess = sample[sample > q[-1]] - q[-1]
                lam_tail = excess.mean() if excess.size > 0 else sample.mean()
                tails.append(max(lam_tail, DEPTH_FLOOR_MWE))
            rows.append(np.maximum(q, DEPTH_FLOOR_MWE))
            energies.append(e0)

        if skipped:
            self._log(f"Warning: StdRock/{tab_species}: skipped {len(skipped)} energies with too few "
                      f"depth samples (e.g. {skipped[:3]} GeV).")
        if truncated:
            self._log(f"Warning: StdRock/{tab_species}: skipped {len(truncated)} energies whose "
                      f"depths reach the end of the target (from {min(truncated):g} GeV); "
                      f"use a longer StdRock target to cover them.")
        if len(energies) < 2:
            raise ValueError(f"StdRock/{tab_species}: need at least 2 usable energies, found {len(energies)}.")

        table = {'energies': np.array(energies), 'p': p_nodes, 'q': np.array(rows)}
        if not is_muon:
            table['tail_lambda'] = np.array(tails)
        self._depth_tables[tab_species] = table
        return table


    def _has_depth_table(self, species):
        """Whether the depth table of a species (the mu- table for mu+) has been built (see _depth_table)."""
        return ('mu-' if species in MUON_SPECIES else species) in self._depth_tables


    def _depth_rows(self, species, energies_gev):
        """Depth quantile nodes (and, for hadrons, exponential-tail lengths) at arbitrary energies.

        Interpolated in log-log between the tabulated energies of _depth_table (_loglog_interp_rows); the nodes are
        then forced non-decreasing. Called by _slab_interaction_probabilities (primaries) and _secondary_recoils
        (products of the secondary cascade).

        Args:
            species (str): 'mu-', 'mu+', 'neutron' or 'proton'.
            energies_gev (array_like): Energies [GeV], shape (n,).

        Returns:
            tuple[np.ndarray, np.ndarray or None]: Quantile nodes [m.w.e.], shape (n, n_nodes), and tail lengths
            [m.w.e.], shape (n,) (None for muons).
        """
        table = self._depth_table(species)
        q = np.maximum.accumulate(_loglog_interp_rows(energies_gev, table['energies'], table['q']), axis=-1)
        if 'tail_lambda' not in table:
            return q, None
        return q, _loglog_interp_rows(energies_gev, table['energies'], table['tail_lambda'])[:, 0]


    def _slab_interaction_probabilities(self, species, e_vals, z_min_grid, z_max_grid):
        """Probability that a primary of each energy interacts in the sample slab, per category of recoils.

        The categories are those of the recoil histograms of the mineral runs (_mineral_run_histograms), so that
        _primary_recoils weights each histogram with its own probability:

        * mu-: 'elastic' (the interactions along the path, with density proportional to P(R > z):
          (G(z_max) - G(z_min)) / E[R]) and 'peak' (the stop, followed by capture: F(z_max) - F(z_min)), R being the
          stopping depth.
        * mu+: 'all', the interactions along the path (as 'elastic' above; the stop gives no capture).
        * neutron, proton: 'all', the interaction depth inside the slab, F(z_max) - F(z_min).

        F and G come from _depth_cdf_rows, for all the energies at once.

        Called by _primary_recoils.

        Args:
            species (str): 'mu-', 'mu+', 'neutron' or 'proton'.
            e_vals (np.ndarray): Energies [GeV], shape (n_E,), along the last axis of the z grids.
            z_min_grid (np.ndarray): Slab top along the path [m.w.e.], shape (..., n_E).
            z_max_grid (np.ndarray): Slab bottom along the path [m.w.e.], same shape.

        Returns:
            dict: {category: probability, same shape as z_min_grid}.
        """
        q_rows, tails = self._depth_rows(species, e_vals)
        p = self._depth_table(species)['p']
        F_lo, G_lo = _depth_cdf_rows(z_min_grid, q_rows, p, tails)
        F_hi, G_hi = _depth_cdf_rows(z_max_grid, q_rows, p, tails)
        if species not in MUON_SPECIES:
            return {'all': F_hi - F_lo}
        mean_range = _depth_cdf_rows(np.full(len(e_vals), np.inf), q_rows, p)[1]
        path = (G_hi - G_lo) / mean_range
        stop = F_hi - F_lo
        if species == 'mu-':
            return {'elastic': path, 'peak': stop}
        return {'all': path}


    def _flux_time_kyr(self, t_kyr):
        """Converts local exposure time to the FluxHistory timeline.

        Local time runs from 0 (start of exposure) to total_age_kyr (present); the
        FluxHistory timeline has 0 at present and negative values in the past. This
        is the only place where the mapping is done.

        Args:
            t_kyr (float or np.ndarray): Local time [kyr].

        Returns:
            float or np.ndarray: Time on the FluxHistory timeline [kyr].
        """
        return t_kyr - self.total_age_kyr


    def _secondary_window_mwe(self, product='neutron'):
        """Width of the secondary production window above the sample.

        The depth beyond which a product particle survives with probability below
        SECONDARY_WINDOW_SURVIVAL, from its StdRock table (empirical nodes up to
        NEUTRON_TAIL_P, exponential tail beyond), maximised over the tabulated energies.

        Args:
            product (str, optional): 'neutron' or 'proton'. Defaults to 'neutron'.

        Returns:
            float: Window width along the path [m.w.e.].
        """
        table = self._depth_table(product)
        survive_last = 1.0 - table['p'][-1]
        depth = table['q'][:, -1] + table['tail_lambda'] * np.log(
            survive_last / SECONDARY_WINDOW_SURVIVAL
        )
        return float(np.max(depth))


    def _stdrock_dir(self, species):
        """StdRock run directory of a primary species, or None if there is none.

        mu+ falls back on the mu- runs when StdRock/mu+ does not exist (see _load_stdrock_rows).

        Args:
            species (str): Primary species.

        Returns:
            tuple[str, bool] or None: (directory, drop_stopping), drop_stopping being True for the mu+ fallback.
        """
        root = os.path.join(self.data_path, "Geant4_data", "StdRock")
        if _list_geant4_energy_files(os.path.join(root, species)):
            return os.path.join(root, species), False
        if species == 'mu+' and _list_geant4_energy_files(os.path.join(root, 'mu-')):
            return os.path.join(root, 'mu-'), True
        return None


    def _load_stdrock_rows(self, species, product, product_bins_gev, total_simulated_particles=1e4,
                           primary_bins_gev=None):
        """Particles of type `product` produced by the primaries of the StdRock runs.

        Depths are absolute (from the surface, where the primary enters) and already include the transport of the
        primary through the rock. For mu+, StdRock/mu+ is used if present; otherwise the mu- runs without the
        stopping step (no capture products) stand in for it.

        The primary energy cells are those of `primary_bins_gev`; each cell uses the StdRock file at the largest
        energy not above its lower edge, so the StdRock grid need not match. By default the cells are the StdRock
        files' own grid (_geant4_energy_edges), which keeps every simulated primary energy. The result depends only
        on the files, their geometry and the arguments, so it is cached (shared cache, see _FILE_CACHE); the files
        are read by _read_stdrock_rows.

        Args:
            species (str): Primary species ('mu-', 'mu+', 'neutron', 'proton').
            product (str): Produced particle to collect ('neutron' or 'proton').
            product_bins_gev (np.ndarray): Energy cell edges of the products [GeV].
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            primary_bins_gev (np.ndarray, optional): Energy cell edges of the primaries [GeV]. Defaults to None
                (the StdRock grid of the species).

        Returns:
            dict: 'depth_mwe' (sorted production depths), 'cell' (product energy cell), 'primary_cell',
            'primary_bins' (primary cell edges [GeV]), 'weight' (1 / primaries per run), 'target_mwe' (StdRock
            target length [m.w.e.]) and 'missing' (primary cells below the first StdRock file).
        """
        found = self._stdrock_dir(species)
        if found is None:
            raise FileNotFoundError(f"No StdRock runs for {species}.")
        data_dir, drop_stopping = found
        files = _list_geant4_energy_files(data_dir)

        cache_key = ('stdrock_rows', tuple(_file_stamp(f) for _, f in files),
                     _file_stamp(os.path.join(data_dir, "geometry.json")), drop_stopping, product,
                     np.asarray(product_bins_gev, dtype=float).tobytes(), float(total_simulated_particles),
                     None if primary_bins_gev is None else np.asarray(primary_bins_gev, dtype=float).tobytes())
        hit, cached = _cache_get(cache_key)
        if hit:
            return cached
        rows = self._read_stdrock_rows(data_dir, drop_stopping, files, product, product_bins_gev,
                                       total_simulated_particles, primary_bins_gev)
        _cache_put(cache_key, rows)
        return rows


    def _read_stdrock_rows(self, data_dir, drop_stopping, files, product, product_bins_gev, total_simulated_particles,
                           primary_bins_gev):
        """Reads the rows of _load_stdrock_rows from the files (see there)."""
        file_e = np.array([e for e, _ in files])
        if primary_bins_gev is None:
            info = _geant4_energy_edges(data_dir)
            primary_bins_gev = info['edges'] if info is not None else np.array([file_e[0], file_e[0] * 1.1])
        primary_bins_gev = np.asarray(primary_bins_gev, dtype=float)
        product_bins_gev = np.asarray(product_bins_gev, dtype=float)

        geo = self._geant4_geometry(data_dir)
        rho = geo["density_g_cm3"] if geo["density_g_cm3"] is not None else 2.65
        z_ref_m = geo["depth_reference_z_m"]

        n_p, n_e = len(primary_bins_gev) - 1, len(product_bins_gev) - 1
        which = np.searchsorted(file_e, primary_bins_gev[:-1] * (1 + _ENERGY_EDGE_RTOL), side='right') - 1
        cache, depth_list, cell_list, prim_list, missing = {}, [], [], [], []
        for i in range(n_p):
            if which[i] < 0:
                missing.append(primary_bins_gev[i])
                continue
            filepath = files[which[i]][1]
            if filepath not in cache:
                cache[filepath] = None
                if os.path.getsize(filepath) > 0:
                    df = pd.read_csv(
                        filepath, sep=r'\s+', header=None, usecols=[0, 2, 3, 5],
                        names=['name', 'rec_e', 'z_mm', 'rem_e'],
                        dtype={'name': 'category', 'rec_e': 'float64', 'z_mm': 'float64', 'rem_e': 'float64'},
                        engine='c',
                    )
                    keep = (df['name'] == product).to_numpy()
                    if drop_stopping:
                        keep = keep & (df['rem_e'].to_numpy() != 0.0)
                    depth = np.clip((z_ref_m - df['z_mm'].values[keep] * 1e-3) * rho, 0.0, None)
                    cell = np.searchsorted(product_bins_gev, df['rec_e'].values[keep] * 1e-3, side='right') - 1
                    inside = (cell >= 0) & (cell < n_e)
                    cache[filepath] = (depth[inside], cell[inside])
            if cache[filepath] is None:
                continue
            depth, cell = cache[filepath]
            depth_list.append(depth)
            cell_list.append(cell)
            prim_list.append(np.full(depth.size, i))

        depth = np.concatenate(depth_list) if depth_list else np.zeros(0)
        cell = np.concatenate(cell_list) if cell_list else np.zeros(0, dtype=int)
        prim = np.concatenate(prim_list) if prim_list else np.zeros(0, dtype=int)
        order = np.argsort(depth, kind='stable')
        return {
            'depth_mwe': depth[order], 'cell': cell[order], 'primary_cell': prim[order],
            'primary_bins': primary_bins_gev, 'weight': 1.0 / total_simulated_particles,
            'target_mwe': geo["target_length_m"] * rho, 'missing': missing,
        }


    def _stdrock_production_matrix(self, source, product, source_bins_gev, product_bins_gev,
                                   total_simulated_particles=1e4, delta_mwe=None, n_offsets=None):
        """Particles of type `product` made by a `source` primary in the StdRock runs, per primary, by energy cell
        and, optionally, by distance travelled: one block of a cascade's generation matrix.

        source == product gives the same-type generations, source != product the cross-generations (e.g. neutrons
        from protons). Called by:
          * _secondary_cascade, with a depth binning: C[l, d, m] is the number of products in energy cell m made
            between d * delta_mwe and (d + 1) * delta_mwe after the entry point, per primary in cell l (products
            beyond n_offsets bins are left out);
          * _neutrino_product_operator (medium 'StdRock'), without: G[l, m] counts the products made anywhere in
            the StdRock target.

        Args:
            source (str): Primary of the StdRock runs ('mu-', 'mu+', 'neutron', 'proton'; see _load_stdrock_rows).
            product (str): Produced particle.
            source_bins_gev (np.ndarray): Energy cell edges of the source [GeV].
            product_bins_gev (np.ndarray): Energy cell edges of the product [GeV].
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            delta_mwe (float, optional): Width of the distance bins [m.w.e.]. Defaults to None (no distance axis).
            n_offsets (int, optional): Number of distance bins (with delta_mwe).

        Returns:
            np.ndarray: Shape (n_source_cells, n_offsets, n_product_cells), or (n_source_cells, n_product_cells)
            without delta_mwe.
        """
        rows = self._load_stdrock_rows(source, product, product_bins_gev, total_simulated_particles,
                                       primary_bins_gev=source_bins_gev)
        n_s, n_q = len(source_bins_gev) - 1, len(product_bins_gev) - 1
        if delta_mwe is None:
            flat = rows['primary_cell'] * n_q + rows['cell']
            return np.bincount(flat, minlength=n_s * n_q).astype(float).reshape(n_s, n_q) * rows['weight']
        d = np.floor(rows['depth_mwe'] / delta_mwe).astype(int)
        ok = d < n_offsets
        flat = (rows['primary_cell'][ok] * n_offsets + d[ok]) * n_q + rows['cell'][ok]
        C = np.bincount(flat, minlength=n_s * n_offsets * n_q).astype(float)
        return C.reshape(n_s, n_offsets, n_q) * rows['weight']


    @staticmethod
    def _window_source_shared(A, times, rows, lo_idx, hi_idx, beyond, window_top, phi_cell, delta, n_b, n_q,
                              product_slice):
        """Source term of _secondary_cascade for several times at the same overburden, added to A.

        Same window, binning and rows as the per-time loop (see there), but the rows are counted once per direction,
        by primary energy cell i, window bin b and product cell k (N[i, b, k]); each time t then gets
        weight * sum_i phi_cell[t, i] N[i, b, k], a matrix product. Equal to the per-time sums up to the order of
        the floating-point additions.

        Args:
            A (np.ndarray): Source array (T, N_C, n_b, K), updated in place.
            times (np.ndarray): Indices of the times sharing this overburden.
            rows (dict): Output of _load_stdrock_rows.
            lo_idx, hi_idx, beyond, window_top (np.ndarray): Their rows for these times, shape (N_C,).
            phi_cell (np.ndarray): Primary intensity per energy cell, (T, n_primary_cells).
            delta (float): Window bin width [m.w.e.].
            n_b, n_q (int): Window bins and product cells.
            product_slice (slice): Columns of this product on the last axis of A.
        """
        depth, prim, cell = rows['depth_mwe'], rows['primary_cell'], rows['cell']
        n_prim = len(rows['primary_bins']) - 1
        phi = phi_cell[times]
        for c in range(len(lo_idx)):
            if beyond[c] or hi_idx[c] <= lo_idx[c]:
                continue
            sl = slice(lo_idx[c], hi_idx[c])
            b = np.minimum(((depth[sl] - window_top[c]) / delta).astype(int), n_b - 1)
            counts = np.bincount(prim[sl] * (n_b * n_q) + b * n_q + cell[sl],
                                 minlength=n_prim * n_b * n_q).reshape(n_prim, n_b * n_q)
            used = np.nonzero(counts.any(axis=1))[0]
            A[times, c, :, product_slice] += (rows['weight'] * (phi[:, used] @ counts[used])).reshape(
                len(times), n_b, n_q)


    def _secondary_cascade(self, t_kyr_array, depth_mwe_array, c_vals, product_bins_gev,
                           total_simulated_particles=1e4, species_list=SECONDARY_SOURCES):
        """Secondary neutrons and protons born in a window above the sample, per direction, as one cascade: its source
        and its response. Called by _secondary_recoils.

        For each time t and direction c, the path to the sample has slant depth S = X(t) / c. The window
        [S - W, S] along the path (W: the largest _secondary_window_mwe of the products) is split into
        SECONDARY_WINDOW_BINS bins of width delta = W / SECONDARY_WINDOW_BINS.

        State: the products of every type on one flat axis, product q occupying offsets[q] (its energy cells).

        Source: products made in the window by the primaries, from the StdRock runs at their actual slant depth
        (so the transport of the primary down to the window is included), weighted by the primary intensity
        integrated over each energy cell of the primary's StdRock grid:

            A_b,k = sum_s sum_i Phi_s,i(t) * (StdRock/<s> products of cell k in bin b per primary of cell i)

        Generations inside the window: the StdRock/<p> runs of every product p give the products of every type
        q it makes (_stdrock_production_matrix blocks C[p -> q]), so neutrons from neutrons, protons from protons and
        the cross-generations (neutrons from protons, protons from neutrons) are all followed, to any order.
        Products are born uniformly within their bin, so a production at offset bin d is shared between bins
        b' + d and b' + d + 1:

            Y_b (I - C[0] / 2) = A_b + sum_{b' < b} Y_b' (C[b - b' - 1] + C[b - b']) / 2

        This recursion is linear and the same at every bin, so its solution is a convolution of the source with
        response blocks M_d (K x K) that do not depend on time or direction:

            Y_b = sum_{b' <= b} A_b' M_{b - b'},   M_0 = S,   M_d = sum_{d'=1..d} M_{d - d'} T_d' S,

        with S = (I - C[0] / 2)^-1 and T_d = (C[d - 1] + C[d]) / 2. The cascade is returned as (A, M): Y itself, shape
        (T, N_C, SECONDARY_WINDOW_BINS, K), is never formed, since only its products with the slab probability are
        needed (see _secondary_recoils).

        Secondaries are assumed to keep the direction of the primary. Sources without StdRock runs are skipped;
        a product without StdRock runs makes no further generations (reported).

        Computation of the source: the flux per primary energy cell is computed once per source species (every
        product of a species uses the same primary grid). Times at the same overburden have the same window and
        binning, so their rows are binned once, by primary energy cell, and each time weights the counts with its
        own flux (_window_source_shared); the other times go through _window_source_numba (or the numpy loop if numba
        is missing).

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            depth_mwe_array (np.ndarray): Overburden [m.w.e.], shape (T,).
            c_vals (np.ndarray): cos(zenith) grid, shape (N_C,).
            product_bins_gev (dict): {product: energy cell edges [GeV]} of the products followed ('neutron',
                'proton'; see _secondary_grids).
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            species_list (tuple[str], optional): Primary species producing the secondaries.
                Defaults to SECONDARY_SOURCES.

        Returns:
            tuple[np.ndarray, np.ndarray, float, dict]:
                A: products made by the primaries in each window bin and cell [cm^-2 s^-1], shape
                (T, N_C, SECONDARY_WINDOW_BINS, K).
                M: response blocks, shape (SECONDARY_WINDOW_BINS, K, K).
                delta: window bin width along the path [m.w.e.].
                offsets: {product: slice of the product axis}.
        """
        products = list(product_bins_gev)
        product_bins_gev = {q: np.asarray(e, dtype=float) for q, e in product_bins_gev.items()}
        offsets, n_k = _product_axis(product_bins_gev)

        n_t, n_c = len(t_kyr_array), len(c_vals)
        n_b = SECONDARY_WINDOW_BINS
        window = max(self._secondary_window_mwe(q) for q in products)
        delta = window / n_b

        slant = depth_mwe_array[:, None] / c_vals[None, :]
        window_top = slant - window

        A = np.zeros((n_t, n_c, n_b, n_k))
        _, depth_group = np.unique(np.asarray(depth_mwe_array, dtype=float), return_inverse=True)
        same_depth = [np.nonzero(depth_group == g)[0] for g in range(depth_group.max() + 1)]
        for species in species_list:
            if self._stdrock_dir(species) is None:
                self._log(f"No StdRock runs for {species}: no secondaries from it.", level=2)
                continue
            phi_cell = None
            for q in products:
                rows = self._load_stdrock_rows(species, q, product_bins_gev[q], total_simulated_particles)
                if phi_cell is None:
                    if rows['missing']:
                        self._log(f"Warning: StdRock/{species}: no file for {len(rows['missing'])} primary "
                                  f"energies (e.g. {rows['missing'][:3]} GeV); their secondaries are missing.")
                    beyond = slant > TRUNCATION_FRACTION * rows['target_mwe']
                    if np.any(beyond[:, -1]):
                        self._log(f"Warning: vertical overburden beyond the StdRock/{species} target "
                                  f"({rows['target_mwe']:.0f} m.w.e.) at some times: secondaries "
                                  "from that species are missing there; use a longer StdRock target.")
                    phi_cell = self._flux_cell_integral(species, t_kyr_array, rows['primary_bins'])

                n_q = len(product_bins_gev[q]) - 1
                depth = rows['depth_mwe']
                lo_idx = np.searchsorted(depth, window_top, side='left')
                hi_idx = np.searchsorted(depth, slant, side='left')
                single = np.zeros(n_t, dtype=bool)
                for times in same_depth:
                    if times.size == 1:
                        single[times] = True
                    else:
                        self._window_source_shared(A, times, rows, lo_idx[times[0]], hi_idx[times[0]],
                                                   beyond[times[0]], window_top[times[0]], phi_cell, delta, n_b,
                                                   n_q, offsets[q])
                if not single.any():
                    continue
                if _numba is not None:
                    _window_source_numba(A, depth, rows['primary_cell'], rows['cell'], lo_idx, hi_idx, beyond,
                                         window_top, phi_cell, float(rows['weight']), float(delta), n_b, n_q,
                                         offsets[q].start, single)
                    continue
                for t in np.nonzero(single)[0]:
                    for c in range(n_c):
                        if beyond[t, c] or hi_idx[t, c] <= lo_idx[t, c]:
                            continue
                        sl = slice(lo_idx[t, c], hi_idx[t, c])
                        b = np.minimum(((depth[sl] - window_top[t, c]) / delta).astype(int), n_b - 1)
                        w = phi_cell[t, rows['primary_cell'][sl]] * rows['weight']
                        A[t, c, :, offsets[q]] += np.bincount(
                            b * n_q + rows['cell'][sl], weights=w, minlength=n_b * n_q
                        ).reshape(n_b, n_q)

        C = np.zeros((n_k, n_b + 1, n_k))
        for p in products:
            if self._stdrock_dir(p) is None:
                self._log(f"Warning: no StdRock/{p} runs: the secondaries made by {p}s are not followed.")
                continue
            for q in products:
                C[offsets[p], :, offsets[q]] = self._stdrock_production_matrix(
                    p, q, product_bins_gev[p], product_bins_gev[q], total_simulated_particles, delta, n_b + 1)
        half_same_bin = 0.5 * C[:, 0, :]
        if np.max(np.abs(np.linalg.eigvals(half_same_bin))) < 1.0:
            same_bin_inv = np.linalg.inv(np.eye(n_k) - half_same_bin)
        else:
            self._log("Warning: same-bin secondary multiplication >= 1; ignoring it.")
            same_bin_inv = np.eye(n_k)
        transfer = [None] + [0.5 * (C[:, k - 1, :] + C[:, k, :]) for k in range(1, n_b)]

        M = np.empty((n_b, n_k, n_k))
        M[0] = same_bin_inv
        for d in range(1, n_b):
            M[d] = sum(M[d - dp] @ transfer[dp] for dp in range(1, d + 1)) @ same_bin_inv
        return A, M, delta, offsets


    def _primary_recoils(self, t_kyr_array, energy_bins_gev=None, total_simulated_particles=1e4, species='mu-'):
        """Recoils in the sample slab made by one primary cosmic-ray species (mu-, mu+, neutron, proton).

        The sample is a slab [X, X + thickness] at overburden X(t), seen along each slanted path (both divided by
        cos(zenith)); its thickness is that of a 1 mg cube. For each energy cell of the mineral runs, the flux is
        integrated over the cell (_energy_cell_nodes) and over cos(zenith) (COS_ZENITH_NODES) times the probability
        of an interaction in the slab, one per category of recoils (_slab_interaction_probabilities), and multiplied
        by the recoils per primary of the run at the lower edge of the cell in that category
        (_mineral_run_histograms): 'all' for mu+ and hadrons, 'elastic' and 'peak' for mu-.

        mu- captures ('peak', the muon stops in the slab) use one capture spectrum for every energy: the mean of the
        peak recoils of the runs with E0 <= CAPTURE_SPECTRUM_MAX_E0_GEV (a stopped muon does not remember its initial
        energy, and the low-energy runs have the most stops), times the total stopping probability in the slab
        summed over all cells. Without such runs each run's own peak recoils are used (warning).

        All times are processed at once and each file is read once. The slab probabilities depend on time only
        through the overburden, so they are computed once per distinct overburden of the time grid (one for a
        constant overburden) and shared by the times that have it. Called by integrate_particle_signal_spectrum (also
        through integrate_all_particles).

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            energy_bins_gev (np.ndarray, optional): Primary energy cell edges [GeV]. Defaults to None (all the
                Geant4 files of the species).
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            species (str, optional): 'mu-', 'mu+', 'neutron' or 'proton'. Defaults to 'mu-'.

        Returns:
            dict: Recoil spectra (see _recoil_spectra), shape (T, n_recoil_bins) per fragment.

        Raises:
            ValueError: If flux_history is not set or the species has no depth table.
        """
        energy_bins_gev = self._energy_edges(species, edges=energy_bins_gev)
        if self.flux_history is None:
            raise ValueError("flux_history not initialized: use set_flux_history() or an integrate_* function.")
        if not self._has_depth_table(species):
            raise ValueError(f"No StdRock depth table for {species}: it cannot be computed (see available_species).")

        rho = self.config['density_g_cm3']
        thickness_cm = np.power(1.e-3 / rho, 1 / 3)
        thickness_mwe = thickness_cm * 0.01 * rho
        t_kyr_array = np.atleast_1d(np.asarray(t_kyr_array, dtype=float))
        n_t = t_kyr_array.size
        depth_mwe, depth_of_time = np.unique(np.asarray(self._overburden_interpolator(t_kyr_array), dtype=float),
                                             return_inverse=True)

        c_vals = COS_ZENITH_NODES
        e_nodes, e_weights = _energy_cell_nodes(energy_bins_gev)
        flux_nodes = self.flux_history.get_map(species, self._flux_time_kyr(t_kyr_array), e_nodes)[2].reshape(
            n_t, *e_nodes.shape)
        run_dir = os.path.join(self.data_path, "Geant4_data", self.name, species)

        fragment_spectra = {}
        peak_per_run, capture_counts = {}, {}
        stop_weight_total, n_capture_runs = np.zeros(n_t), 0
        for i, e_lo in enumerate(energy_bins_gev[:-1]):
            E_grid, C_grid = np.meshgrid(e_nodes[i], c_vals)
            z_min_grid = depth_mwe[:, None, None] / C_grid[None, :, :]
            z_max_grid = z_min_grid + thickness_mwe / C_grid[None, :, :]
            flux_grid = np.broadcast_to(flux_nodes[:, None, i, :], (n_t, *E_grid.shape))
            probabilities = self._slab_interaction_probabilities(species, e_nodes[i], z_min_grid, z_max_grid)
            weights = {category: np.trapezoid(np.sum(flux_grid * prob[depth_of_time] * e_weights[i], axis=-1),
                                              c_vals, axis=-1)
                       for category, prob in probabilities.items()}
            if 'peak' in weights:
                stop_weight_total += weights['peak']

            hist = _mineral_run_histograms(os.path.join(run_dir, f"outNuclei_{e_lo:.6f}.txt"))
            if hist is None:
                continue
            for_capture = 'peak' in weights and e_lo <= CAPTURE_SPECTRUM_MAX_E0_GEV
            n_capture_runs += for_capture
            for frag in sorted(hist['all']):
                for category, w in weights.items():
                    counts = hist[category][frag]
                    if counts is None:
                        continue
                    if category == 'peak':
                        _accumulate_into(peak_per_run, frag, np.outer(w, counts))
                        if for_capture:
                            _accumulate_into(capture_counts, frag, counts)
                    else:
                        _accumulate_into(fragment_spectra, frag, np.outer(w, counts))

        if species == 'mu-':
            if n_capture_runs > 0:
                for frag, counts in capture_counts.items():
                    _accumulate_into(fragment_spectra, frag, np.outer(stop_weight_total, counts / n_capture_runs))
            else:
                self._log(f"Warning: no mu- run with E0 <= {CAPTURE_SPECTRUM_MAX_E0_GEV} GeV; "
                          "using the per-energy capture spectra.")
                for frag, spectrum in peak_per_run.items():
                    _accumulate_into(fragment_spectra, frag, spectrum)

        per_g_kyr = SECONDS_PER_KYR / (thickness_cm * rho * total_simulated_particles)
        return _recoil_spectra({frag: spectrum * per_g_kyr for frag, spectrum in fragment_spectra.items()},
                               t_kyr_array)


    def _secondary_products(self):
        """Products of the atmospheric secondary cascade that can be followed: the DERIVED_SPECIES products with
        mineral runs and a StdRock depth table (i.e. whose secondary species is in available_species())."""
        return [DERIVED_SPECIES[d] for d in self.available_species() if d in DERIVED_SPECIES]


    def _secondary_grids(self, requested=None):
        """Cascade energy grids of the secondary products, and the energy window each requested product reports.

        The cascade always contains every followable product (cross-generations) on its full mineral run grid, so
        that restricting the energies reported for one secondary species does not change the cascade, and hence
        the other species. A request that is a contiguous part of the full grid (as given by e_range, see
        _energy_edges) only sets the reporting window; any other set of edges replaces that product's grid.

        Args:
            requested (dict, optional): {product: energy cell edges [GeV] or None}. Defaults to None.

        Returns:
            tuple[dict, dict]: {product: cascade cell edges [GeV]} and {product: (E_lo, E_hi) [GeV]} for the
            products whose reported cells are restricted.

        Raises:
            ValueError: If a requested product cannot be followed (no mineral runs or no StdRock depth table).
        """
        products = self._secondary_products()
        requested = {q: e for q, e in (requested or {}).items() if e is not None}
        missing = [q for q in requested if q not in products]
        if missing:
            raise ValueError(f"Secondary {missing} cannot be computed for {self.name}: need mineral runs and a "
                             f"StdRock depth table (followable: {products}).")
        grids, windows = {}, {}
        for q in products:
            full = self._energy_edges(q)
            if q not in requested:
                grids[q] = full
                continue
            edges = np.asarray(requested[q], dtype=float)
            on_grid = np.all(np.min(np.abs(full[None, :] / edges[:, None] - 1.0), axis=1) <= _ENERGY_EDGE_RTOL)
            if on_grid:
                grids[q] = full
                windows[q] = (edges[0], edges[-1])
            else:
                grids[q] = edges
        return grids, windows


    def _secondary_recoils(
        self, t_kyr_array, product_bins_gev=None,
        total_simulated_particles=1e4, secondary_sources=SECONDARY_SOURCES, energy_windows=None,
    ):
        """Recoils in the sample slab made by the atmospheric secondary neutrons and protons, all products at once.

        Called by _integrate_secondary_species (one solve of the cascade for every secondary species requested).

        Products born in the window above the sample (see _secondary_cascade, which follows all generations and
        cross-generations) travel along the primary's direction to the slab. For a product of type q born
        uniformly in window bin b, at distance d in [d0, d0 + delta] from the slab top, the probability of
        interacting in the slab (path tau = t_slab / c) is averaged over the bin exactly:

            P_b = [H(d0 + delta + tau) - H(d0 + tau) - H(d0 + delta) + H(d0)] / delta,

        with H(z) = integral_0^z F = z - G(z), F being the StdRock depth CDF of q. Then

            w_m(t) = int dc sum_b Y[t, c, b, m] P_b(E_m, c),

        and the recoils per product of the mineral run of q at the lower edge E_m of each cell are applied with
        the normalisation of _primary_recoils.

        The cascade comes as its source A and response blocks M (Y_b = sum_{b' <= b} A_b' M_{b - b'}), and P does not
        depend on time, so the sum over b is done backwards (adjoint form) instead of building Y:

            sum_b Y_b[k] P_b[k] = sum_{b', j} A_b'[j] V_b'[j, k],   V_b'[j, k] = sum_{d >= 0} M_d[j, k] P_{b' + d}[k],

        V once per direction, then one matrix product per direction for all the times. Equal to building Y up to
        the order of the floating-point additions.

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            product_bins_gev (dict, optional): {product: energy cell edges [GeV]} of the cascade. Defaults to
                None (_secondary_grids(): every followable product on its full grid).
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            secondary_sources (tuple[str], optional): Primary species producing the secondaries.
                Defaults to SECONDARY_SOURCES.
            energy_windows (dict, optional): {product: (E_lo, E_hi) [GeV]}: only the cells inside it give recoils
                (the cascade is unchanged). Defaults to None (all cells).

        Returns:
            dict: {product: recoil spectra (see _recoil_spectra), shape (T, n_recoil_bins) per fragment}.
        """
        if product_bins_gev is None:
            product_bins_gev = self._secondary_grids()[0]
        energy_windows = dict(energy_windows or {})
        rho = self.config['density_g_cm3']

        target_volume_cm3 = 1.e-3 / rho
        target_thickness_cm = np.power(target_volume_cm3, 1/3)
        target_thickness_mwe = target_thickness_cm * 0.01 * rho

        t_kyr_array = np.atleast_1d(np.asarray(t_kyr_array, dtype=float))
        depth_mwe_array = np.atleast_1d(self._overburden_interpolator(t_kyr_array))

        c_vals = COS_ZENITH_NODES
        A, M, delta, offsets = self._secondary_cascade(
            t_kyr_array, depth_mwe_array, c_vals, product_bins_gev,
            total_simulated_particles, secondary_sources,
        )
        n_t, n_c, n_b, n_k = A.shape
        d0 = (n_b - 1 - np.arange(n_b)) * delta
        tau = target_thickness_mwe / c_vals
        z_d0 = np.broadcast_to(d0[None, :], (n_c, n_b))
        z_tau = np.broadcast_to(tau[:, None], (n_c, n_b))

        P = np.empty((n_c, n_b, n_k))
        for product, energy_bins_gev in product_bins_gev.items():
            n_e = len(energy_bins_gev) - 1
            q_rows, tails = self._depth_rows(product, energy_bins_gev[:-1])
            p = self._depth_table(product)['p']

            def H(z):
                """z - G(z), the integral of the product's depth CDF from 0 to z, for every energy cell (last axis)."""
                z = np.broadcast_to(z[..., None], (*z.shape, n_e))
                return z - _depth_cdf_rows(z, q_rows, p, tails)[1]

            P[..., offsets[product]] = (H(z_d0 + delta + z_tau) - H(z_d0 + z_tau) - H(z_d0 + delta) + H(z_d0)) / delta

        W = np.empty((n_t, n_c, n_k))
        for c in range(n_c):
            V = np.zeros((n_b, n_k, n_k))
            for d in range(n_b):
                V[:n_b - d] += M[d][None, :, :] * P[c, d:, None, :]
            W[:, c, :] = A[:, c].reshape(n_t, n_b * n_k) @ V.reshape(n_b * n_k, n_k)

        per_g_kyr = SECONDS_PER_KYR / (target_thickness_cm * rho * total_simulated_particles)
        out = {}
        for product, energy_bins_gev in product_bins_gev.items():
            n_e = len(energy_bins_gev) - 1
            e_cells = energy_bins_gev[:-1]
            geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, product)
            weight_t = np.trapezoid(W[..., offsets[product]], c_vals, axis=1)
            if product in energy_windows:
                lo, hi = energy_windows[product]
                tol = 1 + _ENERGY_EDGE_RTOL
                outside = (energy_bins_gev[:-1] * tol < lo) | (energy_bins_gev[1:] > hi * tol)
                weight_t[:, outside] = 0.0

            fragment_spectra = {}
            for m in range(n_e):
                hist = _mineral_run_histograms(os.path.join(geant4_input_dir, f"outNuclei_{e_cells[m]:.6f}.txt"))
                if hist is None:
                    continue
                for frag in sorted(hist['all']):
                    _accumulate_into(fragment_spectra, frag, np.outer(weight_t[:, m], hist['all'][frag]))
            out[product] = _recoil_spectra({frag: spectrum * per_g_kyr for frag, spectrum in fragment_spectra.items()},
                                           t_kyr_array)
        return out

    def _report_missing_range_tables(self, spectra, missing, label):
        """Warns, once per mineral and fragment, about recoils skipped for lack of a range table, with their share.

        Called by _recoils_to_track_density, which skips fragments whose DEDX_Z<z>_A<a>.txt is not in <mineral>/Range
        (export_all_stopping_powers only exports isotopes known to mendeleev, so unbound fragments such as Be4 or He5,
        occasionally emitted by the Geant4 de-excitation models, have none). The share of the recoils they carry tells
        whether that is harmless (a tiny fraction) or a real gap (a missing table of a common nucleus, which would
        need export_all_stopping_powers again). The recoil-energy observable needs no table and keeps them.
        """
        reported = self.__dict__.setdefault('_missing_range_reported', set())
        new = [f for f in missing if f not in reported]
        if not new:
            return
        reported.update(new)
        fragments = spectra['fragments']

        def n_recoils(f):
            return float(np.sum(np.atleast_2d(fragments[f]).mean(axis=0)))

        total = sum(n_recoils(f) for f in fragments)
        shares = ", ".join(f"{f} {100 * n_recoils(f) / total:.2g}%" if total > 0 else f for f in new)
        print(f"Warning: {self.name}/Range has no table for {shares} of the {label} recoils; skipped "
              f"(export_all_stopping_powers exports only the isotopes known to mendeleev).")


    def _recoils_to_track_density(self, spectra, x_grid, label):
        """Track-length density of each recoiling fragment: one track per recoil. Called by _recoil_density.

        For each fragment, dR/dx = dR/dEr(E(x)) * dE/dx(x), where E(x) inverts the CSDA range of the mineral's range
        table of the fragment and dE/dx is the total (electronic + nuclear) stopping power, both from
        <mineral>/Range/DEDX_Z<z>_A<a>.txt; dR/dEr is the recoil spectrum divided by the bin widths, interpolated
        between bin centres. Fragments without a range table are skipped (_report_missing_range_tables).

        Args:
            spectra (dict): Recoil spectra (_recoil_spectra).
            x_grid (np.ndarray): Fine track-length grid [nm].
            label (str): Source name, for the warning.

        Returns:
            dict: {fragment: dR/dx [tracks per mg per kyr per nm] on x_grid}, shape (T, n_x), or (n_x,) for a constant
            source.
        """
        er_bins = RECOIL_ENERGY_BINS_MEV
        widths = np.diff(er_bins)
        er_mid_mev = er_bins[:-1] + widths / 2.0
        x_mid_nm = x_grid[:-1] + np.diff(x_grid) / 2.0
        range_dir = os.path.join(self.data_path, "Geant4_data", self.name, "Range")

        out, missing = {}, []
        for name, counts in sorted(spectra['fragments'].items()):
            za = fragment_za(name)
            filepath = None if za is None else os.path.join(range_dir, f"DEDX_Z{za[0]}_A{za[1]}.txt")
            if filepath is None or not os.path.exists(filepath):
                missing.append(name)
                continue
            e_mev, dee_dx_mev_um, den_dx_mev_um, length_um = _range_table(filepath)
            order = np.argsort(length_um)
            e_at_x = np.interp(x_mid_nm, length_um[order] * 1e3, e_mev[order], left=0.0, right=0.0)
            dedx_at_x = np.interp(x_mid_nm, length_um[order] * 1e3,
                                  (dee_dx_mev_um[order] + den_dx_mev_um[order]) * 1e-3, left=0.0, right=0.0)
            dRdEr = interp1d(er_mid_mev, np.asarray(counts, dtype=float) / widths, axis=-1, bounds_error=False,
                             fill_value=0.0)
            out[name] = dRdEr(e_at_x) * dedx_at_x * 1e-3
        if missing and self.verbose > 0:
            self._report_missing_range_tables(spectra, missing, label)
        return out


    def _recoil_density(self, spectra, observable, grid, label):
        """Density of the observable for each recoiling fragment, the second step of every source.

        tracks: _recoils_to_track_density (one track per recoil). recoil_energy: the recoil spectrum per MeV, on its own
        grid RECOIL_ENERGY_BINS_MEV. Called by _recoils_to_counts and integrate_radiogenic.

        Args:
            spectra (dict): Recoil spectra (_recoil_spectra).
            observable (str): 'tracks' or 'recoil_energy'.
            grid (np.ndarray): Fine grid of the observable (from _output_axes).
            label (str): Source name, for warnings.

        Returns:
            dict: {fragment: density [per mg per kyr per nm, or per MeV]}, shape (T, n_grid) or (n_grid,).
        """
        if observable == OBSERVABLE_RECOIL_ENERGY:
            widths = np.diff(RECOIL_ENERGY_BINS_MEV)
            return {k: np.asarray(v, dtype=float) / widths * 1e-3 for k, v in sorted(spectra['fragments'].items())}
        return self._recoils_to_track_density(spectra, grid, label)


    def _with_track_sums(self, counts, n_bins, group=None):
        """Adds the sums 'total', 'total_no_light_ions' and one entry per track group to a dict of counts per key.

        Called by _density_to_counts, for every source. Each key is placed in its group by its Z (track_group, with
        the thresholds of set_track_groups) and counted as light if it is H or He (is_light_ion), unless group is
        given: then every key goes to that group (the fission fragments, whose keys can be pairs) and is light only
        if group is 'light'.

        Args:
            counts (dict): {fragment key: counts per output bin}, shape (n_bins,).
            n_bins (int): Number of output bins.
            group (str, optional): Group of every key. Defaults to None (by Z).

        Returns:
            dict: counts plus the sums and the groups.
        """
        total, no_light = np.zeros(n_bins), np.zeros(n_bins)
        groups = {g: np.zeros(n_bins) for g in TRACK_GROUPS}
        for key, c in sorted(counts.items()):
            total += c
            if group is None:
                if not is_light_ion(key):
                    no_light += c
                groups[self.fragment_track_group(key)] += c
            else:
                if group != 'light':
                    no_light += c
                groups[group] += c
        return {**counts, TOTAL_ALL_FRAGMENTS: total, TOTAL_NO_LIGHT_IONS: no_light, **groups}


    @staticmethod
    def _select_nucleus(result, nucleus, label):
        """Picks the requested sum, group set or fragment from a full result (_with_track_sums), as the last step of
        every integrate_* function.

        Args:
            result (dict): Counts per fragment, sums and groups.
            nucleus (str): 'total_no_light_ions', 'total', 'groups' (dict of the track groups and both sums), 'all'
                (the whole dict) or a fragment key.
            label (str): Source name, for the error message.

        Returns:
            np.ndarray or dict: The selected counts, or a dict for 'groups' and 'all'.

        Raises:
            ValueError: If nucleus is not in the dict.
        """
        if nucleus == "all":
            return result
        if nucleus == TRACK_GROUPS_KEY:
            return {k: result[k] for k in (*TRACK_GROUPS, TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS)}
        if nucleus not in result:
            raise ValueError(f"nucleus '{nucleus}' not found: use one of {TRACK_SUMS}, '{TRACK_GROUPS_KEY}', 'all' or "
                             f"a fragment of {label} ({sorted(k for k in result if k not in TRACK_SUMS + TRACK_GROUPS)}).")
        return result[nucleus]


    def _density_to_counts(self, density, grid, x_bins, t_kyr, nucleus, label, group=None):
        """Counts per mg in the output bins from densities on the fine grid: the last step of every source.

        Each density is integrated over time (trapezoid on t_kyr, or times total_age_kyr for a constant source,
        t_kyr None) and rebinned to x_bins through its cumulative distribution (_rebin_density, exact for a density
        constant in each grid bin; counts outside the grid are zero). Then the sums and groups are added
        (_with_track_sums) and nucleus is selected (_select_nucleus).

        Args:
            density (dict): {key: density on grid}, shape (T, n_grid) or (n_grid,).
            grid (np.ndarray): Fine grid edges.
            x_bins (np.ndarray): Output bin edges.
            t_kyr (np.ndarray or None): Times of the rows [kyr], or None for a constant source.
            nucleus (str): See _select_nucleus.
            label (str): Source name, for the error message.
            group (str, optional): See _with_track_sums. Defaults to None.

        Returns:
            np.ndarray or dict: Counts per mg in each bin of x_bins.
        """
        counts = {}
        for key, d in density.items():
            accumulated = d * self.total_age_kyr if t_kyr is None else np.trapezoid(d, t_kyr, axis=0)
            counts[key] = _rebin_density(accumulated, grid, x_bins)
        return self._select_nucleus(self._with_track_sums(counts, len(x_bins) - 1, group), nucleus, label)


    def _recoils_to_counts(self, spectra, observable, x_bins, grid, nucleus, label):
        """Recoil spectra -> counts per mg in the output bins (_recoil_density, then _density_to_counts).

        Called by integrate_particle_signal_spectrum, _integrate_secondary_species and integrate_atmospheric_neutrinos.
        """
        density = self._recoil_density(spectra, observable, grid, label)
        return self._density_to_counts(density, grid, x_bins, spectra['t_kyr'], nucleus, label)


    def integrate_particle_signal_spectrum(
        self, x_bins=None, energy_bins_gev=None,
        exposure_window_kyr=None, total_simulated_particles=1e4,
        x_grid=TRACK_LENGTH_BINS_NM, species='mu-', nucleus=DEFAULT_TRACK_SUM, e_range=None,
        observable=OBSERVABLE_TRACKS,
    ):
        """Track counts (or recoil counts per recoil-energy bin) from one cosmic-ray species over an exposure
        window.

        The recoils are computed on the time grid of the window (_window_grid; _primary_recoils, or
        _integrate_secondary_species for a secondary species) and turned into counts by _recoils_to_counts. Called by
        integrate_all_particles (one worker per primary species), which is the usual entry point.

        Args:
            x_bins (np.ndarray, optional): Output bin edges: track lengths [nm] (default DEFAULT_INTEGRATION_X_BINS_NM)
                or recoil energies [MeV] (default RECOIL_ENERGY_BINS_MEV).
            energy_bins_gev (np.ndarray, optional): Primary energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            exposure_window_kyr (float or list[float], optional): Window in local
                time [kyr]; a single value w means [0, w]. Defaults to None
                ([0, total_age_kyr]).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            x_grid (np.ndarray, optional): Fine track-length grid [nm] (ignored for recoil energies). Defaults to
                TRACK_LENGTH_BINS_NM.
            species (str, optional): 'mu-', 'mu+', 'neutron', 'proton',
                'secondary_neutron' or 'secondary_proton'. Defaults to 'mu-'.
            nucleus (str, optional): 'total_no_light_ions' (all fragments except H and He), 'total', 'groups' (dict of
                the track groups and both sums, see set_track_groups), 'all' (dict of every fragment, the groups and
                both sums) or a fragment key. Defaults to 'total_no_light_ions'.
            e_range (tuple[float, float], optional): (E_min, E_max) [GeV]: use only the Geant4 files whose bins
                lie inside it (see _energy_edges). Ignored if energy_bins_gev is given. Defaults to None (all).
            observable (str, optional): 'tracks', or 'recoil_energy': the same recoils binned in their kinetic
                energy at creation [MeV] instead of their track length, with the same keys. Defaults to 'tracks'.

        Returns:
            tuple[np.ndarray, np.ndarray or dict]: x_bins, and tracks (or recoils) per mg in each bin.
        """
        x_bins, grid = _output_axes(observable, x_bins, x_grid)
        energy_bins_gev = self._energy_edges(species, e_range, energy_bins_gev)

        if species in DERIVED_SPECIES:
            return x_bins, self._integrate_secondary_species(
                [species], {species: energy_bins_gev}, x_bins=x_bins, exposure_window_kyr=exposure_window_kyr,
                total_simulated_particles=total_simulated_particles, x_grid=x_grid, nucleus=nucleus,
                observable=observable,
            )[species]

        t_kyr_array = self._window_grid(self._exposure_window(exposure_window_kyr))
        spectra = self._primary_recoils(t_kyr_array, energy_bins_gev, total_simulated_particles, species)
        return x_bins, self._recoils_to_counts(spectra, observable, x_bins, grid, nucleus, species)


    def _integrate_secondary_species(
        self, secondary_species, energy_bins_gev=None, x_bins=None, exposure_window_kyr=None,
        total_simulated_particles=1e4, x_grid=TRACK_LENGTH_BINS_NM, nucleus=DEFAULT_TRACK_SUM,
        observable=OBSERVABLE_TRACKS,
    ):
        """Track (or recoil-energy) counts of several atmospheric secondary species from one solve of the cascade.

        The neutron + proton cascade (_secondary_cascade) is computed once, with every followable product on its
        full grid, and each requested secondary species takes its own product from it (restricted to its energy
        range, if given, without changing the cascade: see _secondary_grids). Called by integrate_all_particles
        (one worker for all the secondary species) and integrate_particle_signal_spectrum.

        Args:
            secondary_species (list[str]): Keys of DERIVED_SPECIES ('secondary_neutron', 'secondary_proton').
            energy_bins_gev (dict, optional): {secondary species: energy cell edges [GeV]} of its product.
                Defaults to None (full grids).
            x_bins, exposure_window_kyr, total_simulated_particles, x_grid, nucleus, observable: See
                integrate_particle_signal_spectrum.

        Returns:
            dict: {secondary species: tracks (or recoils) per mg in each bin of x_bins}.
        """
        x_bins, grid = _output_axes(observable, x_bins, x_grid)
        energy_bins_gev = dict(energy_bins_gev or {})
        bins, windows = self._secondary_grids({DERIVED_SPECIES[s]: energy_bins_gev.get(s) for s in secondary_species})
        t_kyr_array = self._window_grid(self._exposure_window(exposure_window_kyr))
        recoils = self._secondary_recoils(t_kyr_array, bins, total_simulated_particles, energy_windows=windows)
        return {s: self._recoils_to_counts(recoils[DERIVED_SPECIES[s]], observable, x_bins, grid, nucleus, s)
                for s in secondary_species}


    def integrate_all_particles(
        self,  x_bins=None, e_range=None,
        exposure_window_kyr=None,
        total_simulated_particles=1e4, species_list=None, nucleus=DEFAULT_TRACK_SUM,
        flux_history=None, observable=OBSERVABLE_TRACKS,
    ):
        """Track counts (or recoil counts per recoil-energy bin) from several cosmic-ray species, computed in
        parallel.

        One worker process (_species_worker) per primary species, running integrate_particle_signal_spectrum, and one
        for all the secondary species together (_integrate_secondary_species: one solve of the cascade). The workers
        start with this process's Geant4 file cache and send back what they read, which is merged into it.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            e_range (dict, optional): {species: (E_min, E_max) [GeV]} restricting the Geant4 files used for those
                species (the bins lying entirely inside the range); species not in it use all their files. Raises
                if a range is outside the available one. Defaults to None (all files for every species).
            exposure_window_kyr (float or list[float], optional): See
                integrate_particle_signal_spectrum. Defaults to None.
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            species_list (tuple[str], optional): Species to compute. Defaults to None (available_species()).
            nucleus (str, optional): 'total_no_light_ions', 'total', 'groups', 'all' or a fragment key (see
                integrate_particle_signal_spectrum). Defaults to 'total_no_light_ions'.
            flux_history (FluxHistory, optional): Flux history to set for the integration. Defaults to None
            observable (str, optional): 'tracks' or 'recoil_energy' (x_bins then in MeV, default
                RECOIL_ENERGY_BINS_MEV; see integrate_particle_signal_spectrum). Defaults to 'tracks'.

        Returns:
            tuple[np.ndarray, dict]: x_bins, and tracks (or recoils) per mg in each bin for each
            species plus 'total'.

        Raises:
            ValueError: If a species has no runs, or e_range names a species not computed or an invalid range.
        """
        if flux_history is not None:
            self.set_flux_history(flux_history)

        x_bins, _ = _output_axes(observable, x_bins, TRACK_LENGTH_BINS_NM)
        species_list = self.available_species() if species_list is None else list(species_list)
        e_range = dict(e_range or {})
        unknown = set(e_range) - set(species_list)
        if unknown:
            raise ValueError(f"e_range given for species not computed: {sorted(unknown)} "
                             f"(computed: {species_list}).")
        edges = {s: self._energy_edges(s, e_range.get(s)) for s in species_list}

        shared_kwargs = dict(
            x_bins=x_bins,
            exposure_window_kyr=exposure_window_kyr,
            total_simulated_particles=total_simulated_particles, nucleus=nucleus, observable=observable,
        )

        primaries = [s for s in species_list if s not in DERIVED_SPECIES]
        secondaries = [s for s in species_list if s in DERIVED_SPECIES]
        tasks = [(self, dict(shared_kwargs, species=s, energy_bins_gev=edges[s])) for s in primaries]
        if secondaries:
            tasks.append((self, dict(shared_kwargs, secondary_species=secondaries,
                                     energy_bins_gev={s: edges[s] for s in secondaries})))

        tasks = [(task_self, dict(kwargs, _file_cache=dict(_FILE_CACHE))) for task_self, kwargs in tasks]
        with Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
            results = list(tqdm(pool.imap(_species_worker, tasks), total=len(tasks)))
        for _, new_entries in results:
            for key, value in new_entries.items():
                _cache_put(key, value)

        computed = dict(pair for result, _ in results for pair in result)
        total_tracks_by_species = {s: computed[s] for s in species_list}
        total_tracks_by_species['total'] = _combine_track_results(total_tracks_by_species.values())

        return x_bins, total_tracks_by_species


    # ------------------------------------------------------------------------------------------------------------
    # Atmospheric neutrinos
    # ------------------------------------------------------------------------------------------------------------

    def available_neutrino_species(self):
        """Neutrino species with mineral runs, i.e. those integrate_atmospheric_neutrinos can compute.

        Returns:
            list[str]: Entries of NEUTRINO_SPECIES found by _scan_geant4_species.
        """
        return [s for s in NEUTRINO_SPECIES if s in self.energy_bins]


    def _neutrino_bias(self, species, e0_gev, neutrino_bias=None, data_dir=None):
        """Cross-section bias factor of the neutrino run of `species` at primary energy e0_gev.

        The neutrino runs are made with the neutrino cross-sections multiplied by this factor inside the target
        (PhysicsList: SetNuNucleusBias / SetNuEleCcBias / SetNuEleNcBias, all equal), so their yields per primary are
        divided by it.

        Args:
            species (str): Neutrino species (NEUTRINO_SPECIES).
            e0_gev (float): Primary energy of the run [GeV] (lower edge of its energy cell, as in the file name).
            neutrino_bias (float, dict, callable or None, optional): A number (every run), {species: number}, a
                function f(species, e0_gev) -> number, or None: read from the run folder's geometry.json, key
                "neutrino_bias", either a number or {"<E0 as %.6f>": number} (one per run). Defaults to None.
            data_dir (str, optional): Run folder. Defaults to None (the mineral runs of the species).

        Returns:
            float: Bias factor (> 0).

        Raises:
            ValueError: If no bias is found, or it is not positive.
        """
        if callable(neutrino_bias):
            value = neutrino_bias(species, e0_gev)
        elif isinstance(neutrino_bias, dict):
            if species not in neutrino_bias:
                raise ValueError(f"neutrino_bias has no entry for {species} (keys: {sorted(neutrino_bias)}).")
            value = neutrino_bias[species]
        elif neutrino_bias is not None:
            value = neutrino_bias
        else:
            if data_dir is None:
                data_dir = os.path.join(self.data_path, "Geant4_data", self.name, species)
            value = self._geant4_geometry(data_dir)["neutrino_bias"]
            if isinstance(value, dict):
                value = value.get(f"{e0_gev:.6f}")
            if value is None:
                raise ValueError(f"No bias for the {species} run at {e0_gev:g} GeV: pass neutrino_bias, or write "
                                 f"\"neutrino_bias\" in {os.path.join(data_dir, 'geometry.json')}.")
        value = float(value)
        if not value > 0.0:
            raise ValueError(f"Bias of the {species} run at {e0_gev:g} GeV must be > 0, got {value}.")
        return value


    def _flux_cell_integral(self, species, t_kyr_array, edges_gev):
        """Flux of a species integrated over each energy cell, at local times (_energy_cell_nodes quadrature).

        Called by _secondary_cascade (primaries of the StdRock runs) and _neutrino_run_pass (neutrino runs): the run at
        the lower edge of a cell is weighted by the flux integrated over its cell.

        Args:
            species (str): Flux species (MCEQ_SPECIES key).
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            edges_gev (np.ndarray): Cell edges [GeV], shape (M,).

        Returns:
            np.ndarray: Template flux integrated over energy [cm^-2 s^-1]: the flux crossing a horizontal plane
            (see the flux module docstring; for neutrinos see NEUTRINO_SOLID_ANGLE_FACTOR), shape (T, M - 1).
        """
        t_flux = self._flux_time_kyr(np.atleast_1d(np.asarray(t_kyr_array, dtype=float)))
        nodes, weights = _energy_cell_nodes(edges_gev)
        phi = self.flux_history.get_map(species, t_flux, nodes)[2].reshape(t_flux.size, *nodes.shape)
        return np.sum(phi * weights, axis=-1)


    def _read_recoils_and_products(self, filepath, product_edges):
        """Cached form of _read_recoils_and_products_uncached (see there): the result depends only on the file and
        on the product cells, so each file is read once per session (shared cache, see _FILE_CACHE)."""
        stamp = _file_stamp(filepath)
        key = ('recoils_products', stamp, tuple((p, np.asarray(e, dtype=float).tobytes())
                                                for p, e in product_edges.items()), RECOIL_ENERGY_BINS_MEV.tobytes())
        hit, value = _cache_get(key)
        if hit:
            return value
        value = self._read_recoils_and_products_uncached(filepath, product_edges)
        if stamp is not None:
            _cache_put(key, value)
        return value


    def _read_recoils_and_products_uncached(self, filepath, product_edges):
        """Reads one mineral run file: recoils, produced particles by energy cell, and interaction vertices.

        Purpose: a neutrino run gives both the recoils made by the neutrino and the particles it produces (muons,
        neutrons, protons) that are followed separately; a product run (mu-, mu+, neutron, proton in the mineral)
        gives the recoils of that particle and the next generation of neutrons and protons. Both are read here.

        Columns used: 0 name, 2 kinetic energy of the recorded particle [MeV], 3 z of the vertex [mm], 4 primary
        energy before the step [MeV]. A vertex is a distinct (z, primary energy) pair.

        Args:
            filepath (str): outNuclei_<E>.txt file (missing or empty: everything zero).
            product_edges (dict): {product: energy cell edges [GeV]} of the particles to collect.

        Returns:
            dict: 'recoils' {fragment: counts per RECOIL_ENERGY_BINS_MEV bin} (neutrons and non-nuclei left out),
            'products' {product: counts per cell}, 'below' and 'above' {product: counts outside the cells},
            'n_vertices' (int).
        """
        out = {
            'recoils': {},
            'products': {p: np.zeros(len(e) - 1) for p, e in product_edges.items()},
            'below': dict.fromkeys(product_edges, 0), 'above': dict.fromkeys(product_edges, 0),
            'n_vertices': 0,
        }
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return out
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[0, 2, 3, 4],
                             names=['name', 'rec_e', 'z_mm', 'pre_e'],
                             dtype={'name': str, 'rec_e': float, 'z_mm': float, 'pre_e': float})
        except pd.errors.EmptyDataError:
            return out
        raw = df['name'].values.astype(str)
        names = normalize_fragment_names(raw)
        rec = df['rec_e'].values
        for frag in np.unique(names):
            if frag in ('', 'neutron'):
                continue
            out['recoils'][frag] = np.histogram(rec[names == frag], bins=RECOIL_ENERGY_BINS_MEV)[0].astype(float)
        for p, edges in product_edges.items():
            cell = np.searchsorted(edges, rec[raw == p] * 1e-3, side='right') - 1
            inside = (cell >= 0) & (cell < len(edges) - 1)
            out['products'][p] = np.bincount(cell[inside], minlength=len(edges) - 1).astype(float)
            out['below'][p] = int(np.sum(cell < 0))
            out['above'][p] = int(np.sum(cell >= len(edges) - 1))
        out['n_vertices'] = int(np.unique(np.stack([df['z_mm'].values, df['pre_e'].values], axis=1), axis=0).shape[0])
        return out


    def _neutrino_product_operator(self, product_edges, total_simulated_particles=1e4, medium='mineral'):
        """Recoils and next generation of every produced particle, per particle.

        Purpose: the particles produced by the neutrinos (and their own products) are followed with the runs of the
        same particle. For product p in energy cell m (cell lower edge), per primary:
          * R[p][fragment][m]: the recoils it makes over its whole path, from the mineral runs (the recoils counted
            are those in the sample);
          * G[(p, m), (q, n)]: the products of the next generation, from the runs in the medium where the cascade
            develops: the mineral runs (medium 'mineral'), or the StdRock runs (medium 'StdRock': the rock around
            the sample is generic crust; mu+ without StdRock runs uses the mu- runs without the captures, a product
            without StdRock runs makes no further generation).
        The products are indexed on one flat axis: product p occupies offsets[p].

        Args:
            product_edges (dict): {product: energy cell edges [GeV]} (the mineral run grid of each product).
            total_simulated_particles (float, optional): Primaries per product run. Defaults to 1e4.
            medium (str, optional): 'mineral' or 'StdRock'. Defaults to 'mineral'.

        Returns:
            tuple[dict, dict, np.ndarray, dict]: offsets {product: slice}, R {product: {fragment: array (cells,
            recoil bins)}}, G (K, K) with K the total number of cells, and outside {product: [below, above]} (next
            generation particles outside the cells, per primary, summed over the runs).
        """
        offsets, k = _product_axis(product_edges)
        G = np.zeros((k, k))
        R = {p: {} for p in product_edges}
        outside = {p: [0.0, 0.0] for p in product_edges}
        for p, edges in product_edges.items():
            run_dir = os.path.join(self.data_path, "Geant4_data", self.name, p)
            n_cells = len(edges) - 1
            for m in range(n_cells):
                run = self._read_recoils_and_products(os.path.join(run_dir, f"outNuclei_{edges[m]:.6f}.txt"),
                                                      product_edges)
                for frag, counts in run['recoils'].items():
                    R[p].setdefault(frag, np.zeros((n_cells, counts.size)))[m] = counts / total_simulated_particles
                if medium != 'StdRock':
                    for q in product_edges:
                        G[offsets[p].start + m, offsets[q]] = run['products'][q] / total_simulated_particles
                        outside[q][0] += run['below'][q] / total_simulated_particles
                        outside[q][1] += run['above'][q] / total_simulated_particles
        if medium == 'StdRock':
            for p, edges in product_edges.items():
                if self._stdrock_dir(p) is None:
                    self._log(f"Warning: no StdRock/{p} runs: the products of neutrino-made {p}s are not followed.")
                    continue
                for q, q_edges in product_edges.items():
                    G[offsets[p], offsets[q]] = self._stdrock_production_matrix(p, q, edges, q_edges,
                                                                                total_simulated_particles)
        return offsets, R, G, outside


    def _neutrino_run_pass(self, species, run_dir, edges, t_kyr, product_edges, total_simulated_particles,
                           neutrino_bias, rho, solid_angle_factor, with_recoils=True):
        """One pass over the neutrino runs of a folder: recoils and first products, per gram of the run medium.

        Rate per gram:  solid_angle_factor * sum_i Phi_i(t) * counts_i / (N * bias_i * L * rho), with Phi_i the
        flux integrated over the energy cell of run i, L the target length and rho the density of the run medium.

        Args:
            species (str): Neutrino species.
            run_dir (str): Run folder (mineral or StdRock).
            edges (np.ndarray): Energy cell edges of the runs used [GeV].
            t_kyr (np.ndarray): Local times [kyr], shape (T,).
            product_edges (dict): {product: energy cell edges [GeV]}.
            total_simulated_particles (float): Primaries per run.
            neutrino_bias: See _neutrino_bias.
            rho (float): Density of the run medium [g/cm^3].
            solid_angle_factor (float): See integrate_atmospheric_neutrinos.
            with_recoils (bool, optional): Also collect the recoils. Defaults to True.

        Returns:
            dict: 'recoils' {fragment: (T, n_recoil_bins)} [per g per s], 'S0' (T, K) first products [per g per s],
            'lost' {product: [below, above]} (products outside the cells, time-averaged [per g per s]), 'report'
            (n_runs, minimum and maximum bias, maximum biased interactions per primary).
        """
        n_t, n_r = t_kyr.size, len(RECOIL_ENERGY_BINS_MEV) - 1
        offsets, n_k = _product_axis(product_edges)
        length_cm = self._geant4_geometry(run_dir)["target_length_m"] * 100.0
        phi_cell = self._flux_cell_integral(species, t_kyr, edges)
        if not np.any(phi_cell > 0):
            self._log(f"Warning: the flux history has no {species} flux (templates built without neutrinos?): "
                      f"{species} gives zero.")
        out = {'recoils': {}, 'S0': np.zeros((n_t, n_k)), 'lost': {p: [0.0, 0.0] for p in product_edges}}
        biases, fractions = [], []
        for i, e0 in enumerate(edges[:-1]):
            bias = self._neutrino_bias(species, e0, neutrino_bias, data_dir=run_dir)
            run = self._read_recoils_and_products(os.path.join(run_dir, f"outNuclei_{e0:.6f}.txt"), product_edges)
            w = solid_angle_factor * phi_cell[:, i] / (total_simulated_particles * bias * length_cm * rho)
            if with_recoils:
                for frag, counts in run['recoils'].items():
                    out['recoils'].setdefault(frag, np.zeros((n_t, n_r)))
                    out['recoils'][frag] += np.outer(w, counts)
            for p in product_edges:
                out['S0'][:, offsets[p]] += np.outer(w, run['products'][p])
                out['lost'][p][0] += float(np.mean(w)) * run['below'][p]
                out['lost'][p][1] += float(np.mean(w)) * run['above'][p]
            biases.append(bias)
            fractions.append(run['n_vertices'] / total_simulated_particles)
        out['report'] = (len(edges) - 1, min(biases), max(biases), max(fractions))
        return out


    def integrate_atmospheric_neutrinos(
        self, x_bins=None, e_range=None, exposure_window_kyr=None,
        total_simulated_particles=1e4, neutrino_bias=None, species_list=None,
        secondaries=True, product_simulated_particles=None, nucleus=DEFAULT_TRACK_SUM,
        x_grid=TRACK_LENGTH_BINS_NM,
        solid_angle_factor=NEUTRINO_SOLID_ANGLE_FACTOR, flux_history=None,
        secondary_medium='auto', stdrock_neutrino_bias=None, observable=OBSERVABLE_TRACKS,
    ):
        """Track counts from atmospheric neutrinos: their own interactions in the sample and those of the
        particles they produce in the surrounding rock.

        Differences from integrate_all_particles:

        * No overburden. Neutrinos are not attenuated by any realistic overburden, so the depth tables and the
          overburden history play no role; the rate changes in time only through the flux history.
        * All directions. The sample sees neutrinos from the whole sphere, through the Earth too: the rate per unit
          volume is solid_angle_factor times the template flux (see NEUTRINO_SOLID_ANGLE_FACTOR).
        * Biased runs. The interaction probability per primary is uniform along the target (no depletion while the
          biased probability is small), so a run of length L gives the recoils per neutrino per unit path:
              recoils per primary / (bias * L).
          Rate of a fragment per gram:  solid_angle_factor * sum_i Phi_i(t) * counts_i / (N * bias_i * L * rho),
          with Phi_i the flux integrated over the energy cell of run i.

        Secondaries ('nu_secondary_<product>', if secondaries=True): the muons, neutrons and protons the neutrinos
        make in the rock around the sample, and the neutrons and protons those make in turn (all generations,
        cross-generations included: products = S0 (I - G)^-1, see _neutrino_product_operator). The rock around the
        sample is taken to be much thicker than the products' ranges and the neutrino source uniform in it; then
        the products are in equilibrium and the recoil rate per gram of sample is

            (products made per gram of rock, all generations) x (recoils per product over its whole path in the
            mineral),

        with no overburden or geometry (path lengths in g/cm^2 taken equal in rock and mineral). The rock is
        set by secondary_medium:
          * 'StdRock': generic crust, as for the atmospheric secondaries. The first products come from the
            StdRock/<neutrino species> runs (bias read from their geometry.json, or stdrock_neutrino_bias), the
            generations from the StdRock runs of the products; the recoils from the mineral runs of the products.
          * 'mineral': the sample sits in a large body of the same mineral; everything from the mineral runs.
          * 'auto' (default): 'StdRock' if every neutrino species computed has StdRock runs, else 'mineral'.
        This holds for neutrons and protons (ranges below ~1 m.w.e.) and for muons whose range is smaller than the
        rock around the sample; within a muon range of the surface the down-going contribution is overestimated
        (up to 2x), but there the cosmic-ray muons dominate by far. Recoils of products with energies outside the
        product run grid are missing (reported).

        Steps: (1) the neutrino runs in the mineral give the recoils of the neutrinos and, for medium 'mineral', the
        first products S0 (all per g per s); (2) for medium 'StdRock', the StdRock neutrino runs give S0 instead;
        (3) the products of all generations are S0 (I - G)^-1 (_neutrino_product_operator), and their recoils are
        added per product type; (4) every recoil set (per g per s, converted to per kyr) goes through
        _recoils_to_counts.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm]. Defaults to None
                (DEFAULT_INTEGRATION_X_BINS_NM).
            e_range (dict, optional): {neutrino species: (E_min, E_max) [GeV]} restricting the runs used (see
                _energy_edges). Defaults to None (all runs).
            exposure_window_kyr (float or list[float], optional): See integrate_particle_signal_spectrum.
                Defaults to None (whole age).
            total_simulated_particles (float, optional): Primaries per neutrino run. Defaults to 1e4.
            neutrino_bias (float, dict, callable or None, optional): Cross-section bias of the neutrino runs: a
                number, {species: number}, f(species, E0_gev) -> number, or None to read "neutrino_bias" from each
                run folder's geometry.json (see _neutrino_bias). Defaults to None.
            species_list (tuple[str], optional): Neutrino species. Defaults to None (available_neutrino_species()).
            secondaries (bool, optional): Also compute the products' recoils. Defaults to True.
            product_simulated_particles (float, optional): Primaries per product run (mu-, mu+, neutron, proton in
                the mineral). Defaults to None (total_simulated_particles).
            nucleus (str, optional): 'total_no_light_ions', 'total', 'groups' (dict of the track groups, see set_track_groups) or a fragment key. Defaults to
                'total_no_light_ions'.
            x_grid (np.ndarray, optional): Fine track-length grid [nm]. Defaults to TRACK_LENGTH_BINS_NM.
            solid_angle_factor (float, optional): Rate per unit volume / template flux. Defaults to
                NEUTRINO_SOLID_ANGLE_FACTOR (4: isotropic, from all directions).
            flux_history (FluxHistory, optional): If given, set with set_flux_history before the integration;
                None keeps the current one. Defaults to None.
            secondary_medium (str, optional): Rock in which the secondaries are made: 'auto', 'StdRock' or
                'mineral' (see above). Defaults to 'auto'.
            stdrock_neutrino_bias (float, dict, callable or None, optional): Bias of the StdRock neutrino runs, same
                forms as neutrino_bias. Defaults to None (from their geometry.json).
            observable (str, optional): 'tracks' or 'recoil_energy' (x_bins then in MeV, default
                RECOIL_ENERGY_BINS_MEV; see integrate_particle_signal_spectrum). Defaults to 'tracks'.

        Returns:
            tuple[np.ndarray, dict]: x_bins, and tracks (or recoils) per mg in each bin for each neutrino species, each
            'nu_secondary_<product>' (summed over the neutrino species) and 'total'.

        Raises:
            ValueError: If a species has no runs, e_range is invalid, a run has no bias, or secondary_medium is
                'StdRock' without StdRock neutrino runs.
        """
        if flux_history is not None:
            self.set_flux_history(flux_history)
        x_bins, x_grid = _output_axes(observable, x_bins, x_grid)
        species_list = self.available_neutrino_species() if species_list is None else list(species_list)
        bad = [s for s in species_list if s not in NEUTRINO_SPECIES]
        if bad:
            raise ValueError(f"Not neutrino species: {bad} (use {NEUTRINO_SPECIES}).")
        if not species_list:
            raise ValueError(f"No neutrino runs for {self.name} in "
                             f"{os.path.join(self.data_path, 'Geant4_data', self.name)}/<{'|'.join(NEUTRINO_SPECIES)}>.")
        e_range = dict(e_range or {})
        unknown = set(e_range) - set(species_list)
        if unknown:
            raise ValueError(f"e_range given for species not computed: {sorted(unknown)} (computed: {species_list}).")
        n_prod_sim = total_simulated_particles if product_simulated_particles is None else product_simulated_particles

        rho = self.config['density_g_cm3']
        t_kyr = self._window_grid(self._exposure_window(exposure_window_kyr))
        n_t = t_kyr.size

        products = [p for p in NEUTRINO_PRODUCTS if p in self.energy_bins] if secondaries else []
        if secondaries and len(products) < len(NEUTRINO_PRODUCTS):
            self._log(f"Warning: no mineral runs for {sorted(set(NEUTRINO_PRODUCTS) - set(products))}: "
                      "the neutrino products of that type are not followed.")
        product_edges = {p: self.energy_bins[p] for p in products}
        n_cells_total = _product_axis(product_edges)[1]

        stdrock_nu = {sp: os.path.join(self.data_path, "Geant4_data", "StdRock", sp) for sp in species_list}
        has_stdrock = {sp: _geant4_energy_edges(d) is not None for sp, d in stdrock_nu.items()}
        if secondary_medium == 'auto':
            secondary_medium = 'StdRock' if all(has_stdrock.values()) else 'mineral'
            if products and secondary_medium == 'mineral':
                missing = [sp for sp, ok in has_stdrock.items() if not ok]
                self._log(f"Neutrino secondaries made in the mineral (no StdRock runs for {missing}).")
        if secondary_medium not in ('StdRock', 'mineral'):
            raise ValueError(f"secondary_medium must be 'auto', 'StdRock' or 'mineral', got {secondary_medium!r}.")
        if products and secondary_medium == 'StdRock' and not all(has_stdrock.values()):
            raise ValueError(f"No StdRock neutrino runs for {[sp for sp, ok in has_stdrock.items() if not ok]} in "
                             f"{os.path.join(self.data_path, 'Geant4_data', 'StdRock')}.")

        recoils = {}
        S0 = np.zeros((n_t, n_cells_total))
        lost = {p: [0.0, 0.0] for p in products}
        report = []
        for sp in species_list:
            run_dir = os.path.join(self.data_path, "Geant4_data", self.name, sp)
            res = self._neutrino_run_pass(sp, run_dir, self._energy_edges(sp, e_range.get(sp)), t_kyr,
                                          product_edges, total_simulated_particles, neutrino_bias, rho,
                                          solid_angle_factor)
            recoils[sp] = res['recoils']
            report.append((sp, self.name, *res['report']))
            if secondary_medium == 'mineral':
                S0 += res['S0']
                for p in products:
                    lost[p] = [a + b for a, b in zip(lost[p], res['lost'][p])]

        if products and secondary_medium == 'StdRock':
            for sp in species_list:
                run_dir = stdrock_nu[sp]
                geo = self._geant4_geometry(run_dir)
                rho_rock = geo["density_g_cm3"] if geo["density_g_cm3"] is not None else 2.65
                edges = _restrict_edges(_geant4_energy_edges(run_dir)['edges'], e_range.get(sp), f"StdRock/{sp}")
                res = self._neutrino_run_pass(sp, run_dir, edges, t_kyr, product_edges, total_simulated_particles,
                                              stdrock_neutrino_bias, rho_rock, solid_angle_factor,
                                              with_recoils=False)
                S0 += res['S0']
                for p in products:
                    lost[p] = [a + b for a, b in zip(lost[p], res['lost'][p])]
                report.append((sp, 'StdRock', *res['report']))

        if self.verbose > 0:
            for sp, where, n, b_lo, b_hi, f_max in report:
                bias_txt = f"{b_lo:.3g}" if b_lo == b_hi else f"{b_lo:.3g}-{b_hi:.3g}"
                print(f"Neutrinos {sp} ({where} runs): {n} runs, bias {bias_txt}, biased interactions per primary "
                      f"up to {f_max:.3g}"
                      + ("" if f_max <= NEUTRINO_MAX_INTERACTION_FRACTION else
                         f"  (warning: above {NEUTRINO_MAX_INTERACTION_FRACTION}, the beam is depleted along the "
                         "target and the rate is underestimated: lower the bias)"))

        if products:
            offsets, R, G, outside = self._neutrino_product_operator(product_edges, n_prod_sim, secondary_medium)
            radius = np.max(np.abs(np.linalg.eigvals(G))) if G.size else 0.0
            if radius < 1.0:
                S = np.linalg.solve((np.eye(n_cells_total) - G).T, S0.T).T
            else:
                self._log(f"Warning: product multiplication per generation >= 1 (spectral radius {radius:.3g}); "
                          f"generations truncated at {NEUTRINO_MAX_GENERATIONS}.")
                S, term = S0.copy(), S0.copy()
                for _ in range(NEUTRINO_MAX_GENERATIONS):
                    term = term @ G
                    S += term
            for p in products:
                spec = {}
                for frag, r_rows in R[p].items():
                    spec[frag] = S[:, offsets[p]] @ r_rows
                recoils[NEUTRINO_SECONDARY_PREFIX + p] = spec
            if self.verbose > 0:
                for p in products:
                    made = float(np.mean(S0[:, offsets[p]].sum(axis=1)))
                    below, above = lost[p]
                    if made + below + above > 0 and (below + above) > 1e-3 * (made + below + above):
                        print(f"Warning: {100 * (below + above) / (made + below + above):.2g}% of the {p}s made by "
                              f"neutrinos lie outside the {p} runs ({below / (made + below + above):.2g} below "
                              f"{product_edges[p][0]:.4g} GeV): their recoils are missing.")

        results = {}
        for key, spec in recoils.items():
            spectra = _recoil_spectra({frag: arr * SECONDS_PER_KYR for frag, arr in spec.items()}, t_kyr)
            results[key] = self._recoils_to_counts(spectra, observable, x_bins, x_grid, nucleus, key)

        results['total'] = _combine_track_results(results.values())
        return x_bins, results


def _species_worker(self_and_args):
    """Pool worker of integrate_all_particles: computes the track counts of one task.

    Args:
        self_and_args (tuple): (Paleodetector instance, keyword arguments of integrate_particle_signal_spectrum
            with either 'species' (one primary species) or 'secondary_species' (list of secondary species, solved
            together by _integrate_secondary_species, energy_bins_gev then being a dict per species)).

    Returns:
        tuple[list[tuple[str, np.ndarray]], dict]: (species, tracks per mg in each bin) for each species of the task,
        and the file-cache entries read by this worker (merged into the parent's cache).
    """
    self, kwargs = self_and_args
    kwargs = dict(kwargs)
    for key, value in kwargs.pop('_file_cache', {}).items():
        _cache_put(key, value)
    known = set(_FILE_CACHE)
    if 'secondary_species' in kwargs:
        secondary = kwargs.pop('secondary_species')
        self._log(f"Processing secondary species: {secondary}", level=2)
        result = list(self._integrate_secondary_species(secondary, **kwargs).items())
    else:
        species = kwargs.pop('species')
        self._log(f"Processing species: {species}", level=2)
        result = [(species, self.integrate_particle_signal_spectrum(species=species, **kwargs)[1])]
    return result, {key: _FILE_CACHE[key] for key in _FILE_CACHE if key not in known}
