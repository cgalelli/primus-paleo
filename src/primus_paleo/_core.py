"""Internal building blocks of the paleo-detector code: constants, binnings, species tables, fragment naming,
Geant4 file discovery, depth-table maths, annealing laws and small numerical helpers.

Internal: notebooks use these names through primus_paleo, which re-exports the public ones.

Constants are read by the other modules when they are imported, so to change one (e.g. TIME_GRID_RTOL) edit it here.
The one setting meant to change at run time is the default output binning: use set_integration_x_bins_nm, which
updates DEFAULT_INTEGRATION_X_BINS_NM here, where every function reads it at call time (_resolve_x_bins). Read its
current value as primus_paleo.DEFAULT_INTEGRATION_X_BINS_NM; do not copy it with "from ._core import ...".
"""

import os
import re

import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    'PROTON_MASS_MEV', 'NEUTRON_MASS_MEV', 'B0_U238', 'FISSIONS_U238_G_KYR', 'U238_ABUNDANCE', 'KYR_PER_SECOND',
    'NU_BAR_U238_SF', 'WATT_A_MEV', 'WATT_B_PER_MEV', 'TOTAL_ALL_FRAGMENTS', 'TOTAL_NO_LIGHT_IONS',
    'TRACK_SUMS', 'DEFAULT_TRACK_SUM', 'LIGHT_ION_MAX_Z', 'is_light_ion', 'normalize_fragment_name',
    'normalize_fragment_names', 'RECOIL_N_BINS', 'RECOIL_ER_MIN_LOG_MEV', 'RECOIL_ER_MAX_LOG_MEV',
    'RECOIL_ENERGY_BINS_MEV', 'LENGTH_N_BINS', 'LENGTH_MIN_LOG_NM', 'LENGTH_MAX_LOG_NM', 'TRACK_LENGTH_BINS_NM',
    'COMPUTABLE_SPECIES', 'MUON_SPECIES', 'DERIVED_SPECIES', 'SECONDARY_SOURCES', 'IRREGULAR_STEP_FACTOR',
    'NEUTRINO_SPECIES', 'NEUTRINO_PRODUCTS', 'NEUTRINO_SECONDARY_PREFIX', 'NEUTRINO_SOLID_ANGLE_FACTOR',
    'NEUTRINO_MAX_INTERACTION_FRACTION', 'NEUTRINO_MAX_GENERATIONS', 'set_integration_x_bins_nm',
    'STOPPING_N_QUANTILES', 'STOPPING_P_NODES', 'MIN_DEPTH_SAMPLES', 'DEPTH_FLOOR_MWE', 'TRUNCATION_FRACTION',
    'CAPTURE_SPECTRUM_MAX_E0_GEV', 'NEUTRON_TAIL_P', 'NEUTRON_P_NODES', 'SECONDARY_WINDOW_SURVIVAL',
    'SECONDARY_WINDOW_BINS', 'TIME_GRID_BACKBONE', 'TIME_GRID_RTOL', 'TIME_GRID_MAX_POINTS',
    'TIME_GRID_MAX_PASSES', 'TIME_GRID_ATTENUATION_MWE', 'TIME_GRID_FLUX_PROBES', 'K_B_EV', 'SECONDS_PER_KYR',
    'log_interp1d',
]

# --- Physical Constants ---
PROTON_MASS_MEV = 938.3
NEUTRON_MASS_MEV = 939.6
B0_U238 = 7.570126
FISSIONS_U238_G_KYR = 2.14e8
U238_ABUNDANCE = 0.9927
KYR_PER_SECOND = 1/ (60 * 60 * 24 * 365 * 1e3)

# --- U-238 spontaneous fission: shared by the fragment tracks and the SF neutrons ---
NU_BAR_U238_SF = 2.0
WATT_A_MEV = 0.65
WATT_B_PER_MEV = 3.7

_NAMED_FRAGMENTS = {"neutron", "proton", "deuteron", "triton", "alpha"}
_LIGHT_ION_SYMBOLS = {"proton": "H1", "deuteron": "H2", "triton": "H3", "alpha": "He4"}
_ION_NAME_RE = re.compile(r"^([A-Z][a-z]?)(\d+)(?:\[[^\]]*\])?$")

# Track sums. "total" adds every charged fragment; "total_no_light_ions", the default, leaves out hydrogen and helium
# (Z <= LIGHT_ION_MAX_Z: p, d, t, 3He, alpha). Their energy loss is almost all ionisation spread along the path
# (dE/dx at most ~0.1 keV/nm for protons and ~0.4 keV/nm for alphas, at the Bragg peak), which leaves isolated point
# defects rather than the continuous damaged core, so they are not expected to give etchable tracks.
TOTAL_ALL_FRAGMENTS = "total"
TOTAL_NO_LIGHT_IONS = "total_no_light_ions"
TRACK_SUMS = (TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS)
DEFAULT_TRACK_SUM = TOTAL_NO_LIGHT_IONS
LIGHT_ION_MAX_Z = 2
_LIGHT_ION_ELEMENTS = {"H", "He"}


def is_light_ion(fragment):
    """Tells whether a fragment is a hydrogen or helium ion, i.e. a track that is not counted by default.

    Purpose: the default track sum 'total_no_light_ions' (TOTAL_NO_LIGHT_IONS) leaves out H and He fragments
    (p, d, t, 3He, alpha, ...), whose sparse ionisation does not leave an etchable damage core.
    _convert_recoil_to_track_spectrum calls this for every fragment to build that sum. The test is on the element,
    so it works for both the named Geant4 particles ("proton", "alpha") and ion names ("He3", "H2").

    Args:
        fragment (str): Fragment key (see normalize_fragment_name).

    Returns:
        bool: True if the fragment has Z <= LIGHT_ION_MAX_Z.
    """
    symbol = _LIGHT_ION_SYMBOLS.get(fragment, fragment)
    return ''.join(filter(str.isalpha, symbol)) in _LIGHT_ION_ELEMENTS


def normalize_fragment_name(name):
    """Maps a Geant4 particle name to the fragment key used throughout the analysis.

    Purpose: the outNuclei_<E>.txt files list every recorded particle by its Geant4 name. Recoil spectra are
    accumulated per nucleus, so excited states of the same nucleus must be merged ("Ne21[28.465]" and "Ne21" are
    the same track), the named light ions and the neutron kept, and everything else (gammas, pions, antiparticles)
    dropped. The key is also what selects the range table DEDX_Z<z>_A<a>.txt of the nucleus.

    Examples:
        "Ne21[28.465]" -> "Ne21"
        "Ne21" -> "Ne21"
        "proton", "deuteron", "triton", "alpha", "neutron" -> unchanged
        "antilambda", "pi+", "gamma", "anti_He3" -> ""

    Args:
        name (str): Particle name as written in the Geant4 output.

    Returns:
        str: Fragment key, or "" if the particle is not a nucleus or neutron.
    """
    name = str(name)
    if name in _NAMED_FRAGMENTS:
        return name
    m = _ION_NAME_RE.match(name)
    if m:
        return m.group(1) + m.group(2)
    return ""


def normalize_fragment_names(names):
    """Vectorised normalize_fragment_name, for a whole column of a Geant4 output file.

    Purpose: the output files have up to millions of rows but only a few hundred distinct names, so each distinct
    name is mapped once and the result broadcast back; the cost does not grow with the number of rows. Discarded
    particles become "", which never matches a fragment key and is therefore dropped downstream.

    Args:
        names (array_like): Particle names, any shape.

    Returns:
        np.ndarray: Fragment keys (str), same shape as names.
    """
    names = np.asarray(names, dtype=str)
    if names.size == 0:
        return names
    uniq, inverse = np.unique(names, return_inverse=True)
    mapped = np.array([normalize_fragment_name(u) for u in uniq], dtype=str)
    return mapped[inverse.reshape(names.shape)]

# --- Binning setup ---
RECOIL_N_BINS= 301
RECOIL_ER_MIN_LOG_MEV= -3
RECOIL_ER_MAX_LOG_MEV= 4
RECOIL_ENERGY_BINS_MEV = np.logspace(RECOIL_ER_MIN_LOG_MEV, RECOIL_ER_MAX_LOG_MEV, RECOIL_N_BINS)

LENGTH_N_BINS = 1000
LENGTH_MIN_LOG_NM = 1.5
LENGTH_MAX_LOG_NM = 5.5
TRACK_LENGTH_BINS_NM = np.logspace(LENGTH_MIN_LOG_NM, LENGTH_MAX_LOG_NM, LENGTH_N_BINS)

_ENERGY_EDGE_RTOL = 1e-9

COMPUTABLE_SPECIES = ('mu-', 'mu+', 'neutron', 'proton')
MUON_SPECIES = ('mu-', 'mu+')

DERIVED_SPECIES = {'secondary_neutron': 'neutron', 'secondary_proton': 'proton'}
SECONDARY_SOURCES = ('mu-', 'mu+', 'neutron', 'proton')
IRREGULAR_STEP_FACTOR = 1.5   # a step this many times the median step in log(E) is reported as a gap

NEUTRINO_SPECIES = ('nu_mu', 'anti_nu_mu', 'nu_e', 'anti_nu_e')
NEUTRINO_PRODUCTS = ('mu-', 'mu+', 'neutron', 'proton')
NEUTRINO_SECONDARY_PREFIX = 'nu_secondary_'
NEUTRINO_SOLID_ANGLE_FACTOR = 4.0
NEUTRINO_MAX_INTERACTION_FRACTION = 0.1
NEUTRINO_MAX_GENERATIONS = 50             # cap on the product generations if the series does not converge

DEFAULT_INTEGRATION_X_BINS_NM = np.linspace(0, 50000, 500)

def set_integration_x_bins_nm(integration_x_bins_nm):
    """Sets the module-wide default output track-length binning.

    Purpose: every integrate_* function returns counts on x_bins; when x_bins is None they use this default, read
    at call time (_resolve_x_bins). Changing it once (e.g. to the bins of a microscopy descriptor) avoids passing
    x_bins to every call. It does not change the fine internal grid TRACK_LENGTH_BINS_NM.

    Args:
        integration_x_bins_nm (np.ndarray): Bin edges [nm].
    """
    global DEFAULT_INTEGRATION_X_BINS_NM

    DEFAULT_INTEGRATION_X_BINS_NM = integration_x_bins_nm
    return


def _geant4_energy_edges(data_dir):
    """Primary-energy bins of a Geant4 run directory, built from its file names.

    Purpose: the code integrates the cosmic-ray flux over energy bins and weights each bin by the response of one
    Geant4 run. Each outNuclei_<E>.txt is taken as the response at the lower edge E of its bin; a bin ends at the
    next file energy and the last one at one more step of the same logarithmic size as the last step (the
    convention of the run grids). Built per species at creation (_scan_geant4_species), so every species uses all
    and only the energies actually simulated. Irregular steps are reported because the run below a gap then
    stands for the whole gap.

    Args:
        data_dir (str): Directory containing outNuclei_<E>.txt files.

    Returns:
        dict or None: 'edges' (bin edges [GeV], n_files + 1 values), 'n_files', 'gaps' (list of (E_low, E_high)
        steps larger than IRREGULAR_STEP_FACTOR times the median step in log(E), so one missing file on a
        logarithmic grid, a doubled step, is reported); None if fewer than 2 files.
    """
    energies = np.array([e for e, _ in _list_geant4_energy_files(data_dir)])
    if energies.size < 2:
        return None
    ratios = energies[1:] / energies[:-1]
    edges = np.append(energies, energies[-1] * ratios[-1])
    log_steps = np.log(ratios)
    big = log_steps > IRREGULAR_STEP_FACTOR * np.median(log_steps)
    gaps = [(energies[i], energies[i + 1]) for i in np.where(big)[0]]
    return {'edges': edges, 'n_files': energies.size, 'gaps': gaps}


def _restrict_edges(edges, e_range, label):
    """Keeps the energy bins lying entirely inside a range.

    Purpose: e_range options (integrate_all_particles, integrate_atmospheric_neutrinos) select a part of a Geant4
    run grid; this is the single place where that selection is made, for the mineral and the StdRock grids alike.

    Args:
        edges (np.ndarray): Bin edges [GeV].
        e_range (tuple[float, float] or None): (E_min, E_max) [GeV]; None keeps every bin.
        label (str): Name used in the error messages.

    Returns:
        np.ndarray: The kept bin edges [GeV].

    Raises:
        ValueError: If e_range is not increasing, is outside the grid, or contains no complete bin.
    """
    if e_range is None:
        return edges
    lo, hi = (float(v) for v in e_range)
    tol = 1 + _ENERGY_EDGE_RTOL
    if not lo < hi:
        raise ValueError(f"{label}: e_range {e_range} must be (E_min, E_max) with E_min < E_max.")
    if lo * tol < edges[0] or hi > edges[-1] * tol:
        raise ValueError(f"{label}: e_range ({lo:g}, {hi:g}) GeV is outside the available range "
                         f"({edges[0]:.4g}, {edges[-1]:.4g}) GeV.")
    inside = np.where((edges[:-1] * tol >= lo) & (edges[1:] <= hi * tol))[0]
    if inside.size == 0:
        raise ValueError(f"{label}: e_range ({lo:g}, {hi:g}) GeV contains no complete energy bin.")
    return edges[inside[0]:inside[-1] + 2]


def _resolve_x_bins(x_bins):
    """The given output track-length bins, or the module default (set_integration_x_bins_nm) if None.

    Purpose: one place for the x_bins default, read at call time so that set_integration_x_bins_nm takes effect
    without reloading the module.

    Args:
        x_bins (array_like or None): Bin edges [nm].

    Returns:
        np.ndarray: Bin edges [nm].
    """
    return DEFAULT_INTEGRATION_X_BINS_NM if x_bins is None else np.asarray(x_bins, dtype=float)


# --- Depth-probability tables from the StdRock Geant4 runs ---
STOPPING_N_QUANTILES = 100
STOPPING_P_NODES = np.unique(np.concatenate([
    np.linspace(0.0, 1.0, STOPPING_N_QUANTILES + 1),
    [0.001, 0.002, 0.005, 0.01, 0.99, 0.995, 0.998, 0.999],
]))
MIN_DEPTH_SAMPLES = 20
DEPTH_FLOOR_MWE = 1e-6
TRUNCATION_FRACTION = 0.98
CAPTURE_SPECTRUM_MAX_E0_GEV = 1.0

NEUTRON_TAIL_P = 0.95
NEUTRON_P_NODES = STOPPING_P_NODES[STOPPING_P_NODES <= NEUTRON_TAIL_P]

SECONDARY_WINDOW_SURVIVAL = 1e-3
SECONDARY_WINDOW_BINS = 50

# Master time grid of the detector (_master_time_grid), shared by every time integral and by annealing. It starts
# from a uniform backbone plus the known breakpoints of the overburden and flux histories, and is refined by
# bisection wherever a probe function is not integrated to TIME_GRID_RTOL (relative to its integral between
# consecutive breakpoints) by the trapezoid rule. Probes:
# the overburden itself, its attenuation exp(-X / L) for the lengths below (atmospheric neutrons, shallow and deep
# muons) and the flux history at a few species and energies.
TIME_GRID_BACKBONE = 100
TIME_GRID_RTOL = 1e-5
TIME_GRID_MAX_POINTS = 5000
TIME_GRID_MAX_PASSES = 20
TIME_GRID_ATTENUATION_MWE = (1.6, 30.0, 300.0)
TIME_GRID_FLUX_PROBES = (('mu-', 1.0), ('mu-', 100.0), ('neutron', 0.1))

_G4_OUTPUT_FILE_RE = re.compile(r"^outNuclei_(\d+\.\d+)\.txt$")


def _list_geant4_energy_files(data_dir):
    """Lists the Geant4 output files of a run directory, with the primary energy read from each file name.

    Purpose: every Geant4 run directory (Data/Geant4_data/<mineral>/<species> and Data/Geant4_data/StdRock/<species>)
    holds one outNuclei_<E>.txt per simulated primary energy E [GeV]. This is the single place where those files
    are found; it is used to build the energy grids (_geant4_energy_edges), to tell whether a species has runs at
    all, and to pick the file of a given energy (_load_stdrock_rows).

    Args:
        data_dir (str): Directory containing outNuclei_<E>.txt files.

    Returns:
        list[tuple[float, str]]: (primary energy [GeV], file path), sorted by energy. Empty if the directory does
        not exist or has no such files.
    """
    if not os.path.isdir(data_dir):
        return []
    found = []
    for fn in os.listdir(data_dir):
        m = _G4_OUTPUT_FILE_RE.match(fn)
        if m:
            found.append((float(m.group(1)), os.path.join(data_dir, fn)))
    return sorted(found)


def _loglog_interp_rows(x, xp, fp_rows):
    """Interpolates a table of rows, one row per tabulated energy, linearly in log-log space.

    Purpose: the StdRock depth tables are built only at the simulated energies (one row of depth quantiles, or one
    exponential-tail length, per energy), but the flux is integrated on fine energy grids in between. Depth scales
    are close to power laws of the energy, so interpolating log(value) against log(energy) is accurate between
    runs and gives a sensible extrapolation beyond the first and last one. Used by _depth_quantiles and
    _hadron_tail_lambdas. Each column (e.g. each quantile) is interpolated independently.

    Args:
        x (array_like): Energies to evaluate, shape (n,). Must be > 0.
        xp (array_like): Tabulated energies, increasing, shape (m,). Must be > 0.
        fp_rows (array_like): Tabulated values, shape (m, k) or (m,). Must be > 0.

    Returns:
        np.ndarray: Interpolated values, shape (n, k); extrapolated linearly in log-log from the end segments.
    """
    lx = np.log(np.atleast_1d(np.asarray(x, dtype=float)))
    lxp = np.log(np.asarray(xp, dtype=float))
    lf = np.log(np.asarray(fp_rows, dtype=float))
    if lf.ndim == 1:
        lf = lf[:, None]
    if len(lxp) == 1:
        return np.exp(np.repeat(lf, len(lx), axis=0))
    i = np.clip(np.searchsorted(lxp, lx) - 1, 0, len(lxp) - 2)
    w = (lx - lxp[i]) / (lxp[i + 1] - lxp[i])
    return np.exp(lf[i] + w[:, None] * (lf[i + 1] - lf[i]))


def _piecewise_cdf_and_integral(z, q, p, tail_lambda=None):
    """Cumulative distribution F(z) of a depth, and its integral G(z), from a quantile table.

    Purpose: this is how a StdRock depth distribution is evaluated. For each simulated energy the code stores the
    depths R at which a particle stops (muons) or interacts (hadrons) as quantile nodes q at probabilities p
    (_build_depth_table); this function turns those nodes back into a distribution:
      * F(z) = P(R <= z), linear between the nodes. F(z_max) - F(z_min) is the probability that the stop or the
        interaction falls inside a slab [z_min, z_max]: the capture probability of a mu- in the sample, or the
        interaction probability of a hadron (_slab_probability).
      * G(z) = integral_0^z (1 - F) = E[min(R, z)], the mean path length travelled within the first z m.w.e.
        For muons, which interact all along their path until they stop, the expected path inside a slab is
        G(z_max) - G(z_min), and dividing by G(inf) = E[R] gives the fraction of their interactions that happen
        there (_muon_depth_probabilities). For secondary particles born at different distances above the sample,
        H(z) = z - G(z) gives the probability averaged over a production bin (_process_secondary_geant4_data).
    If p ends below 1 (hadrons: the last nodes are poorly sampled), an exponential tail is attached beyond the last
    node, F(z) = 1 - (1 - p[-1]) exp(-(z - q[-1]) / tail_lambda), so that deep slabs still get a probability.

    Args:
        z (array_like): Depths [m.w.e.], any shape.
        q (np.ndarray): Quantile nodes [m.w.e.], non-decreasing, shape (n,).
        p (np.ndarray): Probabilities of the nodes, shape (n,), p[0] = 0.
        tail_lambda (float, optional): Length of the exponential tail [m.w.e.]. Required when p[-1] < 1.
            Defaults to None (no tail: F = 1 beyond the last node).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            F: P(R <= z), same shape as z.
            G: integral from 0 to z of (1 - F(t)) dt = E[min(R, z)], same shape as z. G(inf) is the mean E[R].
    """
    z = np.asarray(z, dtype=float)
    dq = np.diff(q)
    dp = np.diff(p)
    slope_all = np.divide(dp, dq, out=np.zeros_like(dp), where=dq > 0)

    g_nodes = np.empty_like(q)
    g_nodes[0] = q[0]
    g_nodes[1:] = q[0] + np.cumsum(dq * (1.0 - 0.5 * (p[:-1] + p[1:])))

    k = np.clip(np.searchsorted(q, z, side='right') - 1, 0, len(q) - 2)
    t = np.clip(z, q[0], q[-1]) - q[k]
    slope = slope_all[k]

    if tail_lambda is None:
        F_beyond = 1.0
        G_beyond = g_nodes[-1]
    else:
        excess = np.maximum(z - q[-1], 0.0)
        survive = 1.0 - p[-1]
        decay = np.exp(-excess / tail_lambda)
        F_beyond = 1.0 - survive * decay
        G_beyond = g_nodes[-1] + survive * tail_lambda * (1.0 - decay)

    F = np.where(z < q[0], 0.0, np.where(z >= q[-1], F_beyond, p[k] + slope * t))
    G_inside = g_nodes[k] + t * (1.0 - p[k]) - 0.5 * slope * t**2
    G = np.where(z < q[0], z, np.where(z >= q[-1], G_beyond, G_inside))
    return F, G


def _slab_probability(q_rows, p, z_min_grid, z_max_grid, tail_lambdas=None):
    """Probability that the stop or interaction depth falls inside a slab, for every energy of a grid at once.

    Purpose: the sample is a slab [overburden, overburden + thickness] along each slanted path (both divided by
    cos(zenith)). For a hadron entering the ground, the chance of interacting in the sample is
    F(z_max) - F(z_min) of its StdRock depth distribution at that energy (_piecewise_cdf_and_integral). This
    evaluates it over the whole (time, zenith, energy) grid of _process_geant4_data, row by row of the depth table.

    Args:
        q_rows (np.ndarray): Quantile nodes per energy, shape (n_E, n_nodes).
        p (np.ndarray): Probabilities of the nodes, shape (n_nodes,).
        z_min_grid (np.ndarray): Slab top along the path [m.w.e.], shape (..., n_E); the last axis runs over the
            energies of q_rows.
        z_max_grid (np.ndarray): Slab bottom along the path [m.w.e.], same shape as z_min_grid.
        tail_lambdas (np.ndarray, optional): Exponential-tail length per energy [m.w.e.], shape (n_E,).
            Defaults to None (no tail).

    Returns:
        np.ndarray: P(z_min < R <= z_max), same shape as z_min_grid.
    """
    prob = np.empty_like(z_min_grid, dtype=float)
    for j in range(q_rows.shape[0]):
        lam = None if tail_lambdas is None else tail_lambdas[j]
        F_lo, _ = _piecewise_cdf_and_integral(z_min_grid[..., j], q_rows[j], p, lam)
        F_hi, _ = _piecewise_cdf_and_integral(z_max_grid[..., j], q_rows[j], p, lam)
        prob[..., j] = F_hi - F_lo
    return prob


def _rebin_density(density, grid_edges, out_edges):
    """Counts in coarse output bins from a density defined on a fine grid.

    Purpose: track-length spectra are computed on the fine logarithmic grid TRACK_LENGTH_BINS_NM (so that short
    tracks are resolved) as densities per nm, but returned on the user's x_bins (e.g. 200 linear bins). Taking the
    density constant inside each fine bin, the cumulative count is exact at the fine edges and linear in between,
    so the total is conserved and no numerical quadrature is needed. Used by the fission and radiogenic-neutron
    integrals; counts outside the fine grid are zero.

    Args:
        density (np.ndarray): Density in each fine bin [per unit length], len(grid_edges) - 1 values.
        grid_edges (np.ndarray): Fine-grid edges.
        out_edges (np.ndarray): Output bin edges, same units.

    Returns:
        np.ndarray: Counts in each output bin.
    """
    grid_edges = np.asarray(grid_edges, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(density, dtype=float) * np.diff(grid_edges))))
    return np.diff(np.interp(np.asarray(out_edges, dtype=float), grid_edges, cumulative))

# --- Track annealing laws -------------------------------------------------------------------------------------
K_B_EV = 8.617333262e-5
SECONDS_PER_KYR = 1e3 * 365.25 * 86400.0
_ANNEALING_GEOTHERM_DEFAULTS = {'T_surface_C': 15.0, 'gradient_C_per_km': 25.0, 'rock_density_g_cm3': 2.16}
_R_FRESH = 1.0 - 1e-4            # r above this counts as a fresh (unannealed) track


def _annealing_law(params):
    """Isothermal annealing law r_iso(t, T) and its inverse t_iso(r, T), with t in kyr and T in K.

    Purpose: laboratory annealing laws give the reduced length r of a track held at a constant temperature T for a
    time t. The sample's temperature changes with its burial, so _annealing_from_law follows each track step by
    step with the equivalent-time method, which needs both directions: r_iso to advance a step, and t_iso to turn
    the current r into the time the law would need to reach it at the new temperature. Both are analytic for the
    laws below.

    Laws ('law' key of params):
      'fanning_curvilinear' (Ketcham et al. 1999, 2007, fission tracks in apatite):
          g(r) = C0 + C1 * (ln t - C2) / (ln(1/T) - C3),   g(r) = (((1 - r^beta) / beta)^alpha - 1) / alpha
          (g(r) = ln((1 - r^beta) / beta) for alpha = 0). Keys: C0, C1, C2, C3, alpha, beta and time_unit_s, the
          seconds per unit of t in the published fit (e.g. 1 for seconds, 3600 for hours). No defaults: take the
          constants and the time unit from the paper.
      'first_order': r = exp(-k(T) t), k(T) = A exp(-Ea / (k_B T)). Keys: A_per_kyr, Ea_eV.

    Args:
        params (dict): Law name and constants.

    Returns:
        tuple[callable, callable]: (r_iso, t_iso), vectorised.

    Raises:
        ValueError: If the law is unknown or a constant is missing.
    """
    law = params.get('law')
    need = {'fanning_curvilinear': ('C0', 'C1', 'C2', 'C3', 'alpha', 'beta', 'time_unit_s'),
            'first_order': ('A_per_kyr', 'Ea_eV')}
    if law not in need:
        raise ValueError(f"annealing 'law' must be one of {list(need)}, got {law!r}.")
    missing = [k for k in need[law] if k not in params]
    if missing:
        raise ValueError(f"annealing law '{law}' needs {missing}.")

    if law == 'first_order':
        A, Ea = float(params['A_per_kyr']), float(params['Ea_eV'])
        rate = lambda T: A * np.exp(-Ea / (K_B_EV * T))
        r_iso = lambda t, T: np.exp(-rate(T) * t)
        t_iso = lambda r, T: -np.log(np.clip(r, 1e-300, 1.0)) / rate(T)
        return r_iso, t_iso

    C0, C1, C2, C3 = (float(params[k]) for k in ('C0', 'C1', 'C2', 'C3'))
    alpha, beta = float(params['alpha']), float(params['beta'])
    kyr_to_unit = SECONDS_PER_KYR / float(params['time_unit_s'])

    def g_of_r(r):
        u = (1.0 - np.clip(r, 0.0, _R_FRESH) ** beta) / beta
        return np.log(u) if alpha == 0 else (u ** alpha - 1.0) / alpha

    def r_of_g(g):
        u = np.exp(g) if alpha == 0 else np.power(np.clip(alpha * g + 1.0, 0.0, None), 1.0 / alpha)
        base = 1.0 - beta * u
        return np.where(base > 0, np.power(np.clip(base, 1e-300, None), 1.0 / beta), 0.0).clip(0.0, 1.0)

    def r_iso(t, T):
        ln_t = np.log(np.clip(np.asarray(t, dtype=float) * kyr_to_unit, 1e-300, None))
        return r_of_g(C0 + C1 * (ln_t - C2) / (np.log(1.0 / T) - C3))

    def t_iso(r, T):
        return np.exp(C2 + (g_of_r(r) - C0) * (np.log(1.0 / T) - C3) / C1) / kyr_to_unit

    return r_iso, t_iso


def _shorten_tracks(density, grid_edges, r, r_min=0.0):
    """Shortens every track of a length spectrum by a factor r: the core of the annealing option.

    Purpose: annealing makes a latent track formed at time t shorter by today (reduced length r(t), see
    _annealing_ratio). The spectra are computed time by time, so each time slice (row) is shortened by its own r
    before the time integral. With proportional shortening a track of length x0 becomes r x0, so the cumulative
    count at length x after shortening is the original cumulative count at x / r; this is exact for a density
    constant inside each bin and conserves the number of tracks. Rows with r < r_min (segmented tracks, no longer
    etchable) are removed.

    Args:
        density (np.ndarray): Densities per bin of grid_edges [per unit length], shape (T, n_x) or (n_x,).
        grid_edges (np.ndarray): Bin edges, n_x + 1 values.
        r (array_like): Shortening factor per row, shape (T,) (or scalar for a 1-D density), 0 < r <= 1.
        r_min (float, optional): Rows with r below this are set to 0. Defaults to 0.

    Returns:
        np.ndarray: Shortened densities, same shape as density.
    """
    density = np.asarray(density, dtype=float)
    grid_edges = np.asarray(grid_edges, dtype=float)
    rows = np.atleast_2d(density)
    r = np.broadcast_to(np.asarray(r, dtype=float), (rows.shape[0],))
    if np.any(r <= 0) or np.any(r > 1 + 1e-12):
        raise ValueError("annealing must give 0 < r <= 1.")
    widths = np.diff(grid_edges)
    cum = np.concatenate([np.zeros((rows.shape[0], 1)), np.cumsum(rows * widths, axis=1)], axis=1)
    out = np.zeros_like(rows)
    for k in range(rows.shape[0]):
        if r[k] < r_min:
            continue
        if r[k] >= 1.0:
            out[k] = rows[k]
            continue
        out[k] = np.diff(np.interp(grid_edges / r[k], grid_edges, cum[k])) / widths
    return out.reshape(density.shape)

def log_interp1d(xx, yy, kind='linear'):
    """Builds an interpolator that is linear in log-log space.

    Purpose: utility for quantities close to power laws (spectra, stopping powers, flux shapes) given at a few
    points, e.g. in notebooks; interpolating log(y) against log(x) keeps the power-law shape between points and
    extrapolates it beyond them. Not used by the Paleodetector class itself.

    Args:
        xx (np.ndarray): x-coordinates of the data points (> 0).
        yy (np.ndarray): y-coordinates of the data points (> 0).
        kind (str, optional): Interpolation kind passed to scipy's interp1d. Defaults to 'linear'.

    Returns:
        callable: f(x) = 10**interp(log10(x)), extrapolating beyond the data.
    """
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = interp1d(logx, logy, kind=kind, fill_value='extrapolate')
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp
