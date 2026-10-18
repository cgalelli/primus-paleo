import numpy as np
import pandas as pd
import os
import re
import json
from scipy.interpolate import interp1d
from mendeleev import element
from tqdm import tqdm
from multiprocessing import Pool

from flux_history import FluxHistory, MCEQ_SPECIES

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
# defects rather than the continuous damaged core that etches faster than the bulk (V > 1), so in crystals such as
# halite they are not expected to give etchable tracks. Heavier recoils (Z >= 3) are kept.
TOTAL_ALL_FRAGMENTS = "total"
TOTAL_NO_LIGHT_IONS = "total_no_light_ions"
TRACK_SUMS = (TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS)
DEFAULT_TRACK_SUM = TOTAL_NO_LIGHT_IONS
LIGHT_ION_MAX_Z = 2
_LIGHT_ION_ELEMENTS = {"H", "He"}   # elements with Z <= LIGHT_ION_MAX_Z


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

# Species whose recoil calculation is implemented. Any MCEQ_SPECIES folder found in Data/Geant4_data/<mineral> is
# listed at creation; only these (and the secondary species below) can be computed for now. Muons use the stopping
# depths of StdRock/mu-; the hadrons (neutron, proton) the interaction depths of StdRock/<species>.
COMPUTABLE_SPECIES = ('mu-', 'mu+', 'neutron', 'proton')
MUON_SPECIES = ('mu-', 'mu+')
# Secondary species: particles of type DERIVED_SPECIES[s] produced in the window above the sample by the primaries in
# SECONDARY_SOURCES (StdRock runs), plus all their later generations, cross-generations included (neutrons from
# protons and protons from neutrons): the products are followed as one cascade (_secondary_yield). Their recoils (and
# energy grid) come from the mineral runs of each product. Same generation policy as the neutrino products
# (_neutrino_product_operator); muons are not produced by the hadron runs, so the chain is
# (mu, n, p) -> secondary n, p -> tertiary n, p from both -> ...
DERIVED_SPECIES = {'secondary_neutron': 'neutron', 'secondary_proton': 'proton'}
SECONDARY_SOURCES = ('mu-', 'mu+', 'neutron', 'proton')
IRREGULAR_STEP_FACTOR = 1.5   # a step this many times the median step in log(E) is reported as a gap

# Atmospheric neutrinos (integrate_atmospheric_neutrinos). Folders Data/Geant4_data/<mineral>/<species>, named as the
# MCEQ_SPECIES keys (Geant4 gun names: nu_mu, anti_nu_mu, nu_e, anti_nu_e), with biased cross-sections.
NEUTRINO_SPECIES = ('nu_mu', 'nu_mu_bar', 'nu_e', 'nu_e_bar')
# Particles produced by the neutrinos (written by the neutrino runs) that are followed to their own recoils, with the
# mineral runs of the same particle; the neutrons and protons they produce in turn are followed too (generations).
NEUTRINO_PRODUCTS = ('mu-', 'mu+', 'neutron', 'proton')
NEUTRINO_SECONDARY_PREFIX = 'nu_secondary_'
# The flux templates give the flux crossing a horizontal plane from above, Phi = integral I cos(theta) dOmega over the
# upper hemisphere (= pi I for an isotropic intensity I). Neutrinos reach a buried sample from every direction, through
# the Earth too, and the interaction rate per unit volume is integral I dOmega over 4 pi = 4 pi I = 4 Phi. Exact for an
# isotropic, up-down symmetric flux; it ignores the oscillation of up-going nu_mu (a few tens of % below ~10 GeV) and
# the horizontal enhancement of the flux above a few GeV.
NEUTRINO_SOLID_ANGLE_FACTOR = 4.0
NEUTRINO_MAX_INTERACTION_FRACTION = 0.1   # biased interactions per primary above this are reported (linear regime)
NEUTRINO_MAX_GENERATIONS = 50             # cap on the product generations if the series does not converge

DEFAULT_INTEGRATION_X_BINS_NM = np.linspace(0, 50000, 200)


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

def etch_pit_axes(length_nm, dip_rad, top_depth_nm, removed_nm, v_ratio, lateral_ratio=1.0, n_s=128):
    """Surface opening of etched latent tracks (Huygens construction, constant etch rates).
 
    Every point of a straight latent track starts dissolving when the etchant reaches it (along the track at v_T,
    from its top point) and from then on grows at the bulk rate v_B: vertically at v_B, laterally at
    lateral_ratio * v_B. The pit is the union of these growing ellipses plus the removed bulk layer; its opening is
    the section of that union by the final surface. Covers in one construction the critical angle
    (sin(dip) < 1/V gives no opening), the conical phase, over-etching (the etch front has reached the track end and
    the tip rounds off) and pits that fade out because the bulk has etched through them.
 
    Args:
        length_nm (np.ndarray): Latent track length [nm].
        dip_rad (np.ndarray): Angle between track and surface plane (pi/2 = perpendicular).
        top_depth_nm (np.ndarray): Depth of the upper end of the track below the ORIGINAL surface [nm]; negative if
            the track crosses the original surface (the part above it does not exist).
        removed_nm (float): Bulk layer removed by etching, h = v_B * t [nm].
        v_ratio (float): V = v_T / v_B (> 1).
        lateral_ratio (float, optional): Lateral / vertical bulk etch rate. 1 = isotropic wet etching (HF);
            0 = fully directional etching (pit walls do not widen). Defaults to 1.
        n_s (int, optional): Points along the track used for the construction. Defaults to 128.
 
    Returns:
        tuple: (major axis, minor axis) of the opening [nm]; NaN where the track gives no opening.
    """
    L, dip, z_top = (np.asarray(a, float) for a in (length_nm, dip_rad, top_depth_nm))
    sin_d, cos_d = np.sin(dip), np.cos(dip)
    z0 = np.maximum(z_top, 0.)                                # where etching of the track starts
    R = (z_top + L * sin_d - z0) / np.maximum(sin_d, 1e-12)  # latent length below that point
    hp = removed_nm - z0                                      # bulk removal left after the track was reached
    ok = (hp > 0) & (R > 0)
 
    major, minor = np.full(L.shape, np.nan), np.full(L.shape, np.nan)
    idx = np.where(ok)[0]
    for k in range(0, len(idx), 20000):                       # chunks keep memory bounded
        i = idx[k:k + 20000]
        s_max = np.minimum(R[i], v_ratio * hp[i])             # part of the track the etchant has travelled
        s = s_max[:, None] * np.linspace(0., 1., n_s)[None, :]
        r = hp[i, None] - s / v_ratio                         # vertical growth since that point was reached
        dz = hp[i, None] - s * sin_d[i, None]                 # distance from the point to the final surface
        a2 = r ** 2 - dz ** 2
        a = lateral_ratio * np.sqrt(np.clip(a2, 0., None))
        x = s * cos_d[i, None]
        hit = a2 > 0
        any_hit = hit.any(axis=1)
        lo = np.where(hit, x - a, np.inf).min(axis=1)
        hi = np.where(hit, x + a, -np.inf).max(axis=1)
        major[i[any_hit]] = (hi - lo)[any_hit]
        minor[i[any_hit]] = 2. * np.where(hit, a, 0.).max(axis=1)[any_hit]
    return major, minor
 
 
def rebin_counts(counts, old_edges, new_edges):
    """Redistributes counts from one binning to another, assuming a uniform density inside each old bin.
 
    Each new bin receives, from every old bin, the fraction of that old bin it overlaps. The total is conserved
    where the two binnings cover the same range; counts outside `new_edges` are dropped.
 
    Args:
        counts (np.ndarray): Counts in the bins of `old_edges` (len(old_edges) - 1 values).
        old_edges, new_edges (np.ndarray): Increasing bin edges, same units (e.g. nm).
 
    Returns:
        np.ndarray: Counts in the bins of `new_edges`.
    """
    counts = np.asarray(counts, float)
    old_edges, new_edges = np.asarray(old_edges, float), np.asarray(new_edges, float)
    if len(counts) != len(old_edges) - 1:
        raise ValueError(f"{len(counts)} counts for {len(old_edges) - 1} bins")
    lo = np.maximum(old_edges[:-1][None, :], new_edges[:-1][:, None])
    hi = np.minimum(old_edges[1:][None, :], new_edges[1:][:, None])
    overlap = np.clip(hi - lo, 0., None) / np.diff(old_edges)[None, :]
    return overlap @ counts
 
 
ETCH_DEFAULTS = {
    'HF': {'removed_nm': 4000., 'v_ratio': 2.5, 'lateral_ratio': 1.0},      # wet, isotropic (obsidian)
    'plasma': {'removed_nm': 100., 'v_ratio': 10.0, 'lateral_ratio': 0.1},  # dry, directional (halite)
}
 
 
def slice_spectrum_etch(counts_mg, sample_area_cm2, sample_density_g_cm3, removed_nm=None, v_ratio=None,
                        lateral_ratio=None, angular_pdf=None, min_width_nm=0.345 * 3 * 1e3, measure='major',
                        rng=None, x_bins=None, out_bins=None, etch_type="HF"):
    """Monte Carlo of the etched-pit size distribution on the final surface, from a latent track-length spectrum.
 
    Same population and normalisation as slice_spectrum (tracks per mg -> tracks on sample_area_cm2), with the
    measured size given by the pit opening of `etch_pit_axes` instead of the projected track length.
 
    The latent spectrum and the measured sizes can use different binnings: `x_bins` describes `counts_mg` (e.g. the
    wide bins of the theoretical spectrum) and `out_bins` the returned histogram (e.g. the bins of the descriptor).
    Latent lengths are drawn uniformly inside their bin, so wide input bins do not produce a comb of discrete
    lengths, and the measured sizes are histogrammed directly on `out_bins` (no approximation from rebinning).
 
    Args:
        counts_mg (np.ndarray): Latent track spectrum, tracks per mg in each bin of x_bins.
        sample_area_cm2, sample_density_g_cm3 (float): Analysed area [cm^2] and target density [g/cm^3].
        removed_nm (float, optional): Bulk layer removed, h = v_B * t [nm]. Default from `etch_type`.
        v_ratio (float, optional): V = v_T / v_B. Default from `etch_type`.
        lateral_ratio (float, optional): See etch_pit_axes (1 isotropic, 0 directional). Default from `etch_type`.
        angular_pdf (np.ndarray, optional): As in slice_spectrum. Defaults to isotropic.
        min_width_nm (float, optional): Pits with minor axis below this width are not resolved and are dropped
            (optical resolution). Defaults to 1125 nm.
        measure (str, optional): 'major' (what OptimusPrimus reports as length) or 'minor'.
        rng (np.random.Generator, optional): Random generator.
        x_bins (np.ndarray, optional): Bin edges of `counts_mg` [nm]. Defaults to DEFAULT_INTEGRATION_X_BINS_NM.
        out_bins (np.ndarray, optional): Bin edges of the returned histogram [nm]. Defaults to x_bins.
        etch_type (str, optional): 'HF' or 'plasma': sets the defaults of removed_nm, v_ratio and lateral_ratio
            (ETCH_DEFAULTS); explicitly given values always win, including 0. Defaults to "HF".
 
    Returns:
        np.ndarray: Expected number of pits per bin of out_bins on the analysed area.
 
    Raises:
        ValueError: If etch_type or measure is unknown, or counts_mg does not match x_bins.
    """
    if etch_type not in ETCH_DEFAULTS:
        raise ValueError(f"etch_type must be one of {list(ETCH_DEFAULTS)}, got '{etch_type}'")
    if measure not in ('major', 'minor'):
        raise ValueError("measure must be 'major' or 'minor'")
    d = ETCH_DEFAULTS[etch_type]
    removed_nm = d['removed_nm'] if removed_nm is None else removed_nm
    v_ratio = d['v_ratio'] if v_ratio is None else v_ratio
    lateral_ratio = d['lateral_ratio'] if lateral_ratio is None else lateral_ratio
 
    rng = rng or np.random.default_rng()
    x_bins = np.asarray(DEFAULT_INTEGRATION_X_BINS_NM if x_bins is None else x_bins, float)
    out_bins = x_bins if out_bins is None else np.asarray(out_bins, float)
    counts_mg = np.asarray(counts_mg, float)
    if len(counts_mg) != len(x_bins) - 1:
        raise ValueError(f"counts_mg has {len(counts_mg)} values for {len(x_bins) - 1} bins of x_bins")
    total = counts_mg.sum()
    if total <= 0:
        return np.zeros(len(out_bins) - 1)
 
    side_cm = (1.e-3 / sample_density_g_cm3) ** (1. / 3.)
    side_nm = side_cm * 1.e7
    n = int(round(sample_area_cm2 / side_cm ** 2 * total))
 
    k = rng.choice(len(counts_mg), size=n, p=counts_mg / total)            
    L = rng.uniform(x_bins[k], x_bins[k + 1])                              
    if angular_pdf is not None:
        grid = np.linspace(0., np.pi / 2., len(angular_pdf))
        dip = rng.choice(grid, size=n, p=np.asarray(angular_pdf, float) / np.sum(angular_pdf))
    else:
        dip = np.arcsin(rng.uniform(0., 1., size=n))
    z_top = rng.uniform(-side_nm / 2., side_nm / 2., size=n)
 
    keep = (z_top < removed_nm) & (z_top + L * np.sin(dip) > 0.)
    major, minor = etch_pit_axes(L[keep], dip[keep], z_top[keep], removed_nm, v_ratio, lateral_ratio)
 
    resolved = np.isfinite(minor) & (minor >= min_width_nm)
    size = (major if measure == 'major' else minor)[resolved]
    return np.histogram(size, bins=out_bins)[0]

# --- Main Paleodetector Class ---
class Paleodetector:
    """Signal and background track spectra for one mineral.

    Combines the Geant4 recoil populations of the mineral, the depth
    probabilities derived from the StdRock runs, the overburden history and the
    cosmic-ray flux history to compute track-length spectra from spontaneous
    fission, radiogenic neutrons and cosmic-ray particles.

    Attributes:
        config (dict): Mineral configuration (name, composition, density,
            uranium concentration, (alpha,n) coefficient).
        name (str): Mineral name, used to locate its Geant4 data.
        data_path (str): Root of the data directory.
        total_age_kyr (float): Exposure age of the sample [kyr].
        flux_history (FluxHistory or None): Cosmic-ray flux history.
        verbose (int): Verbosity level (0 = silent).
    """
    
    def __init__(self, mineral_config, total_age_kyr, overburden_history=None, flux_history=None, data_path_prefix="Data"):
        """Initialises the detector and builds the depth-probability tables.

        Args:
            mineral_config (dict): Mineral properties. Required keys: 'name',
                'density_g_cm3', 'uranium_concentration_g_g', 'an_coeff_n_nalpha'.
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
            data_path_prefix (str, optional): Root of the data directory.
                Defaults to "Data".
        """
        self.config = mineral_config
        self.name = mineral_config['name']
        self.data_path = data_path_prefix

        self.verbose = 1

        self.radiogenic_spectrum = self._radiogenic_spectrum()
        
        self.alpha_n_spectrum = self._alpha_n_spectrum()

        self.total_age_kyr = total_age_kyr

        self._overburden_interpolator = self._interpolate_overburden_history(overburden_history)

        self.flux_history = None
        self._time_grid_cache = None
        self._annealing_cache = {}
        self.set_flux_history(flux_history)

        self._nuclear_data_cache = {}
        self._neutron_bkg_cache = {}
        self._depth_interpolators = {}
        self._secondary_n_spectrum = {}
        self._geometry_cache = {}
        self._depth_tables = {}

        self.energy_bins = self._scan_geant4_species()

        for species in COMPUTABLE_SPECIES:
            if species in self.energy_bins:
                try:
                    self._load_depth_interpolators(species=species)
                except (FileNotFoundError, ValueError) as err:
                    if self.verbose > 0:
                        print(f"Warning: {species} cannot be computed, no usable StdRock depth table ({err}).")

        if self.verbose > 0:
            self._print_available_species()


    def set_flux_history(self, flux_history):
        """Sets the cosmic-ray flux history (the only way to change it after creation) and rebuilds the time grid.

        Called by __init__ and whenever the flux history changes. The master time grid depends on the overburden
        (fixed at creation) and on the flux history, so it is rebuilt here, at once, and its size printed; the
        annealing cache, which lives on that grid, is cleared.

        Args:
            flux_history (FluxHistory or None): New flux history; None means a baseline-only history.
        """
        if flux_history is None:
            flux_history = FluxHistory(
                baseline={"kind": "Baseline"}, template_dir=os.path.join(self.data_path, "flux_data")
            )
        self.flux_history = flux_history
        self._time_grid_cache = None
        self._annealing_cache = {}
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
                if self.verbose > 0 and _list_geant4_energy_files(os.path.join(mineral_dir, species)):
                    print(f"Warning: {self.name}/{species}: a single Geant4 file, at least 2 are needed; ignored.")
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
            note = ("" if species in self._depth_interpolators else
                    ("  (integrate_atmospheric_neutrinos; secondaries made in "
                     + ("StdRock)" if _list_geant4_energy_files(os.path.join(self.data_path, "Geant4_data", "StdRock",
                                                                             species)) else "the mineral: no StdRock runs)"))
                    if species in NEUTRINO_SPECIES else
                    "  (no StdRock depth table: not computable)" if species in COMPUTABLE_SPECIES else
                    "  (runs found, calculation not implemented yet)")
            lines.append(f"  {species:<10} energy from {e[0]:.4g} to {e[-2]:.4g} GeV in {info['n_files']} steps{note}")
            for lo, hi in info['gaps']:
                lines.append(f"  {'':<10} warning: no file between {lo:.4g} and {hi:.4g} GeV "
                             f"(the {lo:.4g} GeV run is used for the whole step)")
        for derived, product in DERIVED_SPECIES.items():
            if derived in self.available_species():
                sources = [src for src in SECONDARY_SOURCES if self._stdrock_dir(src) is not None]
                lines.append(f"  {derived:<10} {product}s produced by {', '.join(sources)} above the sample "
                             f"(StdRock runs, all generations of the n + p cascade), recoils from the {product} runs")
        print("\n".join(lines))


    def available_species(self):
        """Species that integrate_all_particles can compute with the Geant4 runs present.

        Returns:
            list[str]: Computable species with mineral runs and a StdRock depth table, plus the secondary species
            whose product is among them.
        """
        out = [s for s in COMPUTABLE_SPECIES if s in self.energy_bins and s in self._depth_interpolators]
        out += [d for d, src in DERIVED_SPECIES.items() if src in out]
        return out


    def _energy_edges(self, species, e_range=None):
        """Primary-energy bin edges of a species, optionally restricted to a range.

        Args:
            species (str): Species ('mu-', 'mu+', 'neutron', 'secondary_neutron', ...).
            e_range (tuple[float, float], optional): (E_min, E_max) [GeV]. The bins lying entirely inside it are
                kept. Defaults to None (all the files of the species).

        Returns:
            np.ndarray: Bin edges [GeV].

        Raises:
            ValueError: If the species has no runs, or e_range is outside the available range or contains no bin.
        """
        source = DERIVED_SPECIES.get(species, species)
        if source not in self.energy_bins:
            raise ValueError(f"No Geant4 runs of '{source}' for {self.name}; available: {list(self.energy_bins)}.")
        return _restrict_edges(self.energy_bins[source], e_range, species)


    def _flux_breakpoints(self, nodes=True):
        """Times [kyr, local] where the flux history changes abruptly (event starts and excursion edges, each with
        a point just before) and, if nodes, the template time nodes on which an event's flux is interpolated."""
        if self.flux_history is None:
            return np.zeros(0)
        eps = 1e-9 * self.total_age_kyr
        to_local = self.total_age_kyr
        points = []
        for event in self.flux_history.events:
            start = event["start_time_kyr"] + to_local
            points.append([start - eps, start])
            t_since = getattr(event["template"], "t_since_kyr", None)
            if nodes and t_since is not None:
                points.append(start + np.asarray(t_since, dtype=float) * 1e-3)   # template times are in yr
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
        1e-8 of the age, so that a genuine jump is not refined forever); repeated until no interval needs it, TIME_GRID_MAX_PASSES,
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
        min_dt = 1e-6 * t_now                                   # steps (jumps) are not refined below this
        p_t = self._grid_probes(t)
        for _ in range(TIME_GRID_MAX_PASSES):
            mid = 0.5 * (t[1:] + t[:-1])
            dt = np.diff(t)
            p_mid = self._grid_probes(mid)
            trap = 0.5 * np.abs(p_t[:, 1:] + p_t[:, :-1]) * dt                       # (probes, intervals)
            seg = np.clip(np.searchsorted(seg_edges, mid) - 1, 0, seg_edges.size - 2)
            seg_total = np.zeros((trap.shape[0], seg_edges.size - 1))
            np.add.at(seg_total.T, seg, trap.T)
            err = np.abs(p_mid - 0.5 * (p_t[:, 1:] + p_t[:, :-1])) * dt
            split = np.any(err > TIME_GRID_RTOL * np.maximum(seg_total[:, seg], 1e-300), axis=0) & (dt > min_dt)
            if not split.any() or t.size + split.sum() > TIME_GRID_MAX_POINTS:
                if split.any() and self.verbose > 0:
                    print(f"Warning: time grid stopped at {t.size} points (TIME_GRID_MAX_POINTS).")
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


    def _annealing_ratio(self, t_kyr, annealing):
        """Reduced length r today of tracks formed at the given local times.

        Args:
            t_kyr (array_like): Formation times [kyr].
            annealing (float or dict): A number (same r for all formation times), or a dict: {'law': 'table',
                't_kyr': [...], 'r': [...]} (r at the given formation times, linear in between, constant outside),
                or a law and a geotherm (see _annealing_law, _annealing_from_law), computed once on the master
                grid and cached until the flux history changes; other times are computed directly.

        Returns:
            np.ndarray: r, shape (len(t_kyr),).
        """
        t_kyr = np.atleast_1d(np.asarray(t_kyr, dtype=float))
        if not isinstance(annealing, dict):
            return np.full(t_kyr.shape, float(annealing))
        if annealing.get('law') == 'table':
            nodes, r = np.asarray(annealing['t_kyr'], dtype=float), np.asarray(annealing['r'], dtype=float)
            order = np.argsort(nodes)
            return np.interp(t_kyr, nodes[order], r[order])
        key = repr(sorted(annealing.items()))
        if key not in self._annealing_cache:
            grid = self._master_time_grid()
            self._annealing_cache[key] = (grid, self._annealing_from_law(grid, annealing))
        grid, r_grid = self._annealing_cache[key]
        idx = np.clip(np.searchsorted(grid, t_kyr), 0, grid.size - 1)
        on_grid = grid[idx] == t_kyr
        r = np.empty(t_kyr.shape)
        r[on_grid] = r_grid[idx[on_grid]]
        if not on_grid.all():                     # e.g. window ends: followed exactly, not interpolated
            r[~on_grid] = self._annealing_from_law(t_kyr[~on_grid], annealing)
        return r


    def _temperature_K(self, t_kyr, T_surface_C, gradient_C_per_km, rock_density_g_cm3):
        """Sample temperature [K] at local times: depth [m] = overburden [m.w.e.] / rock density, linear geotherm."""
        depth_m = np.asarray(self._overburden_interpolator(np.asarray(t_kyr, dtype=float))) / rock_density_g_cm3
        return 273.15 + T_surface_C + gradient_C_per_km * depth_m / 1000.0


    def _annealing_from_law(self, t_form_kyr, params):
        """r today of tracks formed at t_form_kyr, from an isothermal law and the temperature history.

        Temperature: from the overburden history and a linear geotherm (params T_surface_C, gradient_C_per_km,
        rock_density_g_cm3; defaults 15 C, 25 C/km, 2.16 g/cm^3; a single density converts m.w.e. to metres).
        Each track is followed from its formation to the present (total_age_kyr), whatever exposure window it was
        formed in, with the equivalent-time method: at every step of the history, its current r is turned into
        the time the law needs to reach it at the step's temperature, the step is added and the law gives the new
        r. The history grid is the master time grid (_master_time_grid) plus the formation times.

        Args:
            t_form_kyr (np.ndarray): Formation times [kyr].
            params (dict): Law and constants (see _annealing_law) and the geotherm keys above.

        Returns:
            np.ndarray: r, shape (len(t_form_kyr),).
        """
        p = {**_ANNEALING_GEOTHERM_DEFAULTS, **params}
        r_iso, t_iso = _annealing_law(p)
        t_now = float(self.total_age_kyr)
        t_form = np.clip(np.asarray(t_form_kyr, dtype=float), 0.0, t_now)
        edges = np.union1d(self._master_time_grid(), t_form)
        T_mid = self._temperature_K(0.5 * (edges[1:] + edges[:-1]), p['T_surface_C'], p['gradient_C_per_km'],
                                    p['rock_density_g_cm3'])
        dt = np.diff(edges)
        first_step = np.searchsorted(edges, t_form)          # step that starts at each formation time
        r = np.ones(t_form.shape)
        for j in range(dt.size):                             # tracks formed before step j anneal through it
            active = first_step <= j
            if not np.any(active):
                continue
            r_now = r[active]
            t_eq = np.where(r_now >= _R_FRESH, 0.0, t_iso(r_now, T_mid[j]))
            r[active] = np.minimum(r_now, r_iso(t_eq + dt[j], T_mid[j]))
        return r


    def _age_integrated(self, x_grid, density, annealing=None, annealing_r_min=0.0):
        """Tracks accumulated over the sample age from a source constant in time (fission, radiogenic neutrons).

        Without annealing this is density * total_age_kyr. Otherwise the tracks formed at each time of the master
        grid are shortened by their r (see _annealing_ratio) and the result is integrated over the age.

        Args:
            x_grid (np.ndarray): Track-length bin edges [nm].
            density (np.ndarray): Rate per bin of x_grid [tracks per kyr per nm], shape (len(x_grid) - 1,).
            annealing, annealing_r_min: See integrate_particle_signal_spectrum.

        Returns:
            np.ndarray: Accumulated density [tracks per nm], shape (len(x_grid) - 1,).
        """
        density = np.asarray(density, dtype=float)
        if annealing is None:
            return density * self.total_age_kyr
        t = self._master_time_grid()
        rows = np.broadcast_to(density, (t.size, density.size))
        rows = _shorten_tracks(rows, x_grid, self._annealing_ratio(t, annealing), annealing_r_min)
        return np.trapezoid(rows, t, axis=0)


    def _exposure_window(self, exposure_window_kyr):
        """[start, end] of an exposure window [kyr]: None means the whole age, a number w means [0, w]."""
        if exposure_window_kyr is None:
            exposure_window_kyr = self.total_age_kyr
        if isinstance(exposure_window_kyr, (int, float)):
            exposure_window_kyr = [0, exposure_window_kyr]
        return [float(exposure_window_kyr[0]), float(exposure_window_kyr[1])]


    def _resolve_energy_bins(self, energy_bins_gev, species):
        """The given energy bins, or all the bins of the species if None."""
        return self._energy_edges(species) if energy_bins_gev is None else np.asarray(energy_bins_gev, dtype=float)


    def _interpolate_overburden_history(self, overburden_history=None):
        """Builds the overburden [m.w.e.] as a function of local time.

        The overburden starts at initial_depth * initial_density_g_cm3 and grows
        through continuous deposition phases (rate * density over each time window)
        and discrete deposition events (thickness * density from a given time on).

        Args:
            overburden_history (dict or float, optional): Either a constant
                overburden [m.w.e.], or a dict with the optional keys
                'initial_depth' [m], 'initial_density_g_cm3',
                'start_time_continuous_kyr', 'end_time_continuous_kyr',
                'rate_continuous' [m/kyr, negative for erosion], 'density_continuous_g_cm3',
                'time_discrete_kyr', 'overburden_discrete' [m, negative for removal],
                'density_discrete_g_cm3'. Defaults to None (0 m.w.e.).

        Returns:
            scipy.interpolate.interp1d: Local time [kyr] -> overburden [m.w.e.].
        """
        self._overburden_breakpoints = np.array([0.0, float(self.total_age_kyr)])
        if isinstance(overburden_history, (int, float)):
            const = float(overburden_history)
            return interp1d([0.0, self.total_age_kyr], [const, const],
                             bounds_error=False, fill_value=const)
        elif overburden_history is None:
            return interp1d([0.0, self.total_age_kyr], [0.0, 0.0],
                             bounds_error=False, fill_value=0.0)

        initial_depth = overburden_history.get("initial_depth", 0.3)
        initial_density_g_cm3 = overburden_history.get("initial_density_g_cm3", 1.0)

        start_time_continuous_kyr = np.atleast_1d(overburden_history["start_time_continuous_kyr"]) if "start_time_continuous_kyr" in overburden_history else [0.0]
        end_time_continuous_kyr = np.atleast_1d(overburden_history["end_time_continuous_kyr"]) if "end_time_continuous_kyr" in overburden_history else [self.total_age_kyr]
        rate_continuous = np.atleast_1d(overburden_history["rate_continuous"]) if "rate_continuous" in overburden_history else [0.0]
        density_continuous_g_cm3 = np.atleast_1d(overburden_history["density_continuous_g_cm3"]) if "density_continuous_g_cm3" in overburden_history else [1.0]

        time_discrete = np.atleast_1d(overburden_history["time_discrete_kyr"]) if "time_discrete_kyr" in overburden_history else [0.0]
        overburden_discrete = np.atleast_1d(overburden_history["overburden_discrete"]) if "overburden_discrete" in overburden_history else [0.0]
        density_discrete_g_cm3 = np.atleast_1d(overburden_history["density_discrete_g_cm3"]) if "density_discrete_g_cm3" in overburden_history else [1.0]

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
            if self.verbose > 0:
                print(f"Warning: the overburden history erodes below the surface (down to {overburdens.min():.3g} "
                      "m.w.e.); it is set to 0 there.")
            overburdens = np.maximum(overburdens, 0.0)

        if self.verbose>0:
            print("Interpolating overburden history...")

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
        (calculate_fission_spectrum) and the SF neutrons (_radiogenic_spectrum).

        Returns:
            float: Fissions per g of mineral per kyr,
            uranium_concentration_g_g * U238_ABUNDANCE * FISSIONS_U238_G_KYR.
        """
        return self.config["uranium_concentration_g_g"] * U238_ABUNDANCE * FISSIONS_U238_G_KYR


    def _radiogenic_spectrum(self):
        """Neutron emission spectrum from U-238 spontaneous fission.

        Watt spectrum exp(-E/a) sinh(sqrt(b E)) with a = WATT_A_MEV and
        b = WATT_B_PER_MEV (Cranberg et al., Phys. Rev. 103, 662 (1956)),
        normalised to the fission rate of _fission_rate_per_g_kyr times
        NU_BAR_U238_SF neutrons per fission, so the neutron yield and the
        fragment tracks come from the same fission rate.

        Returns:
            scipy.interpolate.interp1d: Energy [GeV] -> neutrons per g per s per GeV.
        """
        sf_yield_per_g_s = self._fission_rate_per_g_kyr() * NU_BAR_U238_SF * KYR_PER_SECOND
        energies = np.logspace(-6, -1, 10000)                       # [GeV]

        energies_mev = energies * 1e3
        watt_shape = np.exp(-energies_mev / WATT_A_MEV) * np.sinh(np.sqrt(WATT_B_PER_MEV * energies_mev))
        
        sf_flux = watt_shape * (sf_yield_per_g_s/ np.trapezoid(watt_shape, energies))

        interpolator = interp1d(energies, sf_flux, bounds_error=False, fill_value='extrapolate')

        return interpolator

    
    def _alpha_n_spectrum(self):
        """Neutron emission spectrum from (alpha,n) reactions.

        Shape E^2 exp(-beta E) with a smooth cut-off, normalised to the (alpha,n)
        yield of the U-238 chain (8 alphas per chain) times the mineral coefficient
        an_coeff_n_nalpha (Kudryavtsev et al., SciPost Phys. Proc. 12 (2023), SOURCES4).

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


    def _process_background_neutron_geant4_data(self, background_type, energy_bins_gev=None, total_simulated_particles=1e4):
        """Recoil spectrum induced by radiogenic neutrons.

        The neutrons are produced inside the mineral, so every recoil they cause is
        counted (infinite homogeneous medium): for each energy bin, the recoils per
        simulated neutron are weighted by the number of neutrons emitted in that bin.

        Args:
            background_type (str): 'fission_n' (spontaneous fission) or 'alpha_n'.
            energy_bins_gev (np.ndarray, optional): Neutron energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.

        Returns:
            dict: 'Er_bins' (recoil energy edges [MeV]) and, for each fragment,
            dR/dEr [recoils per g per kyr per MeV].

        Raises:
            ValueError: If background_type is unknown.
        """
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, 'neutron')

        
        if background_type == 'fission_n':
            flux_interpolator = self.radiogenic_spectrum
        elif background_type == 'alpha_n':
            flux_interpolator = self.alpha_n_spectrum
        else:
            raise ValueError(f"Unknown background_type '{background_type}' "
                             "(expected 'fission_n' or 'alpha_n').")

        all_fragments = self._get_all_fragments(energy_bins_gev[:-1], 'neutron')

        geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, "neutron")
                
        all_recoil_spectra = {}

        fragment_spectra = {frag: np.zeros(len(RECOIL_ENERGY_BINS_MEV) - 1) for frag in all_fragments}

        for i in range(len(energy_bins_gev) - 1):
            e_min = energy_bins_gev[i]
            e_max = energy_bins_gev[i+1]

            N_E = 50
            e_vals = np.linspace(e_min, e_max, N_E)

            flux_grid = flux_interpolator(e_vals)

            weight = np.trapezoid(flux_grid, e_vals)

            filepath = os.path.join(geant4_input_dir, f"outNuclei_{e_min:.6f}.txt")
            if not os.path.exists(filepath): continue

            try:
                df = pd.read_csv(filepath, sep=r'\s+', header=None, usecols=[0, 2], 
                                names=['name', 'rec_e'], dtype={'name': str, 'rec_e': float})
                names = normalize_fragment_names(df['name'].values)
                rec_energies = df['rec_e'].values
            except pd.errors.EmptyDataError:
                continue

            valid_mask = np.isin(names, list(fragment_spectra.keys()))
            valid_names = names[valid_mask]
            valid_rec = rec_energies[valid_mask]

            unique_frags = np.unique(valid_names)

            for frag in unique_frags:
                frag_mask = (valid_names == frag)
                frag_rec_energies = valid_rec[frag_mask]

                counts, _ = np.histogram(frag_rec_energies, bins=RECOIL_ENERGY_BINS_MEV)
                fragment_spectra[frag] += counts * weight

        all_recoil_spectra.update(fragment_spectra)

        bin_widths_mev = np.diff(RECOIL_ENERGY_BINS_MEV)
        norm_factor = (bin_widths_mev * total_simulated_particles * KYR_PER_SECOND)

        normalized_spectra = {}
        for name, spectrum in all_recoil_spectra.items():
            normalized_spectra[name] = np.divide(spectrum, norm_factor, out=np.zeros_like(spectrum), where=norm_factor!=0)

        return {
            'Er_bins': RECOIL_ENERGY_BINS_MEV,
            **normalized_spectra,
        }

    
    def calculate_background_neutron_spectrum(self, 
        x_bins=None, 
        energy_bins_gev=None, 
        background_types=('fission_n', 'alpha_n'),
        total_simulated_particles=1e4, 
        nucleus=DEFAULT_TRACK_SUM,
        ): 
        """Track-length rate from radiogenic neutrons.

        Args:
            x_bins (np.ndarray, optional): Track-length bin edges [nm]. Defaults to
                None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Neutron energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            background_types (tuple[str], optional): Sources to include.
                Defaults to ('fission_n', 'alpha_n').
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key.
                Defaults to 'total_no_light_ions'.

        Returns:
            np.ndarray: dR/dx summed over sources [tracks per mg per kyr per nm].
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, 'neutron')

        x_mids = x_bins[:-1] + np.diff(x_bins) / 2.0

        sum_drdx = np.zeros_like(x_mids)
        found = False

        for background_type in background_types:
            print(f"Processing background neutron tracks from {background_type}")

            recoil_data = self._process_background_neutron_geant4_data(background_type, energy_bins_gev, total_simulated_particles)

            drdx = self._convert_recoil_to_track_spectrum(recoil_data, x_bins=x_bins, energy_bins_gev=energy_bins_gev, species='neutron')

            if nucleus in drdx:                                    # a fragment may be absent from one source
                sum_drdx += drdx[nucleus]
                found = True

        if not found:
            raise ValueError(f"nucleus '{nucleus}' not found: use one of {TRACK_SUMS} or a fragment key.")
        return sum_drdx


    def integrate_background_neutron_spectrum(
        self, 
        x_bins=None, 
        energy_bins_gev=None,  
        background_types=('fission_n', 'alpha_n'),
        total_simulated_particles=1e4, 
        x_grid=TRACK_LENGTH_BINS_NM, 
        nucleus=DEFAULT_TRACK_SUM,
        annealing=None,
        annealing_r_min=0.0,
        ):
        """Track counts from radiogenic neutrons over the sample age.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Neutron energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            background_types (tuple[str], optional): Sources to include.
                Defaults to ('fission_n', 'alpha_n').
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            x_grid (np.ndarray, optional): Fine track-length grid used for the
                calculation [nm]. Defaults to TRACK_LENGTH_BINS_NM.
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key.
                Defaults to 'total_no_light_ions'.
            annealing (float or dict, optional): Reduced track length r = L_today / L_formed of a track formed
                at time t (0 < r <= 1): every track formed at t is shortened to r(t) times its length
                (proportional shortening, as in fission-track annealing models). A number (same r for every
                formation time), a table {'law': 'table', 't_kyr': [...], 'r': [...]} (r at those formation
                times, linear in between, constant outside), or a law with a geotherm, e.g.
                {'law': 'fanning_curvilinear', 'C0': ..., 'C1': ..., 'C2': ..., 'C3': ..., 'alpha': ...,
                'beta': ..., 'time_unit_s': ..., 'T_surface_C': 15, 'gradient_C_per_km': 25,
                'rock_density_g_cm3': 2.16} or {'law': 'first_order', 'A_per_kyr': ..., 'Ea_eV': ...}: then each
                track is followed from its formation to the present through the temperature given by the
                overburden history (see _annealing_law and _annealing_from_law). Defaults to None (no shortening).
            annealing_r_min (float, optional): Tracks with r below this value are removed (segmented, no longer
                etchable; ~0.4-0.55 for apatite). Defaults to 0 (none removed).

        Returns:
            tuple[np.ndarray, np.ndarray]: x_bins, and tracks per mg in each bin
            accumulated over total_age_kyr (after annealing, if given).
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, 'neutron')

        sum_drdx = self.calculate_background_neutron_spectrum(
            x_grid, 
            energy_bins_gev, 
            background_types,
            total_simulated_particles, 
            nucleus=nucleus,
            )

        sum_drdx = self._age_integrated(x_grid, sum_drdx, annealing, annealing_r_min)

        total_tracks_g = _rebin_density(sum_drdx, x_grid, x_bins)

        return x_bins, total_tracks_g
    
    def calculate_fission_spectrum(self, x_bins=TRACK_LENGTH_BINS_NM):
        """Track-length rate from U-238 spontaneous fission.

        For each fission event in U238.dat, the kinetic energies of the two fragments
        follow from two-body kinematics with masses from BindingEne.txt; each energy
        is converted to a range with the mineral's dE/dx tables and the track length
        is the sum of the two ranges. The fragments are taken before neutron
        emission (A1 + A2 = 238; a warning is printed otherwise), and the rate is
        _fission_rate_per_g_kyr, the same as for the SF neutrons.

        Args:
            x_bins (np.ndarray, optional): Track-length bin edges [nm].
                Defaults to TRACK_LENGTH_BINS_NM.

        Returns:
            np.ndarray: dR/dx [tracks per g per kyr per nm].

        Raises:
            ValueError: If no event has binding energies for both fragments.
        """
        if self.verbose>0:
            print("Calculating spontaneous fission background...")
        
        Z_fission, A_fission, _ = self._load_nuclear_data("U238.dat")
        Z_bind, A_bind, B_bind = self._load_nuclear_data("BindingEne.txt")
        binding_map = {(int(z), int(a)): b for z, a, b in zip(Z_bind, A_bind, B_bind)}

        fission_rate_factor = self._fission_rate_per_g_kyr()

        range_dir = os.path.join(self.data_path, "Geant4_data", self.name, "Range")
        M0 = 92 * PROTON_MASS_MEV + (238 - 92) * NEUTRON_MASS_MEV - B0_U238 * 238
        range_cache = {}

        def energy_to_range_nm(z, a):
            """Range-energy interpolator of fragment (z, a), loaded once.

            Args:
                z (int): Atomic number.
                a (int): Mass number.

            Returns:
                scipy.interpolate.interp1d: Kinetic energy [MeV] -> CSDA range [nm].
            """
            if (z, a) not in range_cache:
                e_mev, _, _, length_um = np.loadtxt(
                    os.path.join(range_dir, f"DEDX_Z{z}_A{a}.txt"), skiprows=1, unpack=True
                )
                range_cache[(z, a)] = interp1d(e_mev, length_um * 1e3,
                                               bounds_error=False, fill_value=0.0)
            return range_cache[(z, a)]

        total_track_lengths_nm = []
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

            track1 = energy_to_range_nm(z1, a1)(Ek1_MeV)
            track2 = energy_to_range_nm(z2, a2)(Ek2_MeV)
            total_track_lengths_nm.append(track1 + track2)

        n_used = num_events - n_skipped
        if n_used == 0:
            raise ValueError("No fission event with binding energies for both fragments.")
        if n_not_238 and self.verbose > 0:
            print(f"Warning: {n_not_238}/{num_events} fission events with A1 + A2 != 238; "
                  "the two-body kinematics assume fragments before neutron emission.")
        if n_skipped and self.verbose > 0:
            print(f"Warning: {n_skipped}/{num_events} fission events skipped "
                  "(fragment missing from BindingEne.txt).")

        counts, bin_edges = np.histogram(total_track_lengths_nm, bins=x_bins)
        bin_widths = np.diff(bin_edges)

        # normalised to the events actually used, so skipped events do not count as zero-length tracks
        dRdx = (counts / n_used) * fission_rate_factor / bin_widths

        return dRdx

    
    def integrate_fission_spectrum(self, x_bins=None, x_grid=TRACK_LENGTH_BINS_NM, annealing=None,
                                  annealing_r_min=0.0):
        """Track counts from spontaneous fission over the sample age.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            x_grid (np.ndarray, optional): Fine track-length grid used for the
                calculation [nm]. Defaults to TRACK_LENGTH_BINS_NM.
            annealing (float or dict, optional): Reduced track length r = L_today / L_formed of a track formed
                at time t (0 < r <= 1): every track formed at t is shortened to r(t) times its length
                (proportional shortening, as in fission-track annealing models). A number (same r for every
                formation time), a table {'law': 'table', 't_kyr': [...], 'r': [...]} (r at those formation
                times, linear in between, constant outside), or a law with a geotherm, e.g.
                {'law': 'fanning_curvilinear', 'C0': ..., 'C1': ..., 'C2': ..., 'C3': ..., 'alpha': ...,
                'beta': ..., 'time_unit_s': ..., 'T_surface_C': 15, 'gradient_C_per_km': 25,
                'rock_density_g_cm3': 2.16} or {'law': 'first_order', 'A_per_kyr': ..., 'Ea_eV': ...}: then each
                track is followed from its formation to the present through the temperature given by the
                overburden history (see _annealing_law and _annealing_from_law). Defaults to None (no shortening).
            annealing_r_min (float, optional): Tracks with r below this value are removed (segmented, no longer
                etchable; ~0.4-0.55 for apatite). Defaults to 0 (none removed).

        Returns:
            tuple[np.ndarray, np.ndarray]: x_bins, and tracks per mg in each bin
            accumulated over total_age_kyr (after annealing, if given).
        """
        x_bins = _resolve_x_bins(x_bins)

        drdx = self._age_integrated(x_grid, self.calculate_fission_spectrum(x_grid), annealing, annealing_r_min)

        total_tracks_g = _rebin_density(drdx, x_grid, x_bins)

        total_tracks_mg = total_tracks_g * 1e-3

        return x_bins, total_tracks_mg


    def _geant4_geometry(self, data_dir):
        """Geometry of a Geant4 output directory.

        Read from <data_dir>/geometry.json, written by the steering notebook.

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
            "depth_reference_z_m": 100.,
            "target_length_m": 2.0 * 100.,
            "density_g_cm3": 2.65,
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


    def _geant4_depth_reference_m(self, data_dir):
        """z coordinate of the target top face of a Geant4 output directory.

        Depth below the top face is depth_reference_z_m - z_mm * 1e-3.

        Args:
            data_dir (str): Geant4 output directory.

        Returns:
            float: z of the target top face [m].
        """
        return self._geant4_geometry(data_dir)["depth_reference_z_m"]


    def _load_depth_interpolators(self, species='mu-'):
        """Sets up the depth-probability model of a species from the StdRock runs.

        Muons (mu- and mu+ share the mu- table) use the empirical distribution of the
        stopping depth R. It gives the capture probability in a slab (mu- only),
        F(z_max) - F(z_min), and the normal-interaction probability, with interaction
        density proportional to P(R > z): (G(z_max) - G(z_min)) / E[R].

        Hadrons (neutron, proton: StdRock/<species>) use the empirical distribution of the
        interaction depths up to the NEUTRON_TAIL_P quantile, with an exponential tail
        beyond it; the probability of an interaction in a slab is F(z_max) - F(z_min).
        For protons the interactions stop where the proton stops, which the empirical
        distribution contains.

        Every outNuclei_<E>.txt in the StdRock directory is used, so a denser energy
        grid is picked up automatically.

        Args:
            species (str, optional): 'mu-', 'mu+', 'neutron' or 'proton'. Defaults to 'mu-'.
        """
        tab_species = 'mu-' if species in MUON_SPECIES else species
        if tab_species not in self._depth_tables:
            self._depth_tables[tab_species] = self._build_depth_table(tab_species)

        self._depth_interpolators[species] = {'table': tab_species}


    def _build_depth_table(self, tab_species):
        """Builds the depth quantile table of one StdRock species.

        Energies with fewer than MIN_DEPTH_SAMPLES depths, or whose depths reach
        TRUNCATION_FRACTION of the StdRock target (the particle may have left the
        target), are skipped with a warning.

        Args:
            tab_species (str): 'mu-' (stopping depths, rows with remaining primary
                energy 0) or a hadron, 'neutron' or 'proton' (all interaction depths).

        Returns:
            dict: 'energies' [GeV], 'p' (node probabilities), 'q' (quantile nodes
            [m.w.e.], shape (n_energies, n_nodes)) and, for hadrons, 'tail_lambda'
            (exponential-tail length per energy [m.w.e.]).

        Raises:
            FileNotFoundError: If the StdRock directory has no output files.
            ValueError: If fewer than two energies are usable.
        """
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

            # muons: stopping depths; hadrons: all interaction depths
            sample = depth_mwe[df['rem_e'].values == 0.0] if is_muon else depth_mwe
            if sample.size < MIN_DEPTH_SAMPLES:
                skipped.append(e0)
                continue
            if sample.max() > TRUNCATION_FRACTION * target_mwe:
                truncated.append(e0)
                continue
            q = np.quantile(sample, p_nodes)
            if not is_muon:
                # exponential tail: maximum-likelihood length = mean excess beyond the last node
                excess = sample[sample > q[-1]] - q[-1]
                lam_tail = excess.mean() if excess.size > 0 else sample.mean()
                tails.append(max(lam_tail, DEPTH_FLOOR_MWE))
            rows.append(np.maximum(q, DEPTH_FLOOR_MWE))
            energies.append(e0)

        if self.verbose > 0 and (skipped or truncated):
            if skipped:
                print(f"Warning: StdRock/{tab_species}: skipped {len(skipped)} energies with too few "
                      f"depth samples (e.g. {skipped[:3]} GeV).")
            if truncated:
                print(f"Warning: StdRock/{tab_species}: skipped {len(truncated)} energies whose "
                      f"depths reach the end of the target (from {min(truncated):g} GeV); "
                      f"use a longer StdRock target to cover them.")
        if len(energies) < 2:
            raise ValueError(f"StdRock/{tab_species}: need at least 2 usable energies, found {len(energies)}.")

        table = {'energies': np.array(energies), 'p': p_nodes, 'q': np.array(rows)}
        if not is_muon:
            table['tail_lambda'] = np.array(tails)
        return table


    def _depth_quantiles(self, energies_gev, tab_species='mu-'):
        """Depth quantile nodes at arbitrary energies.

        Interpolated in log-log between the tabulated energies, then forced
        non-decreasing along the nodes.

        Args:
            energies_gev (array_like): Energies [GeV], shape (n,).
            tab_species (str, optional): 'mu-' (stopping depths), 'neutron' or 'proton'
                (interaction depths). Defaults to 'mu-'.

        Returns:
            np.ndarray: Quantile nodes [m.w.e.], shape (n, n_nodes).
        """
        table = self._depth_tables[tab_species]
        q = _loglog_interp_rows(energies_gev, table['energies'], table['q'])
        return np.maximum.accumulate(q, axis=-1)


    def _muon_depth_probabilities(self, e_vals, z_min_grid, z_max_grid):
        """Muon interaction probabilities in a slab.

        Args:
            e_vals (np.ndarray): Muon energies [GeV], shape (n_E,), along the last
                axis of the z grids.
            z_min_grid (np.ndarray): Slab top along the path [m.w.e.], shape (..., n_E).
            z_max_grid (np.ndarray): Slab bottom along the path [m.w.e.], same shape.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                prob_tail: normal interactions, (G(z_max) - G(z_min)) / E[R].
                prob_peak: stopping (capture for mu-), F(z_max) - F(z_min).
        """
        q_rows = self._depth_quantiles(e_vals, 'mu-')
        p = self._depth_tables['mu-']['p']
        prob_tail = np.empty_like(z_min_grid, dtype=float)
        prob_peak = np.empty_like(z_min_grid, dtype=float)
        for j in range(len(e_vals)):
            F_lo, G_lo = _piecewise_cdf_and_integral(z_min_grid[..., j], q_rows[j], p)
            F_hi, G_hi = _piecewise_cdf_and_integral(z_max_grid[..., j], q_rows[j], p)
            _, mean_range = _piecewise_cdf_and_integral(np.inf, q_rows[j], p)
            prob_tail[..., j] = (G_hi - G_lo) / mean_range
            prob_peak[..., j] = F_hi - F_lo
        return prob_tail, prob_peak


    def _hadron_depth_probability(self, e_vals, z_min_grid, z_max_grid, tab_species='neutron'):
        """Probability that a hadron interacts in a slab.

        Args:
            e_vals (np.ndarray): Hadron energies [GeV], shape (n_E,), along the last
                axis of the z grids.
            z_min_grid (np.ndarray): Slab top along the path [m.w.e.], shape (..., n_E).
            z_max_grid (np.ndarray): Slab bottom along the path [m.w.e.], same shape.
            tab_species (str, optional): 'neutron' or 'proton'. Defaults to 'neutron'.

        Returns:
            np.ndarray: F(z_max) - F(z_min), same shape as z_min_grid.
        """
        table = self._depth_tables[tab_species]
        q_rows = self._depth_quantiles(e_vals, tab_species)
        tail_lambdas = self._hadron_tail_lambdas(e_vals, tab_species)
        return _slab_probability(q_rows, table['p'], z_min_grid, z_max_grid, tail_lambdas)


    def _hadron_tail_lambdas(self, energies_gev, tab_species='neutron'):
        """Exponential-tail length of a hadron depth distribution.

        Args:
            energies_gev (array_like): Energies [GeV], shape (n,).
            tab_species (str, optional): 'neutron' or 'proton'. Defaults to 'neutron'.

        Returns:
            np.ndarray: Tail length [m.w.e.], shape (n,), log-log interpolated.
        """
        table = self._depth_tables[tab_species]
        return _loglog_interp_rows(energies_gev, table['energies'], table['tail_lambda'])[:, 0]


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
        table = self._depth_tables[product]
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
        files' own grid (_geant4_energy_edges), which keeps every simulated primary energy.

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


    def _stdrock_offset_matrix(self, source, product, source_bins_gev, product_bins_gev, delta_mwe, n_offsets,
                               total_simulated_particles=1e4):
        """Particles of type `product` produced by a `source` primary, by distance travelled and energy.

        One block of the generation matrix of the atmospheric secondary cascade (see _secondary_yield): from the
        StdRock/<source> runs, C[l, d, m] is the number of products with energy in cell m (of product_bins_gev)
        produced between d * delta and (d + 1) * delta after the entry point, per primary in energy cell l (of
        source_bins_gev). source == product gives the same-type generations, source != product the
        cross-generations (neutrons from protons and protons from neutrons).

        Args:
            source (str): Primary of the StdRock runs, 'neutron' or 'proton'.
            product (str): Produced particle, 'neutron' or 'proton'.
            source_bins_gev (np.ndarray): Energy cell edges of the source [GeV].
            product_bins_gev (np.ndarray): Energy cell edges of the product [GeV].
            delta_mwe (float): Offset bin width [m.w.e.].
            n_offsets (int): Number of offset bins.
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.

        Returns:
            np.ndarray: C, shape (len(source_bins_gev) - 1, n_offsets, len(product_bins_gev) - 1).
        """
        rows = self._load_stdrock_rows(source, product, product_bins_gev, total_simulated_particles,
                                       primary_bins_gev=source_bins_gev)
        n_s, n_q = len(source_bins_gev) - 1, len(product_bins_gev) - 1
        d = np.floor(rows['depth_mwe'] / delta_mwe).astype(int)
        ok = d < n_offsets
        flat = (rows['primary_cell'][ok] * n_offsets + d[ok]) * n_q + rows['cell'][ok]
        C = np.bincount(flat, minlength=n_s * n_offsets * n_q).astype(float)
        return C.reshape(n_s, n_offsets, n_q) * rows['weight']


    def _secondary_yield(self, t_kyr_array, depth_mwe_array, c_vals, product_bins_gev,
                         total_simulated_particles=1e4, species_list=SECONDARY_SOURCES, n_energy_sub=50):
        """Secondary neutrons and protons born in a window above the sample, per direction, as one cascade.

        For each time t and direction c, the path to the sample has slant depth S = X(t) / c. The window
        [S - W, S] along the path (W: the largest _secondary_window_mwe of the products) is split into
        SECONDARY_WINDOW_BINS bins of width delta = W / SECONDARY_WINDOW_BINS.

        State: the products of every type on one flat axis, product q occupying offsets[q] (its energy cells).

        Source: products made in the window by the primaries, from the StdRock runs at their actual slant depth
        (so the transport of the primary down to the window is included), weighted by the primary intensity
        integrated over each energy cell of the primary's StdRock grid:

            A_b,k = sum_s sum_i Phi_s,i(t) * (StdRock/<s> products of cell k in bin b per primary of cell i)

        Generations inside the window: the StdRock/<p> runs of every product p give the products of every type
        q it makes (_stdrock_offset_matrix blocks C[p -> q]), so neutrons from neutrons, protons from protons and
        the cross-generations (neutrons from protons, protons from neutrons) are all followed, to any order.
        Products are born uniformly within their bin, so a production at offset bin d is shared between bins
        b' + d and b' + d + 1:

            Y_b (I - C[0] / 2) = A_b + sum_{b' < b} Y_b' (C[b - b' - 1] + C[b - b']) / 2

        Secondaries are assumed to keep the direction of the primary. Sources without StdRock runs are skipped;
        a product without StdRock runs makes no further generations (reported).

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            depth_mwe_array (np.ndarray): Overburden [m.w.e.], shape (T,).
            c_vals (np.ndarray): cos(zenith) grid, shape (N_C,).
            product_bins_gev (dict): {product: energy cell edges [GeV]} of the products followed ('neutron',
                'proton'; see _secondary_grids).
            total_simulated_particles (float, optional): Primaries per Geant4 run. Defaults to 1e4.
            species_list (tuple[str], optional): Primary species producing the secondaries.
                Defaults to SECONDARY_SOURCES.
            n_energy_sub (int, optional): Points per energy cell for the flux integration. Defaults to 50.

        Returns:
            tuple[np.ndarray, float, dict]:
                Y: products born in each window bin and cell [cm^-2 s^-1 sr^-1], shape (T, N_C, SECONDARY_WINDOW_BINS, K).
                delta: window bin width along the path [m.w.e.].
                offsets: {product: slice of the last axis of Y}.
        """
        products = list(product_bins_gev)
        product_bins_gev = {q: np.asarray(e, dtype=float) for q, e in product_bins_gev.items()}
        offsets, n_k = {}, 0
        for q in products:
            offsets[q] = slice(n_k, n_k + len(product_bins_gev[q]) - 1)
            n_k += len(product_bins_gev[q]) - 1

        n_t, n_c = len(t_kyr_array), len(c_vals)
        n_b = SECONDARY_WINDOW_BINS
        window = max(self._secondary_window_mwe(q) for q in products)
        delta = window / n_b

        t_kyr_flux_array = self._flux_time_kyr(np.asarray(t_kyr_array, dtype=float))
        sub = np.linspace(0.0, 1.0, n_energy_sub)
        slant = depth_mwe_array[:, None] / c_vals[None, :]               # (T, N_C) path to the sample
        window_top = slant - window

        A = np.zeros((n_t, n_c, n_b, n_k))
        for species in species_list:
            if self._stdrock_dir(species) is None:
                if self.verbose > 1:
                    print(f"No StdRock runs for {species}: no secondaries from it.")
                continue
            phi_cell = None
            for q in products:
                rows = self._load_stdrock_rows(species, q, product_bins_gev[q], total_simulated_particles)
                if phi_cell is None:                     # same primary grid for every product of this species
                    if rows['missing'] and self.verbose > 0:
                        print(f"Warning: StdRock/{species}: no file for {len(rows['missing'])} primary "
                              f"energies (e.g. {rows['missing'][:3]} GeV); their secondaries are missing.")
                    beyond = slant > TRUNCATION_FRACTION * rows['target_mwe']
                    if self.verbose > 0 and np.any(beyond[:, -1]):
                        print(f"Warning: vertical overburden beyond the StdRock/{species} target "
                              f"({rows['target_mwe']:.0f} m.w.e.) at some times: secondaries "
                              "from that species are missing there; use a longer StdRock target.")
                    # primary intensity integrated over each primary energy cell [cm^-2 s^-1 sr^-1]
                    pb = rows['primary_bins']
                    e_lo, e_hi = pb[:-1], pb[1:]
                    e_sub = e_lo[:, None] + (e_hi - e_lo)[:, None] * sub[None, :]
                    step = (e_hi - e_lo) / (n_energy_sub - 1)
                    phi = self.flux_history.get_map(species, t_kyr_flux_array, e_sub)[2].reshape(
                        n_t, *e_sub.shape
                    )
                    phi_cell = (phi.sum(axis=-1) - 0.5 * (phi[..., 0] + phi[..., -1])) * step[None, :]

                n_q = len(product_bins_gev[q]) - 1
                depth = rows['depth_mwe']
                lo_idx = np.searchsorted(depth, window_top, side='left')
                hi_idx = np.searchsorted(depth, slant, side='left')
                for t in range(n_t):
                    for c in range(n_c):
                        if beyond[t, c] or hi_idx[t, c] <= lo_idx[t, c]:
                            continue
                        sl = slice(lo_idx[t, c], hi_idx[t, c])
                        b = np.minimum(((depth[sl] - window_top[t, c]) / delta).astype(int), n_b - 1)
                        w = phi_cell[t, rows['primary_cell'][sl]] * rows['weight']
                        A[t, c, :, offsets[q]] += np.bincount(
                            b * n_q + rows['cell'][sl], weights=w, minlength=n_b * n_q
                        ).reshape(n_b, n_q)

        # generations inside the window: every product type from every product type
        C = np.zeros((n_k, n_b + 1, n_k))
        for p in products:
            if self._stdrock_dir(p) is None:
                if self.verbose > 0:
                    print(f"Warning: no StdRock/{p} runs: the secondaries made by {p}s are not followed.")
                continue
            for q in products:
                C[offsets[p], :, offsets[q]] = self._stdrock_offset_matrix(
                    p, q, product_bins_gev[p], product_bins_gev[q], delta, n_b + 1, total_simulated_particles
                )
        half_same_bin = 0.5 * C[:, 0, :]
        if np.max(np.abs(np.linalg.eigvals(half_same_bin))) < 1.0:
            same_bin_inv = np.linalg.inv(np.eye(n_k) - half_same_bin)
        else:
            if self.verbose > 0:
                print("Warning: same-bin secondary multiplication >= 1; ignoring it.")
            same_bin_inv = np.eye(n_k)
        transfer = [None] + [0.5 * (C[:, k - 1, :] + C[:, k, :]) for k in range(1, n_b + 1)]

        A = A.reshape(n_t * n_c, n_b, n_k)
        Y = np.zeros_like(A)
        for b in range(n_b):
            y = A[:, b, :].copy()
            for bp in range(b):
                y += Y[:, bp, :] @ transfer[b - bp]
            Y[:, b, :] = y @ same_bin_inv

        return Y.reshape(n_t, n_c, n_b, n_k), delta, offsets


    def _get_all_fragments(self, energy_names_gev, species='mu-'):
        """Lists the fragments present in a mineral's Geant4 output.

        Args:
            energy_names_gev (array_like): Energies [GeV] of the files to scan.
            species (str, optional): 'mu-', 'mu+', 'neutron' or 'proton'. Defaults to 'mu-'.

        Returns:
            list[str]: Sorted fragment keys (see normalize_fragment_name).
        """
        geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, species)
        all_fragments = set()
        
        for energy_name in energy_names_gev:
            filepath = os.path.join(geant4_input_dir, f"outNuclei_{energy_name:.6f}.txt")
            if not os.path.exists(filepath): continue
            elif os.path.getsize(filepath) > 0:
                names = normalize_fragment_names(
                    np.loadtxt(filepath, usecols=0, dtype=str, ndmin=1)
                )

                for name in names:
                    if name:
                        all_fragments.add(name)
        return sorted(list(all_fragments))


    def _process_geant4_data(
        self,
        t_kyr_array,
        energy_bins_gev=None,
        total_simulated_particles=1e4,
        species='mu-',
    ):
        """Recoil spectra in the sample slab from cosmic-ray primaries.

        For each primary-energy bin, the flux is integrated over energy and zenith
        angle, weighted by the probability of an interaction in the slab
        [overburden, overburden + slab thickness] (along the slanted path), and
        multiplied by the recoils per primary of the mineral run at that energy.
        For mu-, captures use a single energy-independent capture spectrum (averaged
        over runs with E0 <= CAPTURE_SPECTRUM_MAX_E0_GEV) times the total stopping
        probability in the slab. All timesteps are processed at once; each file is
        read once.

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            energy_bins_gev (np.ndarray, optional): Primary energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            species (str, optional): 'mu-', 'mu+', 'neutron' or 'proton'. Defaults to 'mu-'.

        Returns:
            dict: 't_kyr' (T,), 'depth_mwe' (T,), 'Er_bins' (recoil energy edges
            [MeV]) and, for each fragment, dR/dEr [recoils per g per kyr per MeV],
            shape (T, n_recoil_bins).

        Raises:
            ValueError: If flux_history or the depth tables are not initialised.
        """
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, species)

        if self.flux_history is None:
            raise ValueError(
                "flux_history not initialized: use set_flux_history() or an integrate_* function."
            )
        if not self._depth_interpolators.get(species):
            raise ValueError(f"Depth interpolators not initialized for species {species}.")

        target_volume_cm3 = 1.e-3/self.config['density_g_cm3']

        target_thickness_cm = np.power(target_volume_cm3, 1/3)

        t_kyr_array = np.atleast_1d(np.asarray(t_kyr_array, dtype=float))
        n_t = t_kyr_array.shape[0]

        depth_mwe_array = self._overburden_interpolator(t_kyr_array)
        t_kyr_flux_array = self._flux_time_kyr(t_kyr_array)


        all_fragments = self._get_all_fragments(energy_bins_gev[:-1], species)
        geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, species)

        n_recoil_bins = len(RECOIL_ENERGY_BINS_MEV) - 1
        fragment_spectra = {frag: np.zeros((n_t, n_recoil_bins)) for frag in all_fragments}

        target_thickness_mwe = target_thickness_cm * 0.01 * self.config['density_g_cm3']

        if species == 'mu-':
            capture_counts = {frag: np.zeros(n_recoil_bins) for frag in all_fragments}
            per_file_peak = {frag: np.zeros((n_t, n_recoil_bins)) for frag in all_fragments}
            n_capture_files = 0
            weight_peak_total_t = np.zeros(n_t)

        for i in range(len(energy_bins_gev) - 1):
            e_min = energy_bins_gev[i]
            e_max = energy_bins_gev[i + 1]

            N_E, N_C = 50, 50
            e_vals = np.linspace(e_min, e_max, N_E)
            c_vals = np.linspace(0.01, 1.0, N_C)
            E_grid, C_grid = np.meshgrid(e_vals, c_vals)

            z_min_grid = depth_mwe_array[:, None, None] / C_grid[None, :, :]
            z_max_grid = z_min_grid + target_thickness_mwe / C_grid[None, :, :]

            flux_grid_gev_cm2_s_sr = self.flux_history.get_map(species, t_kyr_flux_array, E_grid)[2].reshape(t_kyr_flux_array.size, *E_grid.shape)

            if species in ('mu-', 'mu+'):
                prob_tail_grid, prob_peak_grid = self._muon_depth_probabilities(
                    e_vals, z_min_grid, z_max_grid
                )
                integrand_tail = flux_grid_gev_cm2_s_sr * prob_tail_grid

                weight_elastic_t = np.trapezoid(
                    np.trapezoid(integrand_tail, e_vals, axis=-1), c_vals, axis=-1
                )

                if species == 'mu-':
                    integrand_peak = flux_grid_gev_cm2_s_sr * prob_peak_grid
                    weight_peak_t = np.trapezoid(
                        np.trapezoid(integrand_peak, e_vals, axis=-1), c_vals, axis=-1
                    )
                    weight_peak_total_t += weight_peak_t
            else:
                prob_attenuation_grid = self._hadron_depth_probability(
                    e_vals, z_min_grid, z_max_grid, self._depth_interpolators[species]['table']
                )
                integrand = flux_grid_gev_cm2_s_sr * prob_attenuation_grid
                weight_elastic_t = np.trapezoid(
                    np.trapezoid(integrand, e_vals, axis=-1), c_vals, axis=-1
                )

            filepath = os.path.join(geant4_input_dir, f"outNuclei_{e_min:.6f}.txt")
            if not os.path.exists(filepath):
                continue
            try:
                df = pd.read_csv(
                    filepath, sep=r'\s+', header=None, usecols=[0, 2, 5],
                    names=['name', 'rec_e', 'rem_e'],
                    dtype={'name': str, 'rec_e': float, 'rem_e': float},
                )
                names = normalize_fragment_names(df['name'].values)
                rec_energies = df['rec_e'].values
                rem_energies = df['rem_e'].values
            except pd.errors.EmptyDataError:
                continue

            valid_mask = np.isin(names, list(fragment_spectra.keys()))
            valid_names = names[valid_mask]
            valid_rec = rec_energies[valid_mask]
            valid_rem = rem_energies[valid_mask]
            unique_frags = np.unique(valid_names)

            if species == 'mu-':
                use_for_capture = e_min <= CAPTURE_SPECTRUM_MAX_E0_GEV
                if use_for_capture:
                    n_capture_files += 1

            for frag in unique_frags:
                frag_mask = (valid_names == frag)
                frag_rec_energies = valid_rec[frag_mask]
                frag_rem_energies = valid_rem[frag_mask]

                if species == 'mu-':
                    mask_peak = (frag_rem_energies == 0.0)
                    rec_peak = frag_rec_energies[mask_peak]
                    rec_elastic = frag_rec_energies[~mask_peak]

                    if len(rec_peak) > 0:
                        counts_peak, _ = np.histogram(rec_peak, bins=RECOIL_ENERGY_BINS_MEV)
                        per_file_peak[frag] += np.outer(weight_peak_t, counts_peak)
                        if use_for_capture:
                            capture_counts[frag] += counts_peak

                    if len(rec_elastic) > 0:
                        counts_elastic, _ = np.histogram(rec_elastic, bins=RECOIL_ENERGY_BINS_MEV)
                        fragment_spectra[frag] += np.outer(weight_elastic_t, counts_elastic)
                else:
                    counts, _ = np.histogram(frag_rec_energies, bins=RECOIL_ENERGY_BINS_MEV)
                    fragment_spectra[frag] += np.outer(weight_elastic_t, counts)

        if species == 'mu-':
            if n_capture_files > 0:
                # mean capture counts per run (each run has total_simulated_particles muons)
                for frag in all_fragments:
                    fragment_spectra[frag] += np.outer(
                        weight_peak_total_t, capture_counts[frag] / n_capture_files
                    )
            else:
                if self.verbose > 0:
                    print(f"Warning: no mu- run with E0 <= {CAPTURE_SPECTRUM_MAX_E0_GEV} GeV; "
                          "using the per-energy capture spectra.")
                for frag in all_fragments:
                    fragment_spectra[frag] += per_file_peak[frag]

        bin_widths_mev = np.diff(RECOIL_ENERGY_BINS_MEV)
        norm_factor = (
            bin_widths_mev * target_thickness_cm * self.config['density_g_cm3']
            * total_simulated_particles * KYR_PER_SECOND
        )

        normalized_spectra = {}
        for name, spectrum in fragment_spectra.items():
            normalized_spectra[name] = np.divide(
                spectrum, norm_factor[None, :], out=np.zeros_like(spectrum),
                where=norm_factor[None, :] != 0,
            )

        return {
            't_kyr': t_kyr_array,
            'depth_mwe': depth_mwe_array,
            'Er_bins': RECOIL_ENERGY_BINS_MEV,
            **normalized_spectra,
        }


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


    def _process_secondary_geant4_data(
        self, t_kyr_array, product_bins_gev=None,
        total_simulated_particles=1e4, secondary_sources=SECONDARY_SOURCES, energy_windows=None,
    ):
        """Recoil spectra in the sample slab from the secondary neutrons and protons, all products at once.

        Products born in the window above the sample (see _secondary_yield, which follows all generations and
        cross-generations) travel along the primary's direction to the slab. For a product of type q born
        uniformly in window bin b, at distance d in [d0, d0 + delta] from the slab top, the probability of
        interacting in the slab (path tau = t_slab / c) is averaged over the bin exactly:

            P_b = [H(d0 + delta + tau) - H(d0 + tau) - H(d0 + delta) + H(d0)] / delta,

        with H(z) = integral_0^z F = z - G(z), F being the StdRock depth CDF of q. Then

            w_m(t) = int dc sum_b Y[t, c, b, m] P_b(E_m, c),

        and the recoils per product of the mineral run of q at the lower edge E_m of each cell are applied with
        the normalisation of _process_geant4_data.

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
            dict: {product: recoil data, same structure as _process_geant4_data}.
        """
        if product_bins_gev is None:
            product_bins_gev = self._secondary_grids()[0]
        energy_windows = dict(energy_windows or {})
        rho = self.config['density_g_cm3']

        target_volume_cm3 = 1.e-3 / rho
        target_thickness_cm = np.power(target_volume_cm3, 1/3)
        target_thickness_mwe = target_thickness_cm * 0.01 * rho

        t_kyr_array = np.atleast_1d(np.asarray(t_kyr_array, dtype=float))
        n_t = len(t_kyr_array)
        depth_mwe_array = np.atleast_1d(self._overburden_interpolator(t_kyr_array))

        c_vals = np.linspace(0.01, 1.0, 50)
        Y, delta, offsets = self._secondary_yield(
            t_kyr_array, depth_mwe_array, c_vals, product_bins_gev,
            total_simulated_particles, secondary_sources,
        )
        n_b = Y.shape[2]
        d0 = (n_b - 1 - np.arange(n_b)) * delta                       # distance from bin bottom to slab
        tau = target_thickness_mwe / c_vals                              # slab path
        z_d0 = np.broadcast_to(d0[None, :], (len(c_vals), n_b))
        z_tau = np.broadcast_to(tau[:, None], (len(c_vals), n_b))

        bin_widths_mev = np.diff(RECOIL_ENERGY_BINS_MEV)
        norm_factor = (
            bin_widths_mev * target_thickness_cm * rho
            * total_simulated_particles * KYR_PER_SECOND
        )
        n_recoil_bins = len(RECOIL_ENERGY_BINS_MEV) - 1

        out = {}
        for product, energy_bins_gev in product_bins_gev.items():
            n_e = len(energy_bins_gev) - 1
            e_cells = energy_bins_gev[:-1]
            all_fragments = self._get_all_fragments(e_cells, species=product)
            geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, product)

            # bin-averaged slab probability, shape (N_C, n_b, n_e)
            table = self._depth_tables[product]
            q_rows = self._depth_quantiles(e_cells, product)
            tails = self._hadron_tail_lambdas(e_cells, product)
            prob = np.empty((len(c_vals), n_b, n_e))
            for m in range(n_e):
                def H(z):
                    """Integral of the product's depth CDF from 0 to z [m.w.e.]."""
                    _, G = _piecewise_cdf_and_integral(z, q_rows[m], table['p'], tails[m])
                    return z - G
                prob[:, :, m] = (H(z_d0 + delta + z_tau) - H(z_d0 + z_tau)
                                 - H(z_d0 + delta) + H(z_d0)) / delta

            weight_t = np.trapezoid(np.einsum('tcbm,cbm->tcm', Y[..., offsets[product]], prob), c_vals, axis=1)
            if product in energy_windows:
                lo, hi = energy_windows[product]
                tol = 1 + _ENERGY_EDGE_RTOL
                outside = (energy_bins_gev[:-1] * tol < lo) | (energy_bins_gev[1:] > hi * tol)
                weight_t[:, outside] = 0.0

            fragment_spectra = {frag: np.zeros((n_t, n_recoil_bins)) for frag in all_fragments}
            for m in range(n_e):
                filepath = os.path.join(geant4_input_dir, f"outNuclei_{e_cells[m]:.6f}.txt")
                if not os.path.exists(filepath):
                    continue
                try:
                    df = pd.read_csv(
                        filepath, sep=r'\s+', header=None, usecols=[0, 2],
                        names=['name', 'rec_e'], dtype={'name': str, 'rec_e': float},
                    )
                    names = normalize_fragment_names(df['name'].values)
                    rec_energies = df['rec_e'].values
                except pd.errors.EmptyDataError:
                    continue

                valid_mask = np.isin(names, list(fragment_spectra.keys()))
                valid_names = names[valid_mask]
                valid_rec = rec_energies[valid_mask]

                for frag in np.unique(valid_names):
                    frag_mask = (valid_names == frag)
                    counts, _ = np.histogram(valid_rec[frag_mask], bins=RECOIL_ENERGY_BINS_MEV)
                    fragment_spectra[frag] += np.outer(weight_t[:, m], counts)

            normalized_spectra = {}
            for name, spectrum in fragment_spectra.items():
                normalized_spectra[name] = np.divide(
                    spectrum, norm_factor[None, :], out=np.zeros_like(spectrum),
                    where=norm_factor[None, :] != 0,
                )
            out[product] = {
                't_kyr': t_kyr_array,
                'depth_mwe': depth_mwe_array,
                'Er_bins': RECOIL_ENERGY_BINS_MEV,
                **normalized_spectra,
            }
        return out


    def _convert_recoil_to_track_spectrum(self, recoil_data, x_bins=None, energy_bins_gev=None, species='mu-',
                                          fragments=None):
        """Converts recoil-energy spectra into track-length spectra.

        For each fragment, dR/dx = dR/dEr(E(x)) * dE/dx(x), where E(x) inverts the
        CSDA range of the mineral's dE/dx table and dE/dx is the total (electronic +
        nuclear) stopping power. Neutrons are skipped.

        Args:
            recoil_data (dict): Output of one of the _process_* methods; fragment
                entries have shape (n_recoil_bins,) or (T, n_recoil_bins).
            x_bins (np.ndarray, optional): Track-length bin edges [nm]. Defaults to
                None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Energy bin edges [GeV], used to
                list the fragments. Defaults to None (all the Geant4 files of the species).
            species (str, optional): Species of the Geant4 runs ('mu-', 'mu+',
                'neutron', 'proton', 'secondary_neutron' or 'secondary_proton'). Defaults to 'mu-'.
            fragments (list[str], optional): Fragments to convert. Defaults to None (those found in the Geant4 files
                of the species; energy_bins_gev and species are then used only for this).

        Returns:
            dict: dR/dx per fragment and 'total' [tracks per mg per kyr per nm],
            shape (T, n_x), or (n_x,) if the input was 1-D.
        """
        x_bins = _resolve_x_bins(x_bins)

        er_bins = recoil_data['Er_bins']
        er_mid_mev = er_bins[:-1] + np.diff(er_bins) / 2.0

        if fragments is None:
            energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, species)
            tab_species = DERIVED_SPECIES.get(species, species)
            all_fragments = self._get_all_fragments(energy_bins_gev[:-1], tab_species)
        else:
            all_fragments = sorted(fragments)

        x_mid_nm = x_bins[:-1] + np.diff(x_bins) / 2.0

        sample_frag = next((f for f in all_fragments if f in recoil_data and f != 'neutron'), None)

        input_was_1d = sample_frag is not None and np.asarray(recoil_data[sample_frag]).ndim == 1

        n_t = recoil_data[sample_frag].shape[0] if sample_frag is not None and recoil_data[sample_frag].ndim == 2 else 1

        dRdx_by_nucleus = {}
        dRdx_total = np.zeros((n_t, len(x_bins) - 1))
        dRdx_heavy = np.zeros((n_t, len(x_bins) - 1))

        for nuclide_name in all_fragments:
            if nuclide_name not in recoil_data or nuclide_name == 'neutron':
                continue

            dRdEr_mev = np.atleast_2d(recoil_data[nuclide_name])  # (T, n_recoil_bins)

            dRdEr_interp = interp1d(
                er_mid_mev, dRdEr_mev, axis=-1, bounds_error=False, fill_value=0.0
            )

            corr_nuclide_name = _LIGHT_ION_SYMBOLS.get(nuclide_name, nuclide_name)
            nucleus_name = ''.join(filter(str.isalpha, corr_nuclide_name))
            ion_z = element(nucleus_name).atomic_number
            ion_a = int(''.join(filter(str.isdigit, corr_nuclide_name)))

            geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, "Range")

            filepath = os.path.join(geant4_input_dir, f"DEDX_Z{ion_z}_A{ion_a}.txt")

            e_MeV, dee_dx_mev_um, den_dx_mev_um, length_um = np.loadtxt(filepath, skiprows=1, unpack=True)

            sorted_indices = np.argsort(length_um)
            x_to_e_func = interp1d(length_um[sorted_indices] * 1e3, e_MeV[sorted_indices],
                                    bounds_error=False, fill_value=0.0)
            x_to_dedx_func = interp1d(length_um[sorted_indices] * 1e3,
                                    (dee_dx_mev_um[sorted_indices] + den_dx_mev_um[sorted_indices]) * 1e-3,
                                    bounds_error=False, fill_value=0.0)

            e_at_x = x_to_e_func(x_mid_nm)

            dRdx_nucleus = dRdEr_interp(e_at_x) * x_to_dedx_func(x_mid_nm)[None, :]

            dRdx_nucleus_mg = dRdx_nucleus * 1e-3

            dRdx_by_nucleus[nuclide_name] = dRdx_nucleus_mg
            dRdx_total += dRdx_nucleus_mg
            if not is_light_ion(nuclide_name):
                dRdx_heavy += dRdx_nucleus_mg

        dRdx_by_nucleus[TOTAL_ALL_FRAGMENTS] = dRdx_total
        dRdx_by_nucleus[TOTAL_NO_LIGHT_IONS] = dRdx_heavy

        if input_was_1d:
            dRdx_by_nucleus = {name: arr[0] for name, arr in dRdx_by_nucleus.items()}
 
        return dRdx_by_nucleus


    @staticmethod
    def _select_nucleus(dRdx_by_fragment, nucleus, label):
        """Picks the requested track sum or fragment from the output of _convert_recoil_to_track_spectrum.

        Shared by every integration path (primaries, atmospheric secondaries, neutrinos).

        Args:
            dRdx_by_fragment (dict): Output of _convert_recoil_to_track_spectrum.
            nucleus (str): 'total_no_light_ions', 'total', 'all' (the whole dict) or a fragment key.
            label (str): Species name, for the error message.

        Returns:
            np.ndarray or dict: The selected rate, or the dict for 'all'.

        Raises:
            ValueError: If nucleus is not in the dict.
        """
        if nucleus == "all":
            return dRdx_by_fragment
        if nucleus not in dRdx_by_fragment:
            raise ValueError(f"nucleus '{nucleus}' not found: use one of {TRACK_SUMS}, 'all' or a fragment of "
                             f"{label} ({sorted(k for k in dRdx_by_fragment if k not in TRACK_SUMS)}).")
        return dRdx_by_fragment[nucleus]


    def calculate_particle_signal_spectrum(
        self,  t_kyr_array, x_bins=None,
        energy_bins_gev=None, total_simulated_particles=1e4, 
        species='mu-', nucleus=DEFAULT_TRACK_SUM,
    ):
        
        """Track-length rate from one cosmic-ray species at given times.

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            x_bins (np.ndarray, optional): Track-length bin edges [nm]. Defaults to
                None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Primary energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            species (str, optional): 'mu-', 'mu+', 'neutron', 'proton',
                'secondary_neutron' or 'secondary_proton'. Defaults to 'mu-'. For a secondary species the
                whole neutron + proton cascade is computed (cross-generations need both); energy_bins_gev then
                selects the cells of its own product that are reported (see _secondary_grids).
            nucleus (str, optional): 'total_no_light_ions' (all fragments except H and He, see
                LIGHT_ION_MAX_Z), 'total' (all fragments), 'all' (dict of every fragment and both sums) or a
                fragment key. Defaults to 'total_no_light_ions'.

        Returns:
            np.ndarray or dict: dR/dx [tracks per mg per kyr per nm], shape (T, n_x).
        """
        x_bins = _resolve_x_bins(x_bins)

        if species in DERIVED_SPECIES:
            product = DERIVED_SPECIES[species]
            bins, windows = self._secondary_grids({product: energy_bins_gev})
            recoil_data = self._process_secondary_geant4_data(
                t_kyr_array, bins, total_simulated_particles, energy_windows=windows,
            )[product]
            energy_bins_gev = bins[product]
        else:
            energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, species)
            recoil_data = self._process_geant4_data(
                t_kyr_array, energy_bins_gev, total_simulated_particles, species,
            )

        dRdx_at_depth = self._convert_recoil_to_track_spectrum(recoil_data, x_bins, energy_bins_gev, species)
        return self._select_nucleus(dRdx_at_depth, nucleus, species)


    def integrate_particle_signal_spectrum(
        self, x_bins=None, energy_bins_gev=None,
        exposure_window_kyr=None, total_simulated_particles=1e4, 
        x_grid=TRACK_LENGTH_BINS_NM, species='mu-', nucleus=DEFAULT_TRACK_SUM, e_range=None,
        annealing=None, annealing_r_min=0.0,
    ):
        """Track counts from one cosmic-ray species over an exposure window.

        The rate is computed on a time grid and integrated with the trapezoidal
        rule; the fine-grid spectrum is then rebinned to x_bins through its
        cumulative distribution (_integrate_rate_over_window).

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Primary energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            exposure_window_kyr (float or list[float], optional): Window in local
                time [kyr]; a single value w means [0, w]. Defaults to None
                ([0, total_age_kyr]).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            x_grid (np.ndarray, optional): Fine track-length grid [nm]. Defaults to
                TRACK_LENGTH_BINS_NM.
            species (str, optional): 'mu-', 'mu+', 'neutron', 'proton',
                'secondary_neutron' or 'secondary_proton'. Defaults to 'mu-'.
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key (see
                calculate_particle_signal_spectrum). Defaults to 'total_no_light_ions'.
            e_range (tuple[float, float], optional): (E_min, E_max) [GeV]: use only the Geant4 files whose bins
                lie inside it (see _energy_edges). Ignored if energy_bins_gev is given. Defaults to None (all).
            annealing (float or dict, optional): Reduced track length r = L_today / L_formed of a track formed
                at time t (0 < r <= 1): every track formed at t is shortened to r(t) times its length
                (proportional shortening, as in fission-track annealing models). A number (same r for every
                formation time), a table {'law': 'table', 't_kyr': [...], 'r': [...]} (r at those formation
                times, linear in between, constant outside), or a law with a geotherm, e.g.
                {'law': 'fanning_curvilinear', 'C0': ..., 'C1': ..., 'C2': ..., 'C3': ..., 'alpha': ...,
                'beta': ..., 'time_unit_s': ..., 'T_surface_C': 15, 'gradient_C_per_km': 25,
                'rock_density_g_cm3': 2.16} or {'law': 'first_order', 'A_per_kyr': ..., 'Ea_eV': ...}: then each
                track is followed from its formation to the present through the temperature given by the
                overburden history (see _annealing_law and _annealing_from_law). Defaults to None (no shortening).
            annealing_r_min (float, optional): Tracks with r below this value are removed (segmented, no longer
                etchable; ~0.4-0.55 for apatite). Defaults to 0 (none removed).

        Returns:
            np.ndarray: Tracks per mg in each bin of x_bins (after annealing, if given).
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = (self._energy_edges(species, e_range) if energy_bins_gev is None
                           else np.asarray(energy_bins_gev, dtype=float))

        if species in DERIVED_SPECIES:
            return self._integrate_secondary_species(
                [species], {species: energy_bins_gev}, x_bins=x_bins, exposure_window_kyr=exposure_window_kyr,
                total_simulated_particles=total_simulated_particles, x_grid=x_grid, nucleus=nucleus,
                annealing=annealing, annealing_r_min=annealing_r_min,
            )[species]

        t_kyr_array = self._window_grid(self._exposure_window(exposure_window_kyr))

        dRdx_array = self.calculate_particle_signal_spectrum(
            t_kyr_array, x_bins=x_grid, energy_bins_gev=energy_bins_gev,
            total_simulated_particles=total_simulated_particles,
            species=species, nucleus=nucleus
        )
        return self._integrate_rate_over_window(dRdx_array, t_kyr_array, x_bins, x_grid, annealing, annealing_r_min)


    def _integrate_secondary_species(
        self, secondary_species, energy_bins_gev=None, x_bins=None, exposure_window_kyr=None,
        total_simulated_particles=1e4, x_grid=TRACK_LENGTH_BINS_NM, nucleus=DEFAULT_TRACK_SUM,
        annealing=None, annealing_r_min=0.0,
    ):
        """Track counts of several atmospheric secondary species from one solve of the cascade.

        The neutron + proton cascade (_secondary_yield) is computed once, with every followable product on its
        full grid, and each requested secondary species takes its own product from it (restricted to its energy
        range, if given, without changing the cascade: see _secondary_grids).

        Args:
            secondary_species (list[str]): Keys of DERIVED_SPECIES ('secondary_neutron', 'secondary_proton').
            energy_bins_gev (dict, optional): {secondary species: energy cell edges [GeV]} of its product.
                Defaults to None (full grids).
            x_bins, exposure_window_kyr, total_simulated_particles, x_grid, nucleus, annealing, annealing_r_min:
                See integrate_particle_signal_spectrum.

        Returns:
            dict: {secondary species: tracks per mg in each bin of x_bins}.
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = dict(energy_bins_gev or {})
        bins, windows = self._secondary_grids({DERIVED_SPECIES[s]: energy_bins_gev.get(s) for s in secondary_species})
        t_kyr_array = self._window_grid(self._exposure_window(exposure_window_kyr))
        recoils = self._process_secondary_geant4_data(t_kyr_array, bins, total_simulated_particles,
                                                      energy_windows=windows)
        out = {}
        for s in secondary_species:
            product = DERIVED_SPECIES[s]
            dRdx = self._select_nucleus(
                self._convert_recoil_to_track_spectrum(recoils[product], x_grid, bins[product], s), nucleus, s
            )
            out[s] = self._integrate_rate_over_window(dRdx, t_kyr_array, x_bins, x_grid, annealing, annealing_r_min)
        return out


    def _integrate_rate_over_window(self, dRdx_array, t_kyr_array, x_bins, x_grid, annealing=None,
                                    annealing_r_min=0.0):
        """Tracks accumulated over a window from a time-dependent rate, rebinned to the output bins.

        Shared by integrate_particle_signal_spectrum and integrate_atmospheric_neutrinos. The tracks formed at
        each time are shortened by annealing (if given), the rate is integrated over time with the trapezoidal
        rule, and the fine-grid spectrum is rebinned to x_bins through its cumulative distribution.

        Args:
            dRdx_array (np.ndarray): Rate [tracks per mg per kyr per nm] on x_grid, shape (T, len(x_grid) - 1).
            t_kyr_array (np.ndarray): Times of the rows [kyr], shape (T,).
            x_bins (np.ndarray): Output track-length bin edges [nm].
            x_grid (np.ndarray): Fine track-length grid [nm].
            annealing, annealing_r_min: See integrate_particle_signal_spectrum.

        Returns:
            np.ndarray: Tracks per mg in each bin of x_bins.
        """
        if annealing is not None:
            r = self._annealing_ratio(t_kyr_array, annealing)
            dRdx_array = _shorten_tracks(dRdx_array, x_grid, r, annealing_r_min)

        total_drdx_g = np.trapezoid(dRdx_array, t_kyr_array, axis=0)

        internal_bin_widths = np.diff(x_grid)
        cumulative_counts = np.concatenate(([0], np.cumsum(total_drdx_g * internal_bin_widths)))
        cdf_interp = interp1d(x_grid, cumulative_counts, kind='linear', bounds_error=False,
                            fill_value=(0, cumulative_counts[-1]))
        return cdf_interp(x_bins[1:]) - cdf_interp(x_bins[:-1])


    def integrate_all_particles(
        self,  x_bins=None, e_range=None,
        exposure_window_kyr=None, 
        total_simulated_particles=1e4, species_list=None, nucleus=DEFAULT_TRACK_SUM,
        annealing=None, annealing_r_min=0.0, flux_history=None
    ):
        """Track counts from several cosmic-ray species, computed in parallel.

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
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key (see
                calculate_particle_signal_spectrum). Defaults to 'total_no_light_ions'.
            annealing (float or dict, optional): Reduced track length r = L_today / L_formed of a track formed
                at time t (0 < r <= 1): every track formed at t is shortened to r(t) times its length
                (proportional shortening, as in fission-track annealing models). A number (same r for every
                formation time), a table {'law': 'table', 't_kyr': [...], 'r': [...]} (r at those formation
                times, linear in between, constant outside), or a law with a geotherm, e.g.
                {'law': 'fanning_curvilinear', 'C0': ..., 'C1': ..., 'C2': ..., 'C3': ..., 'alpha': ...,
                'beta': ..., 'time_unit_s': ..., 'T_surface_C': 15, 'gradient_C_per_km': 25,
                'rock_density_g_cm3': 2.16} or {'law': 'first_order', 'A_per_kyr': ..., 'Ea_eV': ...}: then each
                track is followed from its formation to the present through the temperature given by the
                overburden history (see _annealing_law and _annealing_from_law). Defaults to None (no shortening).
            annealing_r_min (float, optional): Tracks with r below this value are removed (segmented, no longer
                etchable; ~0.4-0.55 for apatite). Defaults to 0 (none removed).
            flux_history (FluxHistory, optional): Flux history to set for the integration. Defaults to None

        Returns:
            tuple[np.ndarray, dict]: x_bins, and tracks per mg in each bin for each
            species plus 'total'.

        Raises:
            ValueError: If a species has no runs, or e_range names a species not computed or an invalid range.
        """
        self.set_flux_history(flux_history)

        x_bins = _resolve_x_bins(x_bins)
        species_list = self.available_species() if species_list is None else list(species_list)
        e_range = dict(e_range or {})
        unknown = set(e_range) - set(species_list)
        if unknown:
            raise ValueError(f"e_range given for species not computed: {sorted(unknown)} "
                             f"(computed: {species_list}).")
        edges = {s: self._energy_edges(s, e_range.get(s)) for s in species_list}

        # caches shared by every species, built before the workers start so that they inherit them
        if isinstance(annealing, dict):
            self._annealing_ratio(self._master_time_grid(), annealing)

        shared_kwargs = dict(
            x_bins=x_bins,
            exposure_window_kyr=exposure_window_kyr,
            total_simulated_particles=total_simulated_particles, nucleus=nucleus,
            annealing=annealing, annealing_r_min=annealing_r_min,
        )

        # one task per primary species; the secondary species share one task (one solve of the cascade)
        primaries = [s for s in species_list if s not in DERIVED_SPECIES]
        secondaries = [s for s in species_list if s in DERIVED_SPECIES]
        tasks = [(self, dict(shared_kwargs, species=s, energy_bins_gev=edges[s])) for s in primaries]
        if secondaries:
            tasks.append((self, dict(shared_kwargs, secondary_species=secondaries,
                                     energy_bins_gev={s: edges[s] for s in secondaries})))

        with Pool(processes=min(len(tasks), os.cpu_count() or 1)) as pool:
            results = list(tqdm(pool.imap(_species_worker, tasks), total=len(tasks)))

        computed = dict(pair for result in results for pair in result)
        total_tracks_by_species = {s: computed[s] for s in species_list}
        total_tracks_by_species['total'] = sum(total_tracks_by_species.values())

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


    def _flux_cell_integral(self, species, t_kyr_array, edges_gev, n_energy_sub=50):
        """Flux of a species integrated over each energy cell, at local times.

        Purpose: the Geant4 run at the lower edge of a cell stands for every primary in the cell, so each run is
        weighted by the flux integrated over its cell (trapezoid on n_energy_sub linear points per cell).

        Args:
            species (str): Flux species (MCEQ_SPECIES key).
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            edges_gev (np.ndarray): Cell edges [GeV], shape (M,).
            n_energy_sub (int, optional): Points per cell. Defaults to 50.

        Returns:
            np.ndarray: Integrated flux [cm^-2 s^-1] (crossing a horizontal plane, see NEUTRINO_SOLID_ANGLE_FACTOR),
            shape (T, M - 1).
        """
        t_flux = self._flux_time_kyr(np.atleast_1d(np.asarray(t_kyr_array, dtype=float)))
        edges_gev = np.asarray(edges_gev, dtype=float)
        e_lo, e_hi = edges_gev[:-1], edges_gev[1:]
        e_sub = e_lo[:, None] + (e_hi - e_lo)[:, None] * np.linspace(0.0, 1.0, n_energy_sub)[None, :]
        phi = self.flux_history.get_map(species, t_flux, e_sub)[2].reshape(t_flux.size, *e_sub.shape)
        step = (e_hi - e_lo) / (n_energy_sub - 1)
        return (phi.sum(axis=-1) - 0.5 * (phi[..., 0] + phi[..., -1])) * step[None, :]


    def _read_recoils_and_products(self, filepath, product_edges):
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
        offsets, k = {}, 0
        for p, edges in product_edges.items():
            offsets[p] = slice(k, k + len(edges) - 1)
            k += len(edges) - 1
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
                    if self.verbose > 0:
                        print(f"Warning: no StdRock/{p} runs: the products of neutrino-made {p}s are not followed.")
                    continue
                for q, q_edges in product_edges.items():
                    rows = self._load_stdrock_rows(p, q, q_edges, total_simulated_particles, primary_bins_gev=edges)
                    n_p, n_q = len(edges) - 1, len(q_edges) - 1
                    G[offsets[p], offsets[q]] = np.bincount(rows['primary_cell'] * n_q + rows['cell'],
                                                            minlength=n_p * n_q).reshape(n_p, n_q) * rows['weight']
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
        n_k = sum(len(e) - 1 for e in product_edges.values())
        length_cm = self._geant4_geometry(run_dir)["target_length_m"] * 100.0
        phi_cell = self._flux_cell_integral(species, t_kyr, edges)          # (T, n_cells)
        if self.verbose > 0 and not np.any(phi_cell > 0):
            print(f"Warning: the flux history has no {species} flux (templates built without neutrinos?): "
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
            col = 0
            for p, p_edges in product_edges.items():
                n_p = len(p_edges) - 1
                out['S0'][:, col:col + n_p] += np.outer(w, run['products'][p])
                out['lost'][p][0] += float(np.mean(w)) * run['below'][p]
                out['lost'][p][1] += float(np.mean(w)) * run['above'][p]
                col += n_p
            biases.append(bias)
            fractions.append(run['n_vertices'] / total_simulated_particles)
        out['report'] = (len(edges) - 1, min(biases), max(biases), max(fractions))
        return out


    def integrate_atmospheric_neutrinos(
        self, x_bins=None, e_range=None, exposure_window_kyr=None,
        total_simulated_particles=1e4, neutrino_bias=None, species_list=None,
        secondaries=True, product_simulated_particles=None, nucleus=DEFAULT_TRACK_SUM,
        annealing=None, annealing_r_min=0.0, x_grid=TRACK_LENGTH_BINS_NM,
        solid_angle_factor=NEUTRINO_SOLID_ANGLE_FACTOR, flux_history=None,
        secondary_medium='auto', stdrock_neutrino_bias=None,
    ):
        """Track counts from atmospheric neutrinos: their own interactions in the sample and those of the
        particles they produce in the surrounding rock.

        Differences from integrate_all_particles:

        * No overburden. Neutrinos are not attenuated by any realistic overburden, so the depth tables and the
          overburden history play no role; the rate changes in time only through the flux history (and annealing).
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
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key. Defaults to
                'total_no_light_ions'.
            annealing, annealing_r_min: See integrate_particle_signal_spectrum.
            x_grid (np.ndarray, optional): Fine track-length grid [nm]. Defaults to TRACK_LENGTH_BINS_NM.
            solid_angle_factor (float, optional): Rate per unit volume / template flux. Defaults to
                NEUTRINO_SOLID_ANGLE_FACTOR (4: isotropic, from all directions).
            flux_history (FluxHistory, optional): If given, set with set_flux_history before the integration;
                None keeps the current one. Defaults to None.
            secondary_medium (str, optional): Rock in which the secondaries are made: 'auto', 'StdRock' or
                'mineral' (see above). Defaults to 'auto'.
            stdrock_neutrino_bias (float, dict, callable or None, optional): Bias of the StdRock neutrino runs, same
                forms as neutrino_bias. Defaults to None (from their geometry.json).

        Returns:
            tuple[np.ndarray, dict]: x_bins, and tracks per mg in each bin for each neutrino species, each
            'nu_secondary_<product>' (summed over the neutrino species) and 'total'.

        Raises:
            ValueError: If a species has no runs, e_range is invalid, a run has no bias, or secondary_medium is
                'StdRock' without StdRock neutrino runs.
        """
        if flux_history is not None:
            self.set_flux_history(flux_history)
        x_bins = _resolve_x_bins(x_bins)
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
        n_t, n_r = t_kyr.size, len(RECOIL_ENERGY_BINS_MEV) - 1

        products = [p for p in NEUTRINO_PRODUCTS if p in self.energy_bins] if secondaries else []
        if secondaries and self.verbose > 0 and len(products) < len(NEUTRINO_PRODUCTS):
            print(f"Warning: no mineral runs for {sorted(set(NEUTRINO_PRODUCTS) - set(products))}: "
                  "the neutrino products of that type are not followed.")
        product_edges = {p: self.energy_bins[p] for p in products}
        n_cells_total = sum(len(e) - 1 for e in product_edges.values())

        # --- neutrino runs in the mineral: recoils of the neutrinos (and first products, medium 'mineral') ---
        stdrock_nu = {sp: os.path.join(self.data_path, "Geant4_data", "StdRock", sp) for sp in species_list}
        has_stdrock = {sp: _geant4_energy_edges(d) is not None for sp, d in stdrock_nu.items()}
        if secondary_medium == 'auto':
            secondary_medium = 'StdRock' if all(has_stdrock.values()) else 'mineral'
            if products and self.verbose > 0 and secondary_medium == 'mineral':
                missing = [sp for sp, ok in has_stdrock.items() if not ok]
                print(f"Neutrino secondaries made in the mineral (no StdRock runs for {missing}).")
        if secondary_medium not in ('StdRock', 'mineral'):
            raise ValueError(f"secondary_medium must be 'auto', 'StdRock' or 'mineral', got {secondary_medium!r}.")
        if products and secondary_medium == 'StdRock' and not all(has_stdrock.values()):
            raise ValueError(f"No StdRock neutrino runs for {[sp for sp, ok in has_stdrock.items() if not ok]} in "
                             f"{os.path.join(self.data_path, 'Geant4_data', 'StdRock')}.")

        recoils = {}                                            # species -> {fragment: (T, n_r)} [per g per s]
        S0 = np.zeros((n_t, n_cells_total))                     # first products [per g of rock per s]
        lost = {p: [0.0, 0.0] for p in products}                # products outside the cells [per g per s]
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

        # --- neutrino runs in StdRock: first products (medium 'StdRock') ---
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

        # --- products and their generations ---
        if products:
            offsets, R, G, outside = self._neutrino_product_operator(product_edges, n_prod_sim, secondary_medium)
            radius = np.max(np.abs(np.linalg.eigvals(G))) if G.size else 0.0
            if radius < 1.0:
                S = np.linalg.solve((np.eye(n_cells_total) - G).T, S0.T).T       # S0 (I - G)^-1
            else:
                if self.verbose > 0:
                    print(f"Warning: product multiplication per generation >= 1 (spectral radius {radius:.3g}); "
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

        # --- recoil rates -> tracks, integrated over the window ---
        to_per_kyr_mev = 1.0 / (np.diff(RECOIL_ENERGY_BINS_MEV)[None, :] * KYR_PER_SECOND)
        if isinstance(annealing, dict):
            self._annealing_ratio(self._master_time_grid(), annealing)
        results = {}
        for key, spec in recoils.items():
            recoil_data = {'Er_bins': RECOIL_ENERGY_BINS_MEV, 't_kyr': t_kyr,
                           **{frag: arr * to_per_kyr_mev for frag, arr in spec.items()}}
            dRdx = self._select_nucleus(
                self._convert_recoil_to_track_spectrum(recoil_data, x_grid, fragments=list(spec)), nucleus, key
            )
            results[key] = self._integrate_rate_over_window(dRdx, t_kyr, x_bins, x_grid, annealing, annealing_r_min)

        results['total'] = sum(results.values())
        return x_bins, results


def _species_worker(self_and_args):
    """Pool worker of integrate_all_particles: computes the track counts of one task.

    Args:
        self_and_args (tuple): (Paleodetector instance, keyword arguments of integrate_particle_signal_spectrum
            with either 'species' (one primary species) or 'secondary_species' (list of secondary species, solved
            together by _integrate_secondary_species, energy_bins_gev then being a dict per species)).

    Returns:
        list[tuple[str, np.ndarray]]: (species, tracks per mg in each bin) for each species of the task.
    """
    self, kwargs = self_and_args
    kwargs = dict(kwargs)
    if 'secondary_species' in kwargs:
        secondary = kwargs.pop('secondary_species')
        if self.verbose > 1:
            print(f"Processing secondary species: {secondary}")
        return list(self._integrate_secondary_species(secondary, **kwargs).items())
    species = kwargs.pop('species')
    if self.verbose > 1:
        print(f"Processing species: {species}")
    return [(species, self.integrate_particle_signal_spectrum(species=species, **kwargs))]
