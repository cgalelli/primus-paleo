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
    """True for hydrogen and helium fragments (proton, deuteron, triton, alpha, He3, ...).

    Args:
        fragment (str): Fragment key (see normalize_fragment_name).

    Returns:
        bool: Whether the fragment has Z <= LIGHT_ION_MAX_Z.
    """
    symbol = _LIGHT_ION_SYMBOLS.get(fragment, fragment)
    return ''.join(filter(str.isalpha, symbol)) in _LIGHT_ION_ELEMENTS


def normalize_fragment_name(name):
    """Maps a Geant4 particle name to the fragment key used in the analysis.

    Excited ions are merged into their ground state; named light ions and the
    neutron are kept; every other particle is discarded.

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
    """Vectorised version of normalize_fragment_name.

    Each distinct name is mapped once, so the cost does not grow with the number
    of rows. Discarded particles become "", which never matches a fragment key and
    is therefore dropped downstream.

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
# listed at creation; only these (and 'secondary_neutron', from the neutron runs) can be computed for now.
COMPUTABLE_SPECIES = ('mu-', 'mu+', 'neutron')
# Species without runs of their own: secondary_neutron takes its recoils (and energy grid) from the mineral's neutron
# runs; the neutrons themselves are produced, in the window above the sample, by SECONDARY_NEUTRON_SOURCES (StdRock).
DERIVED_SPECIES = {'secondary_neutron': 'neutron'}
SECONDARY_NEUTRON_SOURCES = ('mu-', 'mu+', 'neutron')
IRREGULAR_STEP_FACTOR = 1.5   # a step this many times the median step in log(E) is reported as a gap

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

# Time grid of the exposure integrals: on top of the uniform grid, the times at which the overburden reaches
# DEPTH_GRID_POINTS values equally spaced in log(depth), so that fast burial is resolved where the rates change
# (atmospheric neutrons fade over ~1.6 m.w.e., muons over tens of m.w.e.).
DEPTH_GRID_POINTS = 200
DEPTH_GRID_MIN_MWE = 0.01

_G4_OUTPUT_FILE_RE = re.compile(r"^outNuclei_(\d+\.\d+)\.txt$")


def _list_geant4_energy_files(data_dir):
    """Lists the Geant4 output files of a run directory.

    Args:
        data_dir (str): Directory containing outNuclei_<E>.txt files.

    Returns:
        list[tuple[float, str]]: (primary energy [GeV], file path), sorted by
        energy. Empty if the directory does not exist.
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
    """Interpolates tabulated rows linearly in log-log space.

    Each column of fp_rows is interpolated independently. Values outside the
    table are extrapolated linearly in log-log from the end segments.

    Args:
        x (array_like): Points to evaluate, shape (n,). Must be > 0.
        xp (array_like): Tabulated abscissae, increasing, shape (m,). Must be > 0.
        fp_rows (array_like): Tabulated values, shape (m, k) or (m,). Must be > 0.

    Returns:
        np.ndarray: Interpolated values, shape (n, k).
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
    """CDF and its complement integral for a quantile-node depth distribution.

    The distribution of a depth R is described by quantile nodes q at
    probabilities p (p[0] = 0), with F piecewise linear between nodes. If p ends
    at 1, F = 1 beyond the last node. If p ends below 1, an exponential tail is
    attached: F(z) = 1 - (1 - p[-1]) * exp(-(z - q[-1]) / tail_lambda).

    Args:
        z (array_like): Depths [m.w.e.], any shape.
        q (np.ndarray): Quantile nodes [m.w.e.], non-decreasing, shape (n,).
        p (np.ndarray): Probabilities of the nodes, shape (n,).
        tail_lambda (float, optional): Length of the exponential tail [m.w.e.].
            Required when p[-1] < 1. Defaults to None (no tail).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            F: P(R <= z), same shape as z.
            G: integral from 0 to z of (1 - F(t)) dt = E[min(R, z)], same shape as z.
            G(inf) is the mean E[R].
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
    """Probability that the depth falls inside a slab, for a set of energies.

    Args:
        q_rows (np.ndarray): Quantile nodes per energy, shape (n_E, n_nodes).
        p (np.ndarray): Probabilities of the nodes, shape (n_nodes,).
        z_min_grid (np.ndarray): Slab top [m.w.e.], shape (..., n_E); the last axis
            runs over the energies of q_rows.
        z_max_grid (np.ndarray): Slab bottom [m.w.e.], same shape as z_min_grid.
        tail_lambdas (np.ndarray, optional): Exponential-tail length per energy
            [m.w.e.], shape (n_E,). Defaults to None (no tail).

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
    """Counts in out_edges from a density given per bin of a fine grid (constant inside each fine bin).

    The cumulative count is exact at the fine-grid edges and linear between them, so the total is conserved and
    no numerical quadrature is needed; counts outside the fine grid are zero.

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


def set_integration_x_bins_nm(integration_x_bins_nm):
    """Sets the module-wide default track-length binning.

    Functions whose x_bins argument is None use this binning at call time.

    Args:
        integration_x_bins_nm (np.ndarray): Bin edges [nm].
    """
    global DEFAULT_INTEGRATION_X_BINS_NM 
    
    DEFAULT_INTEGRATION_X_BINS_NM = integration_x_bins_nm
    return


def _geant4_energy_edges(data_dir):
    """Primary-energy bins of a Geant4 run directory, from its file names.

    Each outNuclei_<E>.txt is the response at the lower edge E of a bin; a bin ends at the next file energy and the
    last one at one more step of the same logarithmic size as the last step (the convention of the run grids).

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


def _resolve_x_bins(x_bins):
    """Returns the given track-length bins, or the current default if None.

    Args:
        x_bins (array_like or None): Bin edges [nm].

    Returns:
        np.ndarray: Bin edges [nm].
    """
    return DEFAULT_INTEGRATION_X_BINS_NM if x_bins is None else np.asarray(x_bins, dtype=float)


def log_interp1d(xx, yy, kind='linear'):
    """Builds an interpolator that works in log-log space.

    Args:
        xx (np.ndarray): x-coordinates of the data points (> 0).
        yy (np.ndarray): y-coordinates of the data points (> 0).
        kind (str, optional): Interpolation kind passed to scipy's interp1d.
            Defaults to 'linear'.

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
                built the first time it is needed. Defaults to None.
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


        self.flux_history = flux_history

        self._nuclear_data_cache = {}
        self._neutron_bkg_cache = {}
        self._depth_interpolators = {}
        self._secondary_n_spectrum = {}
        self._geometry_cache = {}
        self._depth_tables = {}

        self.energy_bins = self._scan_geant4_species()

        for species in COMPUTABLE_SPECIES:
            if species in self.energy_bins:
                self._load_depth_interpolators(species=species)

        if self.verbose > 0:
            self._print_available_species()


    def set_flux_history(self, FluxHistory):
        """Replaces the cosmic-ray flux history.

        Args:
            FluxHistory (FluxHistory): New flux history.
        """
        self.flux_history = FluxHistory


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
            note = "" if species in COMPUTABLE_SPECIES else "  (runs found, calculation not implemented yet)"
            lines.append(f"  {species:<10} energy from {e[0]:.4g} to {e[-2]:.4g} GeV in {info['n_files']} steps{note}")
            for lo, hi in info['gaps']:
                lines.append(f"  {'':<10} warning: no file between {lo:.4g} and {hi:.4g} GeV "
                             f"(the {lo:.4g} GeV run is used for the whole step)")
        if 'neutron' in self._geant4_runs:
            lines.append(f"  {'secondary_neutron':<10} neutrons produced by {', '.join(SECONDARY_NEUTRON_SOURCES)} "
                         f"above the sample (StdRock runs), recoils from the neutron runs")
        print("\n".join(lines))


    def available_species(self):
        """Species that integrate_all_particles can compute with the Geant4 runs present.

        Returns:
            list[str]: Computable species with runs, plus the derived ones whose source runs are present.
        """
        out = [s for s in COMPUTABLE_SPECIES if s in self.energy_bins]
        out += [d for d, src in DERIVED_SPECIES.items() if src in self.energy_bins]
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
        edges = self.energy_bins[source]
        if e_range is None:
            return edges
        lo, hi = (float(v) for v in e_range)
        tol = 1 + _ENERGY_EDGE_RTOL
        if not lo < hi:
            raise ValueError(f"{species}: e_range {e_range} must be (E_min, E_max) with E_min < E_max.")
        if lo * tol < edges[0] or hi > edges[-1] * tol:
            raise ValueError(f"{species}: e_range ({lo:g}, {hi:g}) GeV is outside the available range "
                             f"({edges[0]:.4g}, {edges[-1]:.4g}) GeV.")
        inside = np.where((edges[:-1] * tol >= lo) & (edges[1:] <= hi * tol))[0]
        if inside.size == 0:
            raise ValueError(f"{species}: e_range ({lo:g}, {hi:g}) GeV contains no complete energy bin.")
        return edges[inside[0]:inside[-1] + 2]


    def _time_grid_kyr(self, window_kyr, steps=None):
        """Time points of an exposure integral, placed where the integrand changes.

        The rates depend on time only through the overburden and the flux history, so the grid is built from them
        and then bounded to the window:
          * every time the overburden crosses one of DEPTH_GRID_POINTS depths equally spaced in log between its
            minimum and maximum over the window (burial, erosion, or both; exact on the interpolator nodes);
          * for every flux event, its start (with a point just before it) and the time nodes of its template, on
            which the flux is interpolated;
          * the start and end of every geomagnetic excursion, each with a point just before it (the flux jumps there);
          * the window ends, plus `steps` uniform points if given.
        With a constant overburden and a steady flux the integrand is constant and the two ends suffice.

        Args:
            window_kyr (list[float]): [start, end] in local time [kyr].
            steps (int, optional): Extra uniform points. Defaults to None (none).

        Returns:
            np.ndarray: Sorted, unique time points [kyr] within the window.
        """
        t0, t1 = float(window_kyr[0]), float(window_kyr[1])
        eps = 1e-6 * max(t1 - t0, 1.0)
        points = [np.array([t0, t1])]
        if steps:
            points.append(np.linspace(t0, t1, int(steps)))

        # overburden: crossings of log-spaced depths; piecewise linear between the interpolator nodes
        nodes = np.asarray(getattr(self._overburden_interpolator, 'x', [t0, t1]), dtype=float)
        t_nodes = np.concatenate([[t0], nodes[(nodes > t0) & (nodes < t1)], [t1]])
        depth = np.asarray(self._overburden_interpolator(t_nodes), dtype=float)
        d_lo, d_hi = max(depth.min(), DEPTH_GRID_MIN_MWE), depth.max()
        if d_hi > d_lo * (1.0 + 1e-6):
            targets = np.geomspace(d_lo, d_hi, DEPTH_GRID_POINTS)
            da, db, ta, tb = depth[:-1], depth[1:], t_nodes[:-1], t_nodes[1:]
            moving = da != db
            da, db, ta, tb = da[moving], db[moving], ta[moving], tb[moving]
            frac = (targets[:, None] - da[None, :]) / (db - da)[None, :]
            hit = (frac >= 0.0) & (frac <= 1.0)
            points.append((ta[None, :] + frac * (tb - ta)[None, :])[hit])

        # flux history, converted from its timeline (0 = present) to local time
        if self.flux_history is not None:
            to_local = self.total_age_kyr
            for event in self.flux_history.events:
                start = event["start_time_kyr"] + to_local
                points.append(np.array([start - eps, start]))
                t_since = getattr(event["template"], "t_since_kyr", None)
                if t_since is not None:
                    points.append(start + np.asarray(t_since, dtype=float) * 1e-3)   # template times are in yr
            for exc in self.flux_history.excursions:
                for edge in (exc["start_time_kyr"], exc["end_time_kyr"]):
                    points.append(np.array([edge - eps, edge]) + to_local)

        t = np.unique(np.concatenate(points))
        return t[(t >= t0) & (t <= t1)]


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

        times = np.linspace(0, self.total_age_kyr, 10000)
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

        if self.verbose>0:
            print(f"Calculating radiogenic neutron contribution from {background_type}")
        
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

        Returns:
            tuple[np.ndarray, np.ndarray]: x_bins, and tracks per mg in each bin
            accumulated over total_age_kyr.
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

        sum_drdx *= self.total_age_kyr

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

    
    def integrate_fission_spectrum(self, x_bins=None, x_grid=TRACK_LENGTH_BINS_NM):
        """Track counts from spontaneous fission over the sample age.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            x_grid (np.ndarray, optional): Fine track-length grid used for the
                calculation [nm]. Defaults to TRACK_LENGTH_BINS_NM.
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key.
                Defaults to 'total_no_light_ions'.

        Returns:
            tuple[np.ndarray, np.ndarray]: x_bins, and tracks per mg in each bin
            accumulated over total_age_kyr.
        """
        x_bins = _resolve_x_bins(x_bins)

        drdx = self.calculate_fission_spectrum(x_grid) * self.total_age_kyr

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
            'target_length_m' [m] and 'density_g_cm3' (None if unknown).
        """
        if data_dir in self._geometry_cache:
            return self._geometry_cache[data_dir]

        geo = {
            "depth_reference_z_m": 100.,
            "target_length_m": 2.0 * 100.,
            "density_g_cm3": 2.65,
        }
        meta_path = os.path.join(data_dir, "geometry.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            geo["depth_reference_z_m"] = float(meta["depth_reference_z_m"])
            geo["target_length_m"] = float(meta.get("target_length_m", 2.0 * geo["depth_reference_z_m"]))
            if meta.get("density_g_cm3") is not None:
                geo["density_g_cm3"] = float(meta["density_g_cm3"])

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

        Neutrons use the empirical distribution of the interaction depths up to the
        NEUTRON_TAIL_P quantile, with an exponential tail beyond it; the probability
        of an interaction in a slab is F(z_max) - F(z_min).

        Every outNuclei_<E>.txt in the StdRock directory is used, so a denser energy
        grid is picked up automatically.

        Args:
            species (str, optional): 'mu-', 'mu+' or 'neutron'. Defaults to 'mu-'.
        """
        tab_species = 'mu-' if species in ('mu-', 'mu+') else 'neutron'
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
                energy 0) or 'neutron' (all interaction depths).

        Returns:
            dict: 'energies' [GeV], 'p' (node probabilities), 'q' (quantile nodes
            [m.w.e.], shape (n_energies, n_nodes)) and, for neutrons, 'tail_lambda'
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

            # muons: stopping depths; neutrons: all interaction depths
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
            tab_species (str, optional): 'mu-' (stopping depths) or 'neutron'
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


    def _neutron_depth_probability(self, e_vals, z_min_grid, z_max_grid):
        """Probability that a neutron interacts in a slab.

        Args:
            e_vals (np.ndarray): Neutron energies [GeV], shape (n_E,), along the last
                axis of the z grids.
            z_min_grid (np.ndarray): Slab top along the path [m.w.e.], shape (..., n_E).
            z_max_grid (np.ndarray): Slab bottom along the path [m.w.e.], same shape.

        Returns:
            np.ndarray: F(z_max) - F(z_min), same shape as z_min_grid.
        """
        table = self._depth_tables['neutron']
        q_rows = self._depth_quantiles(e_vals, 'neutron')
        tail_lambdas = self._neutron_tail_lambdas(e_vals)
        return _slab_probability(q_rows, table['p'], z_min_grid, z_max_grid, tail_lambdas)


    def _neutron_tail_lambdas(self, energies_gev):
        """Exponential-tail length of the neutron depth distribution.

        Args:
            energies_gev (array_like): Energies [GeV], shape (n,).

        Returns:
            np.ndarray: Tail length [m.w.e.], shape (n,), log-log interpolated.
        """
        table = self._depth_tables['neutron']
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

    
    def _secondary_window_mwe(self):
        """Width of the secondary-neutron production window above the sample.

        The depth beyond which a neutron survives with probability below
        SECONDARY_WINDOW_SURVIVAL, from the StdRock neutron table (empirical
        nodes up to NEUTRON_TAIL_P, exponential tail beyond), maximised over
        the tabulated energies.

        Returns:
            float: Window width along the path [m.w.e.].
        """
        table = self._depth_tables['neutron']
        survive_last = 1.0 - table['p'][-1]
        depth = table['q'][:, -1] + table['tail_lambda'] * np.log(
            survive_last / SECONDARY_WINDOW_SURVIVAL
        )
        return float(np.max(depth))


    def _load_stdrock_neutron_rows(self, species, energy_bins_gev, total_simulated_particles=1e4):
        """Neutrons produced by the primaries of the StdRock runs.

        Reads the neutron rows of StdRock/<species> at the lower edge E_i of each
        energy cell. Depths are absolute (from the surface, where the primary
        enters) and already include the transport of the primary through the
        rock. For mu+, StdRock/mu+ is used if present; otherwise the mu- runs
        without the stopping step (no capture neutrons) stand in for it.

        Args:
            species (str): Primary species ('mu-', 'mu+' or 'neutron').
            energy_bins_gev (np.ndarray): Energy cell edges [GeV], shape (M,),
                used for the primary and for the produced neutrons.
            total_simulated_particles (float, optional): Primaries per Geant4
                run. Defaults to 1e4.

        Returns:
            dict: 'depth_mwe' (sorted production depths), 'cell' (energy cell of
            the produced neutron), 'primary_cell' (energy cell of the primary),
            'weight' (1 / primaries per run), 'target_mwe' (StdRock target
            length [m.w.e.]) and 'missing' (primary energies without a file).
        """
        tab_species = species
        drop_stopping = False
        if species == 'mu+' and not os.path.isdir(
            os.path.join(self.data_path, "Geant4_data", "StdRock/mu+")
        ):
            tab_species, drop_stopping = 'mu-', True

        data_dir = os.path.join(self.data_path, "Geant4_data", f"StdRock/{tab_species}")
        geo = self._geant4_geometry(data_dir)
        rho = geo["density_g_cm3"] if geo["density_g_cm3"] is not None else 2.65
        z_ref_m = geo["depth_reference_z_m"]

        n_e = len(energy_bins_gev) - 1
        depth_list, cell_list, prim_list, missing = [], [], [], []
        for i in range(n_e):
            filepath = os.path.join(data_dir, f"outNuclei_{energy_bins_gev[i]:.6f}.txt")
            if not os.path.exists(filepath):
                missing.append(energy_bins_gev[i])
                continue
            if os.path.getsize(filepath) == 0:
                continue
            df = pd.read_csv(
                filepath, sep=r'\s+', header=None, usecols=[0, 2, 3, 5],
                names=['name', 'rec_e', 'z_mm', 'rem_e'],
                dtype={'name': 'category', 'rec_e': 'float64', 'z_mm': 'float64', 'rem_e': 'float64'},
                engine='c',
            )
            keep = (df['name'] == 'neutron').to_numpy()
            if drop_stopping:
                keep = keep & (df['rem_e'].to_numpy() != 0.0)
            depth = np.clip((z_ref_m - df['z_mm'].values[keep] * 1e-3) * rho, 0.0, None)
            cell = np.searchsorted(energy_bins_gev, df['rec_e'].values[keep] * 1e-3, side='right') - 1
            inside = (cell >= 0) & (cell < n_e)
            depth_list.append(depth[inside])
            cell_list.append(cell[inside])
            prim_list.append(np.full(inside.sum(), i))

        depth = np.concatenate(depth_list) if depth_list else np.zeros(0)
        cell = np.concatenate(cell_list) if cell_list else np.zeros(0, dtype=int)
        prim = np.concatenate(prim_list) if prim_list else np.zeros(0, dtype=int)
        order = np.argsort(depth, kind='stable')
        return {
            'depth_mwe': depth[order], 'cell': cell[order], 'primary_cell': prim[order],
            'weight': 1.0 / total_simulated_particles,
            'target_mwe': geo["target_length_m"] * rho, 'missing': missing,
        }


    def _stdrock_neutron_offset_matrix(self, energy_bins_gev, delta_mwe, n_offsets,
                                       total_simulated_particles=1e4):
        """Neutrons produced by a neutron, by distance travelled and energy.

        From the StdRock neutron runs: C[l, d, m] is the number of neutrons with
        energy in cell m produced between d * delta and (d + 1) * delta after
        the entry point, per neutron in energy cell l.

        Args:
            energy_bins_gev (np.ndarray): Energy cell edges [GeV], shape (M,).
            delta_mwe (float): Offset bin width [m.w.e.].
            n_offsets (int): Number of offset bins.
            total_simulated_particles (float, optional): Primaries per Geant4
                run. Defaults to 1e4.

        Returns:
            np.ndarray: C, shape (M - 1, n_offsets, M - 1).
        """
        rows = self._load_stdrock_neutron_rows('neutron', energy_bins_gev, total_simulated_particles)
        n_e = len(energy_bins_gev) - 1
        d = np.floor(rows['depth_mwe'] / delta_mwe).astype(int)
        ok = d < n_offsets
        flat = (rows['primary_cell'][ok] * n_offsets + d[ok]) * n_e + rows['cell'][ok]
        C = np.bincount(flat, minlength=n_e * n_offsets * n_e).astype(float)
        return C.reshape(n_e, n_offsets, n_e) * rows['weight']


    def _secondary_neutron_yield(self, t_kyr_array, depth_mwe_array, c_vals, energy_bins_gev,
                                 total_simulated_particles=1e4,
                                 species_list=SECONDARY_NEUTRON_SOURCES, n_energy_sub=50):
        """Secondary neutrons born in a window above the sample, per direction.

        For each time t and direction c, the path to the sample has slant depth
        S = X(t) / c. The window [S - W, S] along the path is split into
        SECONDARY_WINDOW_BINS bins of width delta = W / SECONDARY_WINDOW_BINS.

        Source: neutrons produced in the window by the primaries, from the
        StdRock runs at their actual slant depth (so the transport of the
        primary down to the window is included), weighted by the primary
        intensity integrated over each energy cell:

            A_b,m = sum_s sum_i Phi_s,i(t) * (StdRock/<species> neutrons of cell m in bin b per primary of cell i)

        Generations inside the window use the StdRock neutron runs (offset
        matrix C_n). Neutrons are born uniformly within their bin, so a
        production at offset bin d is shared between bins b' + d and b' + d + 1:

            Y_b (I - C_n[0] / 2) = A_b + sum_{b' < b} Y_b' (C_n[b - b' - 1] + C_n[b - b']) / 2

        Secondary neutrons are assumed to keep the direction of the primary.

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            depth_mwe_array (np.ndarray): Overburden [m.w.e.], shape (T,).
            c_vals (np.ndarray): cos(zenith) grid, shape (N_C,).
            energy_bins_gev (np.ndarray): Energy cell edges [GeV], shape (M,).
            total_simulated_particles (float, optional): Primaries per Geant4
                run. Defaults to 1e4.
            species_list (tuple[str], optional): Primary species producing
                neutrons. Defaults to SECONDARY_NEUTRON_SOURCES.
            n_energy_sub (int, optional): Points per energy cell for the flux
                integration. Defaults to 50.

        Returns:
            tuple[np.ndarray, float]:
                Y: neutrons born in each window bin and energy cell
                    [cm^-2 s^-1 sr^-1], shape (T, N_C, SECONDARY_WINDOW_BINS, M - 1).
                delta: window bin width along the path [m.w.e.].
        """
        energy_bins_gev = np.asarray(energy_bins_gev, dtype=float)
        n_t, n_c, n_e = len(t_kyr_array), len(c_vals), len(energy_bins_gev) - 1
        n_b = SECONDARY_WINDOW_BINS
        window = self._secondary_window_mwe()
        delta = window / n_b

        # primary intensity integrated over each energy cell [cm^-2 s^-1 sr^-1]
        t_kyr_flux_array = self._flux_time_kyr(np.asarray(t_kyr_array, dtype=float))
        sub = np.linspace(0.0, 1.0, n_energy_sub)
        e_lo, e_hi = energy_bins_gev[:-1], energy_bins_gev[1:]
        e_sub = e_lo[:, None] + (e_hi - e_lo)[:, None] * sub[None, :]
        step = (e_hi - e_lo) / (n_energy_sub - 1)

        slant = depth_mwe_array[:, None] / c_vals[None, :]               # (T, N_C) path to the sample
        window_top = slant - window

        A = np.zeros((n_t, n_c, n_b, n_e))
        for species in species_list:
            rows = self._load_stdrock_neutron_rows(species, energy_bins_gev, total_simulated_particles)
            if rows['missing'] and self.verbose > 0:
                print(f"Warning: StdRock/{species}: no file for {len(rows['missing'])} primary "
                      f"energies (e.g. {rows['missing'][:3]} GeV); their neutrons are missing.")
            beyond = slant > TRUNCATION_FRACTION * rows['target_mwe']
            if self.verbose > 0 and np.any(beyond[:, -1]):
                print(f"Warning: vertical overburden beyond the StdRock/{species} target "
                      f"({rows['target_mwe']:.0f} m.w.e.) at some times: secondary neutrons "
                      "from that species are missing there; use a longer StdRock target.")

            phi = self.flux_history.get_map(species, t_kyr_flux_array, e_sub)[2].reshape(
                n_t, *e_sub.shape
            )
            phi_cell = (phi.sum(axis=-1) - 0.5 * (phi[..., 0] + phi[..., -1])) * step[None, :]

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
                    A[t, c] += np.bincount(
                        b * n_e + rows['cell'][sl], weights=w, minlength=n_b * n_e
                    ).reshape(n_b, n_e)

        # generations inside the window
        C_n = self._stdrock_neutron_offset_matrix(
            energy_bins_gev, delta, n_b + 1, total_simulated_particles
        )
        half_same_bin = 0.5 * C_n[:, 0, :]
        if np.max(np.abs(np.linalg.eigvals(half_same_bin))) < 1.0:
            same_bin_inv = np.linalg.inv(np.eye(n_e) - half_same_bin)
        else:
            if self.verbose > 0:
                print("Warning: same-bin neutron multiplication >= 1; ignoring it.")
            same_bin_inv = np.eye(n_e)
        transfer = [None] + [0.5 * (C_n[:, k - 1, :] + C_n[:, k, :]) for k in range(1, n_b + 1)]

        A = A.reshape(n_t * n_c, n_b, n_e)
        Y = np.zeros_like(A)
        for b in range(n_b):
            y = A[:, b, :].copy()
            for bp in range(b):
                y += Y[:, bp, :] @ transfer[b - bp]
            Y[:, b, :] = y @ same_bin_inv

        return Y.reshape(n_t, n_c, n_b, n_e), delta


    def _get_all_fragments(self, energy_names_gev, species='mu-'):
        """Lists the fragments present in a mineral's Geant4 output.

        Args:
            energy_names_gev (array_like): Energies [GeV] of the files to scan.
            species (str, optional): 'mu-', 'mu+' or 'neutron'. Defaults to 'mu-'.

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
            species (str, optional): 'mu-', 'mu+' or 'neutron'. Defaults to 'mu-'.

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
                "flux_history not initialized. Call integrate_particle_signal_spectrum "
                "(or integrate_all_particles) first, or use set_flux_history()."
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
                prob_attenuation_grid = self._neutron_depth_probability(
                    e_vals, z_min_grid, z_max_grid
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


    def _process_secondary_geant4_data(
        self, t_kyr_array, energy_bins_gev=None,
        total_simulated_particles=1e4, secondary_neutrons_species=SECONDARY_NEUTRON_SOURCES,
    ):
        """Recoil spectra in the sample slab from secondary neutrons.

        Neutrons born in the window above the sample (see
        _secondary_neutron_yield) travel along the primary's direction to the
        slab. For a neutron born uniformly in window bin b, at distance
        d in [d0, d0 + delta] from the slab top, the probability of interacting
        in the slab (path tau = t_slab / c) is averaged over the bin exactly:

            P_b = [H(d0 + delta + tau) - H(d0 + tau) - H(d0 + delta) + H(d0)] / delta,

        with H(z) = integral_0^z F_n = z - G(z). Then

            w_m(t) = int dc sum_b Y[t, c, b, m] P_b(E_m, c),

        and the recoils per neutron of the mineral run at the lower edge E_m of
        each cell are applied with the normalisation of _process_geant4_data.

        Args:
            t_kyr_array (np.ndarray): Local times [kyr], shape (T,).
            energy_bins_gev (np.ndarray, optional): Energy cell edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            total_simulated_particles (float, optional): Primaries per Geant4
                run. Defaults to 1e4.
            secondary_neutrons_species (tuple[str], optional): Primary species
                producing neutrons. Defaults to SECONDARY_NEUTRON_SOURCES.

        Returns:
            dict: Same structure as _process_geant4_data.
        """
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, 'secondary_neutron')
        rho = self.config['density_g_cm3']

        target_volume_cm3 = 1.e-3 / rho
        target_thickness_cm = np.power(target_volume_cm3, 1/3)
        target_thickness_mwe = target_thickness_cm * 0.01 * rho

        all_fragments = self._get_all_fragments(energy_bins_gev[:-1], species='neutron')
        geant4_input_dir = os.path.join(self.data_path, "Geant4_data", self.name, "neutron")

        t_kyr_array = np.atleast_1d(np.asarray(t_kyr_array, dtype=float))
        n_t = len(t_kyr_array)
        depth_mwe_array = np.atleast_1d(self._overburden_interpolator(t_kyr_array))

        c_vals = np.linspace(0.01, 1.0, 50)
        Y, delta = self._secondary_neutron_yield(
            t_kyr_array, depth_mwe_array, c_vals, energy_bins_gev,
            total_simulated_particles, secondary_neutrons_species,
        )
        n_b = Y.shape[2]
        n_e = len(energy_bins_gev) - 1

        # bin-averaged slab probability, shape (N_C, n_b, M-1)
        e_cells = energy_bins_gev[:-1]
        table = self._depth_tables['neutron']
        q_rows = self._depth_quantiles(e_cells, 'neutron')
        tails = self._neutron_tail_lambdas(e_cells)
        d0 = (n_b - 1 - np.arange(n_b)) * delta                       # distance from bin bottom to slab
        tau = target_thickness_mwe / c_vals                              # slab path
        z_d0 = np.broadcast_to(d0[None, :], (len(c_vals), n_b))
        z_tau = np.broadcast_to(tau[:, None], (len(c_vals), n_b))
        prob = np.empty((len(c_vals), n_b, n_e))
        for m in range(n_e):
            def H(z):
                """Integral of the neutron depth CDF from 0 to z [m.w.e.]."""
                _, G = _piecewise_cdf_and_integral(z, q_rows[m], table['p'], tails[m])
                return z - G
            prob[:, :, m] = (H(z_d0 + delta + z_tau) - H(z_d0 + z_tau)
                             - H(z_d0 + delta) + H(z_d0)) / delta

        weight_t = np.trapezoid(np.einsum('tcbm,cbm->tcm', Y, prob), c_vals, axis=1)  # (T, M-1)

        n_recoil_bins = len(RECOIL_ENERGY_BINS_MEV) - 1
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

        bin_widths_mev = np.diff(RECOIL_ENERGY_BINS_MEV)
        norm_factor = (
            bin_widths_mev * target_thickness_cm * rho
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


    def _convert_recoil_to_track_spectrum(self, recoil_data, x_bins=None, energy_bins_gev=None, species='mu-'):
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
                'neutron' or 'secondary_neutron'). Defaults to 'mu-'.

        Returns:
            dict: dR/dx per fragment and 'total' [tracks per mg per kyr per nm],
            shape (T, n_x), or (n_x,) if the input was 1-D.
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, species)

        er_bins = recoil_data['Er_bins']
        er_mid_mev = er_bins[:-1] + np.diff(er_bins) / 2.0

        tab_species = 'neutron' if species == 'secondary_neutron' else species
        all_fragments = self._get_all_fragments(energy_bins_gev[:-1], tab_species)

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
            species (str, optional): 'mu-', 'mu+', 'neutron' or
                'secondary_neutron'. Defaults to 'mu-'.
            nucleus (str, optional): 'total_no_light_ions' (all fragments except H and He, see
                LIGHT_ION_MAX_Z), 'total' (all fragments), 'all' (dict of every fragment and both sums) or a
                fragment key. Defaults to 'total_no_light_ions'.

        Returns:
            np.ndarray or dict: dR/dx [tracks per mg per kyr per nm], shape (T, n_x).
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = self._resolve_energy_bins(energy_bins_gev, species)

        if species == 'secondary_neutron':
            recoil_data = self._process_secondary_geant4_data(
                t_kyr_array, energy_bins_gev, total_simulated_particles,
            )
        else:
            recoil_data = self._process_geant4_data(
                t_kyr_array, energy_bins_gev, total_simulated_particles, species,
            )

        dRdx_at_depth = self._convert_recoil_to_track_spectrum(recoil_data, x_bins, energy_bins_gev, species)

        if nucleus == "all":
            return dRdx_at_depth
        if nucleus not in dRdx_at_depth:
            raise ValueError(f"nucleus '{nucleus}' not found: use one of {TRACK_SUMS}, 'all' or a fragment of "
                             f"{species} ({sorted(k for k in dRdx_at_depth if k not in TRACK_SUMS)}).")
        return dRdx_at_depth[nucleus]


    def integrate_particle_signal_spectrum(
        self, x_bins=None, energy_bins_gev=None,
        exposure_window_kyr=None, flux_history=None, steps=None, total_simulated_particles=1e4, 
        x_grid=TRACK_LENGTH_BINS_NM, species='mu-', nucleus=DEFAULT_TRACK_SUM, e_range=None,
    ):
        """Track counts from one cosmic-ray species over an exposure window.

        The rate is computed on a time grid and integrated with the trapezoidal
        rule; the fine-grid spectrum is then rebinned to x_bins through its
        cumulative distribution.

        Args:
            x_bins (np.ndarray, optional): Output track-length bin edges [nm].
                Defaults to None (DEFAULT_INTEGRATION_X_BINS_NM).
            energy_bins_gev (np.ndarray, optional): Primary energy bin edges [GeV].
                Defaults to None (all the Geant4 files of the species).
            exposure_window_kyr (float or list[float], optional): Window in local
                time [kyr]; a single value w means [0, w]. Defaults to None
                ([0, total_age_kyr]).
            flux_history (FluxHistory, optional): Replaces the current flux history.
                Defaults to None (current one, or a baseline history if none).
            steps (int, optional): Extra uniform time points added to the adaptive grid (_time_grid_kyr).
                Defaults to None (adaptive grid only).
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            x_grid (np.ndarray, optional): Fine track-length grid [nm]. Defaults to
                TRACK_LENGTH_BINS_NM.
            species (str, optional): 'mu-', 'mu+', 'neutron' or
                'secondary_neutron'. Defaults to 'mu-'.
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key (see
                calculate_particle_signal_spectrum). Defaults to 'total_no_light_ions'.
            e_range (tuple[float, float], optional): (E_min, E_max) [GeV]: use only the Geant4 files whose bins
                lie inside it (see _energy_edges). Ignored if energy_bins_gev is given. Defaults to None (all).

        Returns:
            np.ndarray: Tracks per mg in each bin of x_bins.
        """
        x_bins = _resolve_x_bins(x_bins)
        energy_bins_gev = (self._energy_edges(species, e_range) if energy_bins_gev is None
                           else np.asarray(energy_bins_gev, dtype=float))

        if flux_history is not None:
            self.flux_history = flux_history
        elif self.flux_history is None:
            self.flux_history = FluxHistory(
                baseline={"kind": "Baseline"}, template_dir=os.path.join(self.data_path, "flux_data")
            )

        if exposure_window_kyr is None:
            exposure_window_kyr = self.total_age_kyr
        if isinstance(exposure_window_kyr, (int, float)):
            exposure_window_kyr = [0, exposure_window_kyr]

        t_kyr_array = self._time_grid_kyr(exposure_window_kyr, steps)

        dRdx_array = self.calculate_particle_signal_spectrum(
            t_kyr_array, x_bins=x_grid, energy_bins_gev=energy_bins_gev,
            total_simulated_particles=total_simulated_particles,
            species=species, nucleus=nucleus
        )

        total_drdx_g = np.trapezoid(dRdx_array, t_kyr_array, axis=0)

        internal_bin_widths = np.diff(x_grid)
        cumulative_counts = np.concatenate(([0], np.cumsum(total_drdx_g * internal_bin_widths)))
        cdf_interp = interp1d(x_grid, cumulative_counts, kind='linear', bounds_error=False,
                            fill_value=(0, cumulative_counts[-1]))
        total_tracks = cdf_interp(x_bins[1:]) - cdf_interp(x_bins[:-1])

        return total_tracks


    def integrate_all_particles(
        self,  x_bins=None, e_range=None,
        exposure_window_kyr=None, flux_history=None,steps=None, 
        total_simulated_particles=1e4, species_list=None, nucleus=DEFAULT_TRACK_SUM
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
            flux_history (FluxHistory, optional): See
                integrate_particle_signal_spectrum. Defaults to None.
            steps (int, optional): See integrate_particle_signal_spectrum.
                Defaults to None.
            total_simulated_particles (float, optional): Primaries per Geant4 run.
                Defaults to 1e4.
            species_list (tuple[str], optional): Species to compute. Defaults to None (available_species()).
            nucleus (str, optional): 'total_no_light_ions', 'total' or a fragment key (see
                calculate_particle_signal_spectrum). Defaults to 'total_no_light_ions'.

        Returns:
            tuple[np.ndarray, dict]: x_bins, and tracks per mg in each bin for each
            species plus 'total'.

        Raises:
            ValueError: If a species has no runs, or e_range names a species not computed or an invalid range.
        """
        x_bins = _resolve_x_bins(x_bins)
        species_list = self.available_species() if species_list is None else list(species_list)
        e_range = dict(e_range or {})
        unknown = set(e_range) - set(species_list)
        if unknown:
            raise ValueError(f"e_range given for species not computed: {sorted(unknown)} "
                             f"(computed: {species_list}).")
        edges = {s: self._energy_edges(s, e_range.get(s)) for s in species_list}   # validates before the pool

        shared_kwargs = dict(
            x_bins=x_bins,
            exposure_window_kyr=exposure_window_kyr, flux_history=flux_history, steps=steps, 
            total_simulated_particles=total_simulated_particles, nucleus=nucleus
        )

        tasks = [(self, dict(shared_kwargs, species=s, energy_bins_gev=edges[s])) for s in species_list]

        with Pool(processes=min(len(species_list), os.cpu_count() or 1)) as pool:
            results = list(tqdm(pool.imap(_species_worker, tasks), total=len(tasks)))

        total_tracks_by_species = dict(results)
        total_tracks_by_species['total'] = sum(total_tracks_by_species.values())

        return x_bins, total_tracks_by_species


def _species_worker(self_and_args):
    """Pool worker: computes the track counts of one species.

    Args:
        self_and_args (tuple): (Paleodetector instance, keyword arguments of
            integrate_particle_signal_spectrum including 'species').

    Returns:
        tuple[str, np.ndarray]: (species, tracks per mg in each bin).
    """
    self, kwargs = self_and_args
    species = kwargs.pop('species')
    if self.verbose > 1:
        print(f"Processing species: {species}")
    return species, self.integrate_particle_signal_spectrum(species=species, **kwargs)