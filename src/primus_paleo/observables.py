"""What is measured: the etch-pit openings of latent tracks and the expected pit-size spectrum on the analysed
surface (etching, optical resolution of the microscope), and rebinning of spectra to the bins of a measurement.

The diagnostics of a Paleodetector (interaction_profiles, plot_secondary_production) are in plotting.
"""

import numpy as np

from ._core import _rebin_density, _resolve_x_bins

__all__ = [
    'etch_pit_axes', 'rebin_counts', 'slice_spectrum_etch', 'resolution_limit_nm', 'ETCH_DEFAULTS',
    'PIXEL_SIZE_UM_AT_10X', 'MIN_RESOLVED_PIXELS', 'DEFAULT_MAGNIFICATION',
]


def etch_pit_axes(length_nm, dip_rad, top_depth_nm, removed_nm, v_ratio, lateral_ratio=1.0, n_s=128):
    """Surface opening of etched latent tracks (Huygens construction, constant etch rates).

    Every point of a straight latent track starts dissolving when the etchant reaches it (along the track at v_T,
    from its top point) and from then on grows at the bulk rate v_B: vertically at v_B, laterally at
    lateral_ratio * v_B. The pit is the union of these growing ellipses plus the removed bulk layer; its opening is
    the section of that union by the final surface. Covers in one construction the critical angle
    (sin(dip) < 1/V gives no opening), the conical phase, over-etching (the etch front has reached the track end and
    the tip rounds off) and pits that fade out because the bulk has etched through them.

    Construction, for each track: etching of the track starts at depth z0 = max(z_top, 0) (the part above the
    original surface does not exist); R is the latent length below z0, and h' = h - z0 the bulk removal left once
    the etchant has reached the track (no opening if h' <= 0 or R <= 0). The etchant runs along the track for
    s_max = min(R, V h'); a point at distance s along it has then grown vertically by r = h' - s / V, and lies
    dz = h' - s sin(dip) below the final surface, so its ellipse cuts the surface in a segment of half-width
    a = lateral_ratio * sqrt(r^2 - dz^2) centred at x = s cos(dip). The opening spans the union of the segments of
    n_s points along the track: major axis = its extent, minor axis = 2 max(a). Tracks are processed in chunks of
    ETCH_CHUNK to bound the memory.

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
    z0 = np.maximum(z_top, 0.)
    R = (z_top + L * sin_d - z0) / np.maximum(sin_d, 1e-12)
    hp = removed_nm - z0
    ok = (hp > 0) & (R > 0)

    major, minor = np.full(L.shape, np.nan), np.full(L.shape, np.nan)
    idx = np.where(ok)[0]
    for k in range(0, len(idx), ETCH_CHUNK):
        i = idx[k:k + ETCH_CHUNK]
        s_max = np.minimum(R[i], v_ratio * hp[i])
        s = s_max[:, None] * np.linspace(0., 1., n_s)[None, :]
        r = hp[i, None] - s / v_ratio
        dz = hp[i, None] - s * sin_d[i, None]
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
    where the two binnings cover the same range; counts outside `new_edges` are dropped. The counts form of
    _core._rebin_density (which works on densities), e.g. to bring a spectrum to the bins of a descriptor.

    Args:
        counts (np.ndarray): Counts in the bins of `old_edges` (len(old_edges) - 1 values).
        old_edges, new_edges (np.ndarray): Increasing bin edges, same units (e.g. nm).

    Returns:
        np.ndarray: Counts in the bins of `new_edges`.

    Raises:
        ValueError: If counts does not match old_edges.
    """
    counts = np.asarray(counts, float)
    old_edges, new_edges = np.asarray(old_edges, float), np.asarray(new_edges, float)
    if len(counts) != len(old_edges) - 1:
        raise ValueError(f"{len(counts)} counts for {len(old_edges) - 1} bins")
    return _rebin_density(counts / np.diff(old_edges), old_edges, new_edges)


# Etching: ETCH_DEFAULTS gives removed_nm, v_ratio and lateral_ratio of each etching method (slice_spectrum_etch,
# etch_type=...). The plasma values are placeholders, to be replaced by measured ones.
ETCH_DEFAULTS = {
    'HF': {'removed_nm': 4000., 'v_ratio': 2.5, 'lateral_ratio': 1.0},
    'plasma': {'removed_nm': 100., 'v_ratio': 10.0, 'lateral_ratio': 0.1},
}
ETCH_CHUNK = 20000


# Microscope: camera pixel size on the sample at 10x; at magnification M it is PIXEL_SIZE_UM_AT_10X * 10 / M (same
# camera and tube lens). A pit must be at least MIN_RESOLVED_PIXELS pixels wide to be seen (resolution_limit_nm).
PIXEL_SIZE_UM_AT_10X = 0.345
MIN_RESOLVED_PIXELS = 3
DEFAULT_MAGNIFICATION = 10
DEFAULT_WAVELENGTH_NM = 550.


def resolution_limit_nm(magnification=DEFAULT_MAGNIFICATION, n_pixels=MIN_RESOLVED_PIXELS, numerical_aperture=None,
                        wavelength_nm=DEFAULT_WAVELENGTH_NM):
    """Smallest pit width the microscope resolves [nm]: the default minimum width of slice_spectrum_etch.

    The camera criterion is n_pixels pixels, a pixel being PIXEL_SIZE_UM_AT_10X * 10 / magnification on the sample
    (0.69 um at 5x, 0.345 um at 10x, 0.086 um at 40x). At high magnification the optics, not the camera, limit the
    resolution: if numerical_aperture is given, the limit is the larger of the camera criterion and the Rayleigh
    distance 0.61 * wavelength / NA (e.g. 40x with NA 0.65: 0.52 um, against 0.26 um for 3 pixels).

    Args:
        magnification (float, optional): Objective magnification. Defaults to DEFAULT_MAGNIFICATION (10).
        n_pixels (float, optional): Minimum width in pixels. Defaults to MIN_RESOLVED_PIXELS (3).
        numerical_aperture (float, optional): NA of the objective. Defaults to None (camera criterion only).
        wavelength_nm (float, optional): Wavelength of the illumination [nm]. Defaults to 550.

    Returns:
        float: Minimum resolved width [nm].
    """
    limit = n_pixels * PIXEL_SIZE_UM_AT_10X * 1e3 * 10. / magnification
    if numerical_aperture is not None:
        limit = max(limit, 0.61 * wavelength_nm / numerical_aperture)
    return limit


def slice_spectrum_etch(counts_mg, sample_area_cm2, sample_density_g_cm3, removed_nm=None, v_ratio=None,
                        lateral_ratio=None, angular_pdf=None, min_width_nm=None, measure='major', rng=None,
                        x_bins=None, out_bins=None, etch_type="HF", magnification=DEFAULT_MAGNIFICATION,
                        numerical_aperture=None, n_samples=100_000, realisation=False):
    """Expected etched-pit size distribution on the analysed surface, from a latent track-length spectrum.

    The latent tracks of counts_mg (tracks per mg) fill the sample uniformly, with an isotropic direction (or the
    dip distribution angular_pdf); a surface area A of a sample of density rho cuts through A / side^2 cubes of 1 mg
    (side = (1e-3 / rho)^(1/3)), i.e. A / side^2 * sum(counts_mg) tracks per side of depth. A track of length L and
    dip d is etched if its top lies between -L sin(d) (it crosses the original surface) and the removed thickness h,
    a fraction (h + L sin d) / side of the tracks; its measured size is the opening of its pit (etch_pit_axes), and
    pits narrower than the resolution (minor axis below min_width_nm) are not seen.

    Computed by Monte Carlo with importance sampling: n_samples tracks are drawn (lengths from counts_mg, uniform
    inside each bin; dips; tops uniform in the etched interval only), each weighted by the fraction above, so every
    sample contributes and the result is the expected number of pits per bin, with a relative Monte Carlo error of
    about 1 / sqrt(samples in the bin). With realisation=True one measurement is returned instead: a Poisson draw
    of every bin around that expectation.

    The latent spectrum and the measured sizes can use different binnings: `x_bins` describes `counts_mg` (e.g. the
    wide bins of the theoretical spectrum) and `out_bins` the returned histogram (e.g. the bins of the descriptor).

    Args:
        counts_mg (np.ndarray): Latent track spectrum, tracks per mg in each bin of x_bins.
        sample_area_cm2, sample_density_g_cm3 (float): Analysed area [cm^2] and target density [g/cm^3].
        removed_nm (float, optional): Bulk layer removed, h = v_B * t [nm]. Default from `etch_type`.
        v_ratio (float, optional): V = v_T / v_B. Default from `etch_type`.
        lateral_ratio (float, optional): See etch_pit_axes (1 isotropic, 0 directional). Default from `etch_type`.
        angular_pdf (np.ndarray, optional): Relative probability of the dip angle on a uniform grid from 0 to pi/2
            (len(angular_pdf) points). Defaults to None (isotropic: sin(dip) uniform).
        min_width_nm (float, optional): Pits with a minor axis below this width are not seen. Defaults to None:
            resolution_limit_nm(magnification, numerical_aperture=numerical_aperture) (1035 nm at 10x).
        measure (str, optional): 'major' (what OptimusPrimus reports as length) or 'minor'.
        rng (np.random.Generator, optional): Random generator.
        x_bins (np.ndarray, optional): Bin edges of `counts_mg` [nm]. Defaults to DEFAULT_INTEGRATION_X_BINS_NM.
        out_bins (np.ndarray, optional): Bin edges of the returned histogram [nm]. Defaults to x_bins.
        etch_type (str, optional): 'HF' or 'plasma': sets the defaults of removed_nm, v_ratio and lateral_ratio
            (ETCH_DEFAULTS; wet HF is isotropic, dry plasma directional); explicitly given values always win,
            including 0. Defaults to "HF".
        magnification (float, optional): Objective magnification, for the default min_width_nm. Defaults to 10.
        numerical_aperture (float, optional): NA of the objective, for the default min_width_nm (see
            resolution_limit_nm). Defaults to None (pixel criterion only).
        n_samples (int, optional): Monte Carlo tracks. Defaults to 100000 (about 2% error in a bin with 1/40 of
            the pits).
        realisation (bool, optional): Return one Poisson realisation (integer counts) instead of the expectation.
            Defaults to False.

    Returns:
        np.ndarray: Expected number of pits per bin of out_bins on the analysed area (or one realisation).

    Raises:
        ValueError: If etch_type or measure is unknown, or counts_mg does not match x_bins.
    """
    if etch_type not in ETCH_DEFAULTS:
        raise ValueError(f"etch_type must be one of {list(ETCH_DEFAULTS)}, got '{etch_type}'")
    if measure not in ('major', 'minor'):
        raise ValueError("measure must be 'major' or 'minor'")
    given = {'removed_nm': removed_nm, 'v_ratio': v_ratio, 'lateral_ratio': lateral_ratio}
    etch = {**ETCH_DEFAULTS[etch_type], **{k: v for k, v in given.items() if v is not None}}
    removed_nm, v_ratio, lateral_ratio = etch['removed_nm'], etch['v_ratio'], etch['lateral_ratio']
    if min_width_nm is None:
        min_width_nm = resolution_limit_nm(magnification, numerical_aperture=numerical_aperture)

    rng = rng or np.random.default_rng()
    x_bins = _resolve_x_bins(x_bins)
    out_bins = x_bins if out_bins is None else np.asarray(out_bins, float)
    counts_mg = np.asarray(counts_mg, float)
    if len(counts_mg) != len(x_bins) - 1:
        raise ValueError(f"counts_mg has {len(counts_mg)} values for {len(x_bins) - 1} bins of x_bins")
    total = counts_mg.sum()
    if total <= 0:
        return np.zeros(len(out_bins) - 1)

    side_cm = (1.e-3 / sample_density_g_cm3) ** (1. / 3.)
    side_nm = side_cm * 1.e7
    tracks_per_side = sample_area_cm2 / side_cm ** 2 * total

    n = int(n_samples)
    k = rng.choice(len(counts_mg), size=n, p=counts_mg / total)
    L = rng.uniform(x_bins[k], x_bins[k + 1])
    if angular_pdf is not None:
        grid = np.linspace(0., np.pi / 2., len(angular_pdf))
        dip = rng.choice(grid, size=n, p=np.asarray(angular_pdf, float) / np.sum(angular_pdf))
    else:
        dip = np.arcsin(rng.uniform(0., 1., size=n))
    z_lo = np.maximum(-L * np.sin(dip), -side_nm / 2.)
    z_hi = np.minimum(removed_nm, side_nm / 2.)
    weight = tracks_per_side / n * np.clip(z_hi - z_lo, 0., None) / side_nm
    z_top = rng.uniform(z_lo, np.maximum(z_hi, z_lo))

    major, minor = etch_pit_axes(L, dip, z_top, removed_nm, v_ratio, lateral_ratio)
    resolved = np.isfinite(minor) & (minor >= min_width_nm)
    size = (major if measure == 'major' else minor)[resolved]
    expected = np.histogram(size, bins=out_bins, weights=weight[resolved])[0]
    return rng.poisson(expected) if realisation else expected
