"""What is measured, in two steps.

Etching (this module): the latent tracks cut by the analysed surface become etch pits, each with a major and a minor
axis (etch_pit_axes). The etching response (etch_response, EtchResponse) is a matrix that turns a latent
track-length spectrum [tracks per mg] into the expected pits per cm^2, binned in the measured size (major axis by
default) and in pit width (minor axis): every etched pit, whatever its size. slice_spectrum_etch applies it to one
spectrum, compare_scenarios to several, with the significance of their differences.

Observation (later step, not applied here): which pits the microscope resolves, at a given magnification and pixel
size, depends on their width; the width axis of the etching response is kept for it. resolution_limit_nm gives the
smallest resolved width.

Also: rebin_counts (spectra to the bins of a measurement). The diagnostics of a Paleodetector are in plotting.
"""

import numpy as np

from ._core import _rebin_density, _resolve_x_bins

__all__ = [
    'etch_pit_axes', 'rebin_counts', 'EtchResponse', 'etch_response', 'slice_spectrum_etch', 'compare_scenarios',
    'resolution_limit_nm', 'ETCH_DEFAULTS', 'ETCH_SAMPLES_PER_BIN', 'ETCH_WIDTH_BINS_NM', 'PIXEL_SIZE_UM_AT_10X',
    'MIN_RESOLVED_PIXELS', 'DEFAULT_MAGNIFICATION',
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


# Etching: ETCH_DEFAULTS gives removed_nm, v_ratio and lateral_ratio of each etching method (etch_type=...). The
# plasma values are placeholders, to be replaced by measured ones. The etching response draws ETCH_SAMPLES_PER_BIN
# Monte Carlo tracks per latent-length bin (relative error ~1/sqrt of the samples landing in a size bin) and bins the
# pit width (minor axis) on ETCH_WIDTH_BINS_NM, the axis the observation step needs: 20 log bins per decade from 0.1 to
# 10 um (the range of the microscope resolution limits), coarser outside (10 nm - 100 um; narrower and wider pits go
# to the first and last bin). The matrix is stored in float32 (~44 MB for the default 499 x 499 bins).
ETCH_DEFAULTS = {
    'HF': {'removed_nm': 4000., 'v_ratio': 2.5, 'lateral_ratio': 1.0},
    'plasma': {'removed_nm': 100., 'v_ratio': 10.0, 'lateral_ratio': 0.1},
}
ETCH_CHUNK = 20000
ETCH_SAMPLES_PER_BIN = 2000
ETCH_WIDTH_BINS_NM = np.concatenate([[10.], np.geomspace(100., 1e4, 41), [1e5]])
_ETCH_RESPONSES = {}


# Microscope (observation step): camera pixel size on the sample at 10x; at magnification M it is
# PIXEL_SIZE_UM_AT_10X * 10 / M (same camera and tube lens). A pit must be at least MIN_RESOLVED_PIXELS pixels wide to
# be seen (resolution_limit_nm).
PIXEL_SIZE_UM_AT_10X = 0.345
MIN_RESOLVED_PIXELS = 3
DEFAULT_MAGNIFICATION = 10
DEFAULT_WAVELENGTH_NM = 550.


def resolution_limit_nm(magnification=DEFAULT_MAGNIFICATION, n_pixels=MIN_RESOLVED_PIXELS, numerical_aperture=None,
                        wavelength_nm=DEFAULT_WAVELENGTH_NM):
    """Smallest pit width the microscope resolves [nm]: for the observation step (not applied by the etching).

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


class EtchResponse:
    """Expected etch pits per cm^2 of analysed surface per latent track per mg, by latent-length bin: a linear map
    from a latent track spectrum to the etched-pit spectrum. Made by etch_response.

    Attributes:
        matrix (np.ndarray): R[size bin, width bin, latent bin], pits per cm^2 per (track per mg).
        x_bins (np.ndarray): Latent track-length bins [nm].
        out_bins (np.ndarray): Measured-size bins [nm] (major axis, or minor with measure='minor').
        width_bins (np.ndarray): Pit-width (minor axis) bins [nm], for the observation step.
        params (dict): Etching and sample parameters it was built with.
    """

    def __init__(self, matrix, x_bins, out_bins, width_bins, params):
        self.matrix, self.x_bins, self.out_bins, self.width_bins = matrix, x_bins, out_bins, width_bins
        self.params = dict(params)

    def pits(self, counts_mg, sample_area_cm2, by_width=False):
        """Expected etched pits on the analysed area for a latent spectrum.

        Args:
            counts_mg (array_like): Latent tracks per mg in each bin of x_bins.
            sample_area_cm2 (float): Analysed area [cm^2].
            by_width (bool, optional): Keep the width axis: (size bins, width bins). Defaults to False (summed).

        Returns:
            np.ndarray: Expected pits per size bin (or per size and width bin).

        Raises:
            ValueError: If counts_mg does not match x_bins.
        """
        counts_mg = np.asarray(counts_mg, float)
        if counts_mg.shape != (len(self.x_bins) - 1,):
            raise ValueError(f"counts_mg has {counts_mg.size} values for {len(self.x_bins) - 1} bins of x_bins")
        per_width = sample_area_cm2 * np.einsum('owi,i->ow', self.matrix, counts_mg, dtype=float)
        return per_width if by_width else per_width.sum(axis=1)


def _etch_parameters(etch_type, removed_nm, v_ratio, lateral_ratio):
    """removed_nm, v_ratio, lateral_ratio: the given ones, the others from ETCH_DEFAULTS[etch_type] (explicit values
    win, including 0). Called by etch_response.

    Raises:
        ValueError: If etch_type is unknown.
    """
    if etch_type not in ETCH_DEFAULTS:
        raise ValueError(f"etch_type must be one of {list(ETCH_DEFAULTS)}, got '{etch_type}'")
    given = {'removed_nm': removed_nm, 'v_ratio': v_ratio, 'lateral_ratio': lateral_ratio}
    return {**ETCH_DEFAULTS[etch_type], **{k: v for k, v in given.items() if v is not None}}


def etch_response(sample_density_g_cm3, x_bins=None, out_bins=None, etch_type="HF", removed_nm=None, v_ratio=None,
                  lateral_ratio=None, angular_pdf=None, measure='major', width_bins=None,
                  n_per_bin=ETCH_SAMPLES_PER_BIN, seed=0):
    """The etching response of a sample: expected etched pits per cm^2 per latent track per mg, by latent-length bin
    (EtchResponse). Built once per set of arguments and kept for the session.

    The latent tracks fill the sample uniformly, with an isotropic direction (or the dip distribution angular_pdf). A
    surface of 1 cm^2 cuts through 1 / side^2 cubes of 1 mg (side = (1e-3 / rho)^(1/3)), i.e. 1 / side^2 tracks per
    side of depth for 1 track per mg. A track of length L and dip d is etched if its top lies between -L sin(d) (it
    crosses the original surface) and the removed thickness h, a fraction (h + L sin d) / side of the tracks; its
    pit has the axes of etch_pit_axes. Every etched pit is counted: which ones the microscope resolves is the
    observation step (the width axis is kept for it).

    Monte Carlo with importance sampling: n_per_bin tracks per latent bin (lengths uniform in the bin; dips; tops
    uniform in the etched interval only), each weighted by the fraction above, so every sample contributes. The
    same samples serve every spectrum, so differences between spectra carry no Monte Carlo noise of their own;
    seed fixes them (reproducible results).

    Args:
        sample_density_g_cm3 (float): Sample density [g/cm^3].
        x_bins (np.ndarray, optional): Latent track-length bins [nm]. Defaults to DEFAULT_INTEGRATION_X_BINS_NM.
        out_bins (np.ndarray, optional): Measured-size bins [nm]. Defaults to x_bins.
        etch_type (str, optional): 'HF' or 'plasma': the defaults of removed_nm, v_ratio and lateral_ratio
            (ETCH_DEFAULTS; wet HF isotropic, dry plasma directional). Defaults to "HF".
        removed_nm (float, optional): Bulk layer removed, h = v_B * t [nm]. Default from etch_type.
        v_ratio (float, optional): V = v_T / v_B. Default from etch_type.
        lateral_ratio (float, optional): See etch_pit_axes (1 isotropic, 0 directional). Default from etch_type.
        angular_pdf (np.ndarray, optional): Relative probability of the dip angle on a uniform grid from 0 to pi/2
            (len(angular_pdf) points). Defaults to None (isotropic: sin(dip) uniform).
        measure (str, optional): Measured size: 'major' (what OptimusPrimus reports as length) or 'minor'.
            Defaults to 'major'.
        width_bins (np.ndarray, optional): Pit-width bins [nm]. Defaults to ETCH_WIDTH_BINS_NM.
        n_per_bin (int, optional): Monte Carlo tracks per latent bin. Defaults to ETCH_SAMPLES_PER_BIN.
        seed (int, optional): Seed of the Monte Carlo. Defaults to 0.

    Returns:
        EtchResponse: The response.

    Raises:
        ValueError: If etch_type or measure is unknown.
    """
    if measure not in ('major', 'minor'):
        raise ValueError("measure must be 'major' or 'minor'")
    etch = _etch_parameters(etch_type, removed_nm, v_ratio, lateral_ratio)
    x_bins = np.asarray(_resolve_x_bins(x_bins), float)
    out_bins = x_bins if out_bins is None else np.asarray(out_bins, float)
    width_bins = ETCH_WIDTH_BINS_NM if width_bins is None else np.asarray(width_bins, float)
    pdf = None if angular_pdf is None else np.asarray(angular_pdf, float)
    key = (float(sample_density_g_cm3), x_bins.tobytes(), out_bins.tobytes(), width_bins.tobytes(),
           tuple(sorted(etch.items())), None if pdf is None else pdf.tobytes(), measure, int(n_per_bin), seed)
    if key in _ETCH_RESPONSES:
        return _ETCH_RESPONSES[key]

    rng = np.random.default_rng(seed)
    side_cm = (1.e-3 / sample_density_g_cm3) ** (1. / 3.)
    side_nm = side_cm * 1.e7
    n_in, n_out, n_w = len(x_bins) - 1, len(out_bins) - 1, len(width_bins) - 1
    j = np.repeat(np.arange(n_in), int(n_per_bin))
    L = rng.uniform(x_bins[j], x_bins[j + 1])
    if pdf is not None:
        grid = np.linspace(0., np.pi / 2., len(pdf))
        dip = rng.choice(grid, size=j.size, p=pdf / pdf.sum())
    else:
        dip = np.arcsin(rng.uniform(0., 1., size=j.size))
    z_lo = np.maximum(-L * np.sin(dip), -side_nm / 2.)
    z_hi = np.minimum(etch['removed_nm'], side_nm / 2.)
    weight = np.clip(z_hi - z_lo, 0., None) / side_nm / side_cm ** 2 / n_per_bin
    z_top = rng.uniform(z_lo, np.maximum(z_hi, z_lo))

    major, minor = etch_pit_axes(L, dip, z_top, etch['removed_nm'], etch['v_ratio'], etch['lateral_ratio'])
    size = major if measure == 'major' else minor
    i_out = np.searchsorted(out_bins, size, side='right') - 1
    keep = np.isfinite(minor) & (i_out >= 0) & (i_out < n_out)
    i_w = np.clip(np.searchsorted(width_bins, minor[keep], side='right') - 1, 0, n_w - 1)
    flat = (i_out[keep] * n_w + i_w) * n_in + j[keep]
    matrix = np.bincount(flat, weights=weight[keep], minlength=n_out * n_w * n_in).reshape(n_out, n_w, n_in)
    matrix = matrix.astype(np.float32)
    response = EtchResponse(matrix, x_bins, out_bins, width_bins,
                            {**etch, 'etch_type': etch_type, 'sample_density_g_cm3': sample_density_g_cm3,
                             'measure': measure, 'n_per_bin': int(n_per_bin), 'seed': seed})
    _ETCH_RESPONSES[key] = response
    return response


def slice_spectrum_etch(counts_mg, sample_area_cm2, sample_density_g_cm3, removed_nm=None, v_ratio=None,
                        lateral_ratio=None, angular_pdf=None, measure='major', x_bins=None, out_bins=None,
                        etch_type="HF", realisation=False, rng=None):
    """Expected etched-pit size spectrum on the analysed surface, from a latent track-length spectrum: every etched
    pit (the microscope resolution is the later observation step).

    Applies the etching response of the sample (etch_response, built once and kept for the session; see there for
    the model and the arguments). The latent spectrum and the measured sizes can use different binnings: x_bins
    describes counts_mg (e.g. the wide bins of the theoretical spectrum) and out_bins the returned histogram (e.g. the
    bins of the descriptor).

    Args:
        counts_mg (np.ndarray): Latent track spectrum, tracks per mg in each bin of x_bins.
        sample_area_cm2, sample_density_g_cm3 (float): Analysed area [cm^2] and sample density [g/cm^3].
        removed_nm, v_ratio, lateral_ratio, angular_pdf, measure, x_bins, out_bins, etch_type: See etch_response.
        realisation (bool, optional): Return one Poisson realisation (integer counts) instead of the expectation.
            Defaults to False.
        rng (np.random.Generator, optional): Generator of the realisation. Defaults to None (a new one).

    Returns:
        np.ndarray: Expected number of pits per bin of out_bins on the analysed area (or one realisation).

    Raises:
        ValueError: If etch_type or measure is unknown, or counts_mg does not match x_bins.
    """
    response = etch_response(sample_density_g_cm3, x_bins=x_bins, out_bins=out_bins, etch_type=etch_type,
                             removed_nm=removed_nm, v_ratio=v_ratio, lateral_ratio=lateral_ratio,
                             angular_pdf=angular_pdf, measure=measure)
    expected = response.pits(counts_mg, sample_area_cm2)
    return (rng or np.random.default_rng()).poisson(expected) if realisation else expected


def compare_scenarios(spectra, reference, sample_area_cm2, sample_density_g_cm3, size_range_nm=None, **etch):
    """Expected etched pits of several scenarios of one sample, and the significance of each one's difference from a
    reference scenario.

    All spectra go through the same etching response (etch_response), so the differences carry no Monte Carlo
    noise. The significance of a scenario is (N - N_ref) / sqrt(N_ref), N the expected pits with measured size in
    size_range_nm: Poisson error of the reference counts only (no systematics, no uncertainty of the age or of the
    model). A range of long pits often separates scenarios better than the total, where the short pits dominate.

    Args:
        spectra (dict): {label: latent tracks per mg in each bin of x_bins} (the whole spectrum of the sample:
            cosmic rays, neutrinos, radiogenic...).
        reference (str): Label of the reference scenario.
        sample_area_cm2, sample_density_g_cm3 (float): Analysed area [cm^2] and sample density [g/cm^3].
        size_range_nm (tuple, optional): (min, max) measured size counted [nm]; bins are counted if their centre is
            inside. Defaults to None (all).
        **etch: Arguments of etch_response (x_bins, out_bins, etch_type, removed_nm, ...).

    Returns:
        dict: {label: {'pits': expected pits per size bin, 'n': pits in the range, 'excess': n - n_ref,
        'significance': excess / sqrt(n_ref)}} (the reference included, with excess 0); and 'out_bins'.

    Raises:
        KeyError: If reference is not in spectra.
    """
    response = etch_response(sample_density_g_cm3, **etch)
    centres = 0.5 * (response.out_bins[1:] + response.out_bins[:-1])
    lo, hi = (-np.inf, np.inf) if size_range_nm is None else size_range_nm
    inside = (centres >= lo) & (centres <= hi)
    pits = {label: response.pits(counts, sample_area_cm2) for label, counts in spectra.items()}
    n_ref = float(np.sum(pits[reference][inside]))
    out = {'out_bins': response.out_bins}
    for label, p in pits.items():
        n = float(np.sum(p[inside]))
        out[label] = {'pits': p, 'n': n, 'excess': n - n_ref,
                      'significance': (n - n_ref) / np.sqrt(n_ref) if n_ref > 0 else np.nan}
    return out
