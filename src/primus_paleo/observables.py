"""What is measured and how to look at it: etch-pit openings and sliced/etched spectra of latent tracks, rebinning,
and diagnostic profiles and plots of a Paleodetector's depth tables and secondary production."""

import numpy as np

from ._core import (
    MUON_SPECIES, SECONDARY_SOURCES, _resolve_x_bins,
)

__all__ = [
    'interaction_profiles', 'plot_secondary_production', 'etch_pit_axes', 'rebin_counts', 'ETCH_DEFAULTS',
    'slice_spectrum_etch',
]

def interaction_profiles(mineral, energies_gev, z_edges_mwe, species=('neutron', 'proton', 'mu-'), capture=True):
    """Interaction depth profiles from the StdRock depth tables, as used by the integrations.

    Hadrons: dP/dz of the interaction depth, F(z_hi) - F(z_lo) per bin (exponential tail included).
    Muons:   in-flight interactions only (no capture), (G(z_hi) - G(z_lo)) / E[R] per bin, i.e. interaction
             density proportional to the survival probability P(R > z); mu+ uses the same mu- table.

    Args:
        mineral (Paleodetector): any detector of the mineral (the tables come from StdRock, not from the mineral).
        energies_gev (array_like): primary energies [GeV].
        z_edges_mwe (np.ndarray): depth bin edges along the path [m.w.e.].
        species (tuple[str]): any of 'neutron', 'proton', 'mu-', 'mu+' (with a depth table).

    Returns:
        dict: species -> dP/dz [1/m.w.e.], shape (n_depth_bins, n_energies).
    """
    E = np.atleast_1d(np.asarray(energies_gev, float))
    z_lo = np.broadcast_to(z_edges_mwe[:-1, None], (len(z_edges_mwe) - 1, E.size))
    z_hi = np.broadcast_to(z_edges_mwe[1:, None], z_lo.shape)
    dz = np.diff(z_edges_mwe)[:, None]
    out = {}
    for s in species:
        if s in MUON_SPECIES:
            prob, _capture = mineral._muon_depth_probabilities(E, z_lo, z_hi)   # in flight; _capture = stopping peak
            if capture:
                prob += _capture
        else:
            prob = mineral._hadron_depth_probability(E, z_lo, z_hi, s)
        out[s] = prob / dz
    return out

def plot_secondary_production(self, sources=None, products=('neutron', 'proton'),
                                total_simulated_particles=1e4, product_bins_gev=None, z_edges_mwe=None,
                                example_species='mu-', example_energy_gev=None, t_kyr=None):
    """Overview of the secondary neutrons and protons produced in the StdRock runs, in three panels.

    Purpose: a diagnostic of the input of the atmospheric secondary cascade (_secondary_yield), straight from the
    StdRock rows (_load_stdrock_rows), to see which primaries make the secondaries, where and with what energy.
        (a) Yield: secondaries per primary vs primary energy, one line per source and product.
        (b) Where and with what energy: production depth vs secondary energy (neutrons + protons) for one run.
        (c) Who makes them at each depth: production rate per gram of rock vs depth, each run weighted by the flux
            integrated over its energy cell (flux history at t_kyr). Every primary is taken as vertical, so the
            panel shows shapes and crossover depths; the integrations also spread the primaries over zenith angle.
    Yields only count products made inside the StdRock target (a lower limit for muons leaving it).

    Args:
        sources (list[str], optional): Primary species. Defaults to None (SECONDARY_SOURCES with StdRock runs;
            mu+ falls back on the mu- runs without captures, see _stdrock_dir).
        products (tuple[str], optional): Secondaries shown. Defaults to ('neutron', 'proton').
        total_simulated_particles (float, optional): Primaries per StdRock run. Defaults to 1e4.
        product_bins_gev (np.ndarray, optional): Secondary energy bins [GeV]. Defaults to 40 log bins, 1 MeV-10 GeV.
        z_edges_mwe (np.ndarray, optional): Depth bins [m.w.e.]. Defaults to 120 log bins from 0.01 m.w.e. to the
            longest StdRock target.
        example_species (str, optional): Source shown in panel (b). Defaults to 'mu-'.
        example_energy_gev (float, optional): Primary energy of panel (b); the run whose cell contains it is used.
            Defaults to None (the highest-energy run).
        t_kyr (float, optional): Local time of the flux used in panel (c) [kyr]. Defaults to None (present).

    Returns:
        tuple: (figure, array of the three axes).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    sources = [src for src in (SECONDARY_SOURCES if sources is None else sources) if self._stdrock_dir(src)]
    if not sources:
        raise ValueError("No StdRock runs for any of the requested sources.")
    product_bins_gev = np.logspace(-3, 1, 41) if product_bins_gev is None else np.asarray(product_bins_gev, float)
    rows = {(src, q): self._load_stdrock_rows(src, q, product_bins_gev, total_simulated_particles)
            for src in sources for q in products}
    if z_edges_mwe is None:
        z_max = max(r['target_mwe'] for r in rows.values())
        z_edges_mwe = np.logspace(-2, np.log10(z_max), 120)
    z_edges_mwe = np.asarray(z_edges_mwe, float)
    z_mid = np.sqrt(np.maximum(z_edges_mwe[1:] * z_edges_mwe[:-1], 1e-300))
    styles = ('-', '--', ':', '-.')

    fig, ax = plt.subplots(1, 3, figsize=[17, 4.6])

    # (a) yield per primary vs primary energy
    for k, src in enumerate(sources):
        for q, ls in zip(products, styles):
            r = rows[(src, q)]
            e0 = r['primary_bins'][:-1]
            y = np.bincount(r['primary_cell'], minlength=e0.size) * r['weight']
            ax[0].loglog(e0, np.where(y > 0, y, np.nan), ls, color=f"C{k}", label=f"{src} \u2192 {q}")
    ax[0].set(xlabel="Primary energy (GeV)", ylabel="Secondaries per primary", title="(a) Yield")
    ax[0].legend(fontsize='x-small', ncol=2)

    # (b) production depth x secondary energy for one run
    if example_species not in sources:
        example_species = sources[0]
    pb = rows[(example_species, products[0])]['primary_bins']
    i0 = (len(pb) - 2 if example_energy_gev is None
            else int(np.clip(np.searchsorted(pb, example_energy_gev, side='right') - 1, 0, len(pb) - 2)))
    H = np.zeros((len(z_edges_mwe) - 1, len(product_bins_gev) - 1))
    for q in products:
        r = rows[(example_species, q)]
        sel = r['primary_cell'] == i0
        H += np.histogram2d(r['depth_mwe'][sel], product_bins_gev[r['cell'][sel]],
                            bins=[z_edges_mwe, product_bins_gev])[0] * r['weight']
    dens = H / np.diff(z_edges_mwe)[:, None] / np.diff(np.log10(product_bins_gev))[None, :]
    if np.any(dens > 0):
        m = ax[1].pcolormesh(z_edges_mwe, product_bins_gev, np.where(dens > 0, dens, np.nan).T,
                                norm=LogNorm(), shading='flat')
        fig.colorbar(m, ax=ax[1], label="per primary per m.w.e. per decade")
    ax[1].set(xscale='log', yscale='log', xlabel="Production depth (m.w.e.)", ylabel="Secondary energy (GeV)",
                title=f"(b) {' + '.join(products)} from {example_species} at {pb[i0]:.3g} GeV")

    # (c) flux-weighted production rate vs depth
    t_now = np.array([self.total_age_kyr if t_kyr is None else float(t_kyr)])
    for k, src in enumerate(sources):
        for q, ls in zip(products, styles):
            r = rows[(src, q)]
            phi = self._flux_cell_integral(src, t_now, r['primary_bins'])[0]          # cm^-2 s^-1 per cell
            w = phi[r['primary_cell']] * r['weight']
            rate = np.histogram(r['depth_mwe'], bins=z_edges_mwe, weights=w)[0] / (np.diff(z_edges_mwe) * 100.)
            ax[2].loglog(z_mid, np.where(rate > 0, rate, np.nan), ls, color=f"C{k}", label=f"{src} \u2192 {q}")
    ax[2].set(xlabel="Depth (m.w.e.)", ylabel=r"Production rate (g$^{-1}$ s$^{-1}$)",
                title="(c) Who makes the secondaries")
    ax[2].legend(fontsize='x-small', ncol=2)
    fig.tight_layout()
    return fig, ax

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
