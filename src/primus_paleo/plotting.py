"""Plots: track-length (or recoil-energy) spectra split by track group, and diagnostics of a Paleodetector.

Spectra split by track group (heavy, medium_heavy, medium_light, light): plot_track_groups.
Diagnostics of the inputs of a detector: interaction_profiles (interaction depth profiles of its StdRock depth
tables) and plot_secondary_production (where and by whom the atmospheric secondaries are made).

plot_track_groups:

Each curve (a scenario, a species, a sample...) gets one colour family; its track groups are shades of that family,
darkest for the heaviest group. Three layouts, chosen by `style`:

  'stack'   one curve: the groups stacked as filled areas, the total as an outline. Shows the composition of a
            spectrum at a glance (which group makes the tracks at each length).
  'lines'   a few curves (2-3): every group of every curve as a line, shade = group, colour family = curve, the
            total as a thick line. Compares curves group by group in one panel.
  'panels'  many curves: one panel per group (plus one for the total), every curve in its own colour in each panel.
            The clearest when the groups differ by orders of magnitude or more than three curves are compared.

'auto' picks 'stack' for one curve, 'lines' for two or three, 'panels' beyond.

Input: the dicts returned with nucleus='groups', e.g.

    x_bins, cr = det.integrate_all_particles(nucleus='groups')
    _, nu = det.integrate_atmospheric_neutrinos(x_bins=x_bins, nucleus='groups')
    _, rad = det.integrate_radiogenic(x_bins=x_bins, nucleus='groups')
    plot_track_groups({'cosmic rays': cr['total'], 'neutrinos': nu['total'],
                       'radiogenic n': rad['fission_n'], 'fission': rad['fission_fragments']}, x_bins)

Recoil-energy spectra (observable='recoil_energy' in the integrate_* functions) are plotted the same way with
observable='recoil_energy' here, which sets the axis labels and a logarithmic energy axis.
"""

import numpy as np

from ._core import (MUON_SPECIES, SECONDARY_SOURCES, TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS, TRACK_GROUPS,
                    _depth_cdf_rows)

__all__ = ['plot_track_groups', 'group_colors', 'GROUP_LABELS', 'GROUP_SHADES', 'DEFAULT_COLORMAPS',
           'interaction_profiles', 'plot_secondary_production']

GROUP_LABELS = {
    'heavy': 'heavy',
    'medium_heavy': 'medium heavy',
    'medium_light': 'medium light',
    'light': 'light (H, He)',
    TOTAL_NO_LIGHT_IONS: 'total (no H, He)',
    TOTAL_ALL_FRAGMENTS: 'total',
}
# position in the colour map of each group (1 = darkest end of a sequential map)
GROUP_SHADES = {'heavy': 0.95, 'medium_heavy': 0.75, 'medium_light': 0.55, 'light': 0.35}
TOTAL_SHADE = 1.0
GROUP_LINESTYLES = {'heavy': '-', 'medium_heavy': '-', 'medium_light': '--', 'light': ':'}
# sequential matplotlib colour maps, one family per curve
DEFAULT_COLORMAPS = ('Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'PuRd')
# default axes of each observable of the integrate_* functions
_AXES = {
    'tracks': {'xlabel': 'Track length [nm]', 'counts': 'Tracks', 'unit': 'nm', 'logx': False},
    'recoil_energy': {'xlabel': 'Recoil energy [MeV]', 'counts': 'Recoils', 'unit': 'MeV', 'logx': True},
}


def group_colors(cmap, groups=TRACK_GROUPS):
    """Shades of one colour family for the track groups (and the total).

    Args:
        cmap (str or matplotlib colormap): A sequential colour map, e.g. 'Blues'.
        groups (tuple[str], optional): Groups to colour. Defaults to TRACK_GROUPS.

    Returns:
        dict: group -> RGBA colour, plus 'total' for the total.
    """
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = {g: cm(GROUP_SHADES.get(g, 0.6)) for g in groups}
    colors['total'] = cm(TOTAL_SHADE)
    return colors


def _as_curves(curves):
    """{label: group dict}: accepts that, a list of (label, group dict), or a single group dict."""
    if isinstance(curves, dict) and any(g in curves for g in TRACK_GROUPS):
        return {'': curves}
    if isinstance(curves, dict):
        return dict(curves)
    return dict(curves)


def plot_track_groups(curves, x_bins, style='auto', groups=TRACK_GROUPS, total=TOTAL_NO_LIGHT_IONS, ax=None,
                      per_nm=False, logx=None, logy=None, colormaps=DEFAULT_COLORMAPS, show_light=True,
                      xlim='auto', y_decades=None, xlabel=None, ylabel=None, legend=True, observable='tracks'):
    """Plots track-length (or recoil-energy) spectra split by track group.

    Args:
        curves (dict): {label: dict with the track groups and the totals} (nucleus='groups' output), or one such
            dict alone.
        x_bins (np.ndarray): Bin edges of the spectra [nm, or MeV for recoil energies].
        style (str, optional): 'auto', 'stack', 'lines' or 'panels' (see the module docstring). Defaults to 'auto'.
        groups (tuple[str], optional): Groups to show, heaviest first. Defaults to TRACK_GROUPS.
        total (str or None, optional): Total drawn with the groups: 'total_no_light_ions', 'total' or None.
            Defaults to 'total_no_light_ions'.
        ax (matplotlib Axes, optional): Axes for 'stack' and 'lines' (a new figure if None). Ignored by 'panels'.
        per_nm (bool, optional): Plot counts per unit of the x axis (counts / bin width: per nm, or per MeV for
            recoil energies) instead of counts per bin. Defaults to False.
        logx (bool, optional): Logarithmic x axis. Defaults to None: False for tracks, True for recoil energies.
        logy (bool, optional): Logarithmic y axis. Defaults to None: False for 'stack', True otherwise.
        colormaps (tuple, optional): Colour families of the curves, in order. Defaults to DEFAULT_COLORMAPS.
        show_light (bool, optional): Include the 'light' group. Defaults to True.
        xlim (tuple or str or None, optional): x range; 'auto' ends it a little after the last bin above 1e-6 of
            the maximum of any curve; None uses the whole x_bins. Defaults to 'auto'.
        y_decades (float or None, optional): With a log y axis, limit the y range of each axes to this many
            decades below its maximum (e.g. 6, to hide far tails). Defaults to None (matplotlib's range, which
            keeps curves many orders of magnitude apart visible).
        xlabel (str, optional): x-axis label. Defaults to the track length or the recoil energy.
        ylabel (str, optional): y-axis label. Defaults to tracks (or recoils) per mg per bin (or per nm, per MeV).
        legend (bool, optional): Draw the legend(s). Defaults to True.
        observable (str, optional): 'tracks' or 'recoil_energy', as given to the integrate_* function; sets the
            default labels and x scale. Defaults to 'tracks'.

    Returns:
        tuple: (fig, axes), axes being one Axes or, for 'panels', an array of them.

    Raises:
        ValueError: If style or observable is unknown, or a curve lacks a group.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    curves = _as_curves(curves)
    x_bins = np.asarray(x_bins, dtype=float)
    groups = tuple(g for g in groups if show_light or g != 'light')
    for label, c in curves.items():
        missing = [g for g in (*groups, *([total] if total else [])) if g not in c]
        if missing:
            raise ValueError(f"curve {label!r} has no {missing}: compute it with nucleus='groups'.")
    if style == 'auto':
        style = 'stack' if len(curves) == 1 else 'lines' if len(curves) <= 3 else 'panels'
    if style not in ('stack', 'lines', 'panels'):
        raise ValueError(f"style must be 'auto', 'stack', 'lines' or 'panels', got {style!r}.")
    if logy is None:
        logy = style != 'stack'
    if observable not in _AXES:
        raise ValueError(f"observable must be one of {tuple(_AXES)}, got {observable!r}.")
    axis = _AXES[observable]
    if logx is None:
        logx = axis['logx']
    if xlabel is None:
        xlabel = axis['xlabel']
    widths = np.diff(x_bins)
    scale = (lambda v: np.asarray(v, dtype=float) / widths) if per_nm else (lambda v: np.asarray(v, dtype=float))
    if ylabel is None:
        ylabel = f"{axis['counts']} / mg / {axis['unit'] if per_nm else 'bin'}"
    families = {label: group_colors(colormaps[i % len(colormaps)], groups) for i, label in enumerate(curves)}

    x_lo = x_bins[0] if not logx else x_bins[x_bins > 0][0]
    x_hi = x_bins[-1]
    if isinstance(xlim, str) and xlim == 'auto':
        peak = max(float(np.max(scale(c[k]))) for c in curves.values() for k in (*groups, *([total] if total else [])))
        above = [np.nonzero(scale(c[k]) > 1e-6 * peak)[0] for c in curves.values()
                 for k in (*groups, *([total] if total else []))]
        last = max((int(i[-1]) for i in above if i.size), default=len(widths) - 1)
        x_hi = min(x_bins[-1], x_bins[last + 1] * (1.5 if logx else 1.0) + (0.0 if logx else 0.1 * (x_bins[last + 1] - x_lo)))
    elif xlim is not None:
        x_lo, x_hi = xlim

    def finish(a):
        a.set_xscale('log' if logx else 'linear')
        a.set_yscale('log' if logy else 'linear')
        a.set_xlim(x_lo, x_hi)
        if logy and y_decades:
            top = max((float(np.max(line.get_data()[0])) for line in a.patches if hasattr(line, 'get_data')),
                      default=None)
            if top and top > 0:
                a.set_ylim(top * 10 ** (-y_decades), top * 2)

    if style == 'stack':
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
        for label, c in curves.items():
            colors = families[label]
            bottom = np.zeros(len(widths))
            for g in groups:                                   # heaviest at the bottom
                top = bottom + scale(c[g])
                ax.stairs(top, x_bins, baseline=bottom, fill=True, color=colors[g], linewidth=0,
                          label=f"{label} {GROUP_LABELS[g]}".strip())
                bottom = top
            if total:
                ax.stairs(scale(c[total]), x_bins, color=colors['total'], linewidth=1.2,
                          label=f"{label} {GROUP_LABELS[total]}".strip())
        finish(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(frameon=False, fontsize=8)
        return fig, ax

    if style == 'lines':
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
        for label, c in curves.items():
            colors = families[label]
            for g in groups:
                ax.stairs(scale(c[g]), x_bins, color=colors[g], linestyle=GROUP_LINESTYLES.get(g, '-'),
                          linewidth=1.2)
            if total:
                ax.stairs(scale(c[total]), x_bins, color=colors['total'], linewidth=2.4)
        finish(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            # two legends: colour family = curve, shade and line style = group
            curve_handles = [Patch(color=families[label][groups[0]], label=label or 'spectrum') for label in curves]
            grey = group_colors('Greys', groups)
            group_handles = [Line2D([], [], color=grey[g], linestyle=GROUP_LINESTYLES.get(g, '-'), linewidth=1.2,
                                    label=GROUP_LABELS[g]) for g in groups]
            if total:
                group_handles.insert(0, Line2D([], [], color=grey['total'], linewidth=2.4,
                                               label=GROUP_LABELS[total]))
            first = ax.legend(handles=curve_handles, frameon=False, fontsize=8, loc='upper left',
                              bbox_to_anchor=(1.01, 1.0))
            ax.add_artist(first)
            ax.legend(handles=group_handles, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(1.01, 0.0))
        return fig, ax

    shown = ([total] if total else []) + list(groups)
    n = len(shown)
    ncols = 2 if n > 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), sharex=True, squeeze=False)
    for k, (a, g) in enumerate(zip(axes.flat, shown)):
        for label, c in curves.items():
            color = families[label]['total'] if g == total else families[label][groups[0]]
            a.stairs(scale(c[g]), x_bins, color=color, linewidth=1.4, label=label or None)
        a.set_title(GROUP_LABELS.get(g, g), fontsize=10)
        finish(a)
        if k % ncols == 0:
            a.set_ylabel(ylabel)
    for a in axes.flat[n:]:
        a.set_visible(False)
    for col in range(ncols):                                     # x labels on the lowest visible panel of each column
        visible = [axes[row, col] for row in range(nrows) if row * ncols + col < n]
        if visible:
            visible[-1].set_xlabel(xlabel)
            visible[-1].xaxis.set_tick_params(labelbottom=True)
    if legend and any(curves):
        axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


# --- Diagnostics of a Paleodetector ----------------------------------------------------------------------------------
G_CM2_PER_MWE = 100.     # rock column of 1 m.w.e. [g/cm^2]


def interaction_profiles(detector, energies_gev, z_edges_mwe, species=('neutron', 'proton', 'mu-'), capture=True):
    """Interaction depth profiles from the StdRock depth tables, as used by the integrations (a diagnostic: returns
    the profiles, to be plotted against z_edges_mwe).

    The same distributions as Paleodetector._slab_interaction_probabilities, on depth bins instead of the sample
    slab (both from _depth_cdf_rows on the tables of Paleodetector._depth_table):

    * Hadrons: dP/dz of the interaction depth, F(z_hi) - F(z_lo) per bin (exponential tail included).
    * Muons: the in-flight interactions, (G(z_hi) - G(z_lo)) / E[R] per bin, i.e. an interaction density
      proportional to the survival probability P(R > z); with capture=True, plus the stops, F(z_hi) - F(z_lo) (the
      mu- capture peak; for mu+ the stop is a decay, and the detector does not count it). mu+ uses the mu- table.

    Args:
        detector (Paleodetector): Any detector (the tables come from the StdRock runs, not from its mineral).
        energies_gev (array_like): Primary energies [GeV].
        z_edges_mwe (np.ndarray): Depth bin edges along the path [m.w.e.].
        species (tuple[str], optional): Any of 'neutron', 'proton', 'mu-', 'mu+' (with a depth table). Defaults
            to ('neutron', 'proton', 'mu-').
        capture (bool, optional): Add the muon stops to the in-flight interactions. Defaults to True.

    Returns:
        dict: species -> dP/dz [1/m.w.e.], shape (n_depth_bins, n_energies).
    """
    E = np.atleast_1d(np.asarray(energies_gev, float))
    z_edges_mwe = np.asarray(z_edges_mwe, float)
    z_lo = np.broadcast_to(z_edges_mwe[:-1, None], (len(z_edges_mwe) - 1, E.size))
    z_hi = np.broadcast_to(z_edges_mwe[1:, None], z_lo.shape)
    dz = np.diff(z_edges_mwe)[:, None]
    out = {}
    for s in species:
        q_rows, tails = detector._depth_rows(s, E)
        p = detector._depth_table(s)['p']
        F_lo, G_lo = _depth_cdf_rows(z_lo, q_rows, p, tails)
        F_hi, G_hi = _depth_cdf_rows(z_hi, q_rows, p, tails)
        if s in MUON_SPECIES:
            mean_range = _depth_cdf_rows(np.full(E.size, np.inf), q_rows, p)[1]
            prob = (G_hi - G_lo) / mean_range + ((F_hi - F_lo) if capture else 0.0)
        else:
            prob = F_hi - F_lo
        out[s] = prob / dz
    return out

def plot_secondary_production(detector, sources=None, products=('neutron', 'proton'),
                              total_simulated_particles=1e4, product_bins_gev=None, z_edges_mwe=None,
                              example_species='mu-', example_energy_gev=None, t_kyr=None):
    """Overview of the secondary neutrons and protons produced in the StdRock runs, in three panels.

    Purpose: a diagnostic of the input of the atmospheric secondary cascade (Paleodetector._secondary_cascade),
    straight from the StdRock rows (_load_stdrock_rows), to see which primaries make the secondaries, where and with
    what energy.
        (a) Yield: secondaries per primary vs primary energy, one line per source and product.
        (b) Where and with what energy: production depth vs secondary energy (neutrons + protons) for one run.
        (c) Who makes them at each depth: production rate per gram of rock vs depth, each run weighted by the flux
            integrated over its energy cell (flux history at t_kyr; _flux_cell_integral, per cm^2 per s), divided by
            the rock column of each depth bin (1 m.w.e. = 100 g/cm^2) to give a rate per gram. Every primary is
            taken as vertical, so the panel shows shapes and crossover depths; the integrations also spread the
            primaries over zenith angle.
    Yields only count products made inside the StdRock target (a lower limit for muons leaving it).

    Args:
        detector (Paleodetector): Any detector (the StdRock runs are shared; its flux history weights panel (c)).
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

    sources = [src for src in (SECONDARY_SOURCES if sources is None else sources) if detector._stdrock_dir(src)]
    if not sources:
        raise ValueError("No StdRock runs for any of the requested sources.")
    product_bins_gev = np.logspace(-3, 1, 41) if product_bins_gev is None else np.asarray(product_bins_gev, float)
    rows = {(src, q): detector._load_stdrock_rows(src, q, product_bins_gev, total_simulated_particles)
            for src in sources for q in products}
    if z_edges_mwe is None:
        z_max = max(r['target_mwe'] for r in rows.values())
        z_edges_mwe = np.logspace(-2, np.log10(z_max), 120)
    z_edges_mwe = np.asarray(z_edges_mwe, float)
    z_mid = np.sqrt(np.maximum(z_edges_mwe[1:] * z_edges_mwe[:-1], 1e-300))
    styles = ('-', '--', ':', '-.')

    fig, ax = plt.subplots(1, 3, figsize=[17, 4.6])

    for k, src in enumerate(sources):
        for q, ls in zip(products, styles):
            r = rows[(src, q)]
            e0 = r['primary_bins'][:-1]
            y = np.bincount(r['primary_cell'], minlength=e0.size) * r['weight']
            ax[0].loglog(e0, np.where(y > 0, y, np.nan), ls, color=f"C{k}", label=f"{src} \u2192 {q}")
    ax[0].set(xlabel="Primary energy (GeV)", ylabel="Secondaries per primary", title="(a) Yield")
    ax[0].legend(fontsize='x-small', ncol=2)

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

    t_now = np.array([detector.total_age_kyr if t_kyr is None else float(t_kyr)])
    for k, src in enumerate(sources):
        for q, ls in zip(products, styles):
            r = rows[(src, q)]
            phi = detector._flux_cell_integral(src, t_now, r['primary_bins'])[0]
            w = phi[r['primary_cell']] * r['weight']
            rate = np.histogram(r['depth_mwe'], bins=z_edges_mwe, weights=w)[0] / (np.diff(z_edges_mwe) * G_CM2_PER_MWE)
            ax[2].loglog(z_mid, np.where(rate > 0, rate, np.nan), ls, color=f"C{k}", label=f"{src} \u2192 {q}")
    ax[2].set(xlabel="Depth (m.w.e.)", ylabel=r"Production rate (g$^{-1}$ s$^{-1}$)",
                title="(c) Who makes the secondaries")
    ax[2].legend(fontsize='x-small', ncol=2)
    fig.tight_layout()
    return fig, ax
