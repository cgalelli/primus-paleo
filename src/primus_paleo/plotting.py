"""Plots of track-length (or recoil-energy) spectra split by track group (heavy, medium_heavy, medium_light, light).

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

from ._core import TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS, TRACK_GROUPS

__all__ = ['plot_track_groups', 'group_colors', 'GROUP_LABELS', 'GROUP_SHADES', 'DEFAULT_COLORMAPS']

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
