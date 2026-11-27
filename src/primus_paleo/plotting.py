"""Plots: track-length (or recoil-energy) spectra split into components, the history of a sample, and diagnostics of
a Paleodetector.

Spectra split into components (track groups by default; nuclides, species or sources with keys=...):
plot_track_groups. History: plot_flux_history (a FluxHistory against time: events, excursions) and
plot_exposure_timeline (a detector's overburden and flux over its exposure, with the exposure windows), to check
the timing of a scenario before integrating. Diagnostics of the inputs of a detector: interaction_profiles
(interaction depth profiles of its StdRock depth tables) and plot_secondary_production (where and by whom the
atmospheric secondaries are made).

plot_track_groups
-----------------
Each curve (a scenario, a species, a sample...) gets one colour family; its components are shades of that family,
darkest for the first (for the track groups: the heaviest). Three layouts, chosen by `style`:

  'stack'   one curve: the components stacked as filled areas, the total as an outline. Shows the composition of a
            spectrum at a glance (which component makes the tracks at each length).
  'lines'   a few curves (2-3): every component of every curve as a line, shade (and line style) = component, colour
            family = curve, the total as a thick line. Compares curves component by component in one panel.
  'panels'  many curves: one panel per component (plus one for the total), every curve in its own colour in each
            panel. The clearest when the components differ by orders of magnitude or more than three curves are
            compared.

'auto' picks 'stack' for one curve, 'lines' for two or three, 'panels' beyond. With a single curve, 'lines' gives
each component its own colour (one legend), as for a breakdown by species or source.

Comparing scenarios (ratio_to='<label of a curve>'): every other curve is drawn divided by that one, component by
component, on a linear axis around 1 ('lines' or 'panels'); a few-percent effect, invisible on a log axis of counts,
is read directly. Bins where the reference is below RATIO_MIN_FRACTION of its maximum are left out.

The x axis is logarithmic by default when the spectrum spans more than LOGX_SPAN (track lengths from the first bin
to the end of the tails), so that the short tracks, where most counts are, and the long tails both show.

Input: the dicts of the integrate_* functions, e.g. with nucleus='groups'

    x_bins, cr = det.integrate_all_particles(nucleus='groups')
    _, nu = det.integrate_atmospheric_neutrinos(x_bins=x_bins, nucleus='groups')
    _, rad = det.integrate_radiogenic(x_bins=x_bins, nucleus='groups')
    plot_track_groups({'cosmic rays': cr['total'], 'neutrinos': nu['total'],
                       'radiogenic n': rad['fission_n'], 'fission': rad['fission_fragments']}, x_bins)

or any other breakdown, through keys: the species of one run (keys=['mu-', 'neutron', ...], total='total', on the
whole output dict of integrate_all_particles with the default nucleus) or the nuclides (nucleus='all',
keys=['Na23', 'Cl35', ...]). Recoil-energy spectra (observable='recoil_energy' in the integrate_* functions) are
plotted the same way with observable='recoil_energy' here, which sets the axis labels and a logarithmic energy axis.
"""

import numpy as np

from ._core import (MUON_SPECIES, SECONDARY_SOURCES, TOTAL_ALL_FRAGMENTS, TOTAL_NO_LIGHT_IONS, TRACK_GROUPS,
                    _depth_cdf_rows)

__all__ = ['plot_track_groups', 'group_colors', 'GROUP_LABELS', 'GROUP_SHADES', 'DEFAULT_COLORMAPS', 'LOGX_SPAN',
           'plot_flux_history', 'plot_exposure_timeline', 'interaction_profiles', 'plot_secondary_production']

# Labels, shades (position in the colour map, 1 = darkest end of a sequential map) and line styles of the track
# groups and totals. Other keys get their own name as label, evenly spaced shades and cycling line styles.
GROUP_LABELS = {
    'heavy': 'heavy',
    'medium_heavy': 'medium heavy',
    'medium_light': 'medium light',
    'light': 'light (H, He)',
    TOTAL_NO_LIGHT_IONS: 'total (no H, He)',
    TOTAL_ALL_FRAGMENTS: 'total',
}
GROUP_SHADES = {'heavy': 0.95, 'medium_heavy': 0.75, 'medium_light': 0.55, 'light': 0.35}
TOTAL_SHADE = 1.0
GROUP_LINESTYLES = {'heavy': '-', 'medium_heavy': '-', 'medium_light': '--', 'light': ':'}
OTHER_LINESTYLES = ('-', '--', ':', '-.')
# sequential matplotlib colour maps, one family per curve
DEFAULT_COLORMAPS = ('Blues', 'Oranges', 'Greens', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'PuRd')
# Default log x axis when the shown range spans more than this factor (from the first positive bin edge); bins of a
# ratio whose reference is below RATIO_MIN_FRACTION of the reference's maximum are not drawn (noise of empty bins).
LOGX_SPAN = 100.
RATIO_MIN_FRACTION = 1e-6
# Colours of the components of a single curve in the 'lines' layout (categorical), and of its total.
SINGLE_CURVE_COLORMAP = 'tab10'
SINGLE_CURVE_TOTAL_COLOR = 'black'
# default axes of each observable of the integrate_* functions
_AXES = {
    'tracks': {'xlabel': 'Track length [nm]', 'counts': 'Tracks', 'unit': 'nm', 'logx': False},
    'recoil_energy': {'xlabel': 'Recoil energy [MeV]', 'counts': 'Recoils', 'unit': 'MeV', 'logx': True},
}


def _shades(keys):
    """Colour-map position of each key: GROUP_SHADES for the track groups, else evenly spaced, darkest first."""
    if all(k in GROUP_SHADES for k in keys):
        return {k: GROUP_SHADES[k] for k in keys}
    return dict(zip(keys, np.linspace(0.95, 0.35, len(keys)) if len(keys) > 1 else [0.75]))


def _linestyle(key, keys):
    """Line style of a key in the 'lines' layout: GROUP_LINESTYLES for the track groups, else cycling."""
    return GROUP_LINESTYLES.get(key, OTHER_LINESTYLES[list(keys).index(key) % len(OTHER_LINESTYLES)])


def group_colors(cmap, groups=TRACK_GROUPS):
    """Shades of one colour family for the components (and the total) of a curve.

    Args:
        cmap (str or matplotlib colormap): A sequential colour map, e.g. 'Blues'.
        groups (tuple[str], optional): Components to colour (track groups or any keys). Defaults to TRACK_GROUPS.

    Returns:
        dict: component -> RGBA colour, plus 'total' for the total.
    """
    import matplotlib
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = {k: cm(shade) for k, shade in _shades(tuple(groups)).items()}
    colors['total'] = cm(TOTAL_SHADE)
    return colors


def _as_curves(curves):
    """{label: {key: counts}}: accepts that, a list of (label, dict) pairs, or one {key: counts} dict alone (whose
    values are arrays, not dicts)."""
    if isinstance(curves, dict) and curves and not isinstance(next(iter(curves.values())), dict):
        return {'': curves}
    return dict(curves)


def _plot_setup(curves, x_bins, style, keys, total, per_nm, logx, logy, colormaps, show_light, xlim, xlabel, ylabel,
                observable, ratio_to=None):
    """What every layout of plot_track_groups needs, worked out once: the curves, the keys drawn, the scaling, the
    colours, the axes ranges and labels (see plot_track_groups for the arguments).

    Returns:
        dict: 'curves', 'x_bins', 'style', 'keys' (components), 'total', 'drawn' (components and total), 'scale'
        (counts -> plotted values), 'ratio_to', 'shown' (labels of the curves drawn: all but the reference),
        'single' (one curve in 'lines': categorical colours), 'families' ({curve: colours}), 'labels' ({key: label}),
        'xlim', 'logx', 'logy', 'xlabel', 'ylabel'.

    Raises:
        ValueError: If style or observable is unknown, a curve lacks a key, or ratio_to is not a curve (or is used
            with 'stack').
    """
    curves = _as_curves(curves)
    x_bins = np.asarray(x_bins, dtype=float)
    keys = tuple(k for k in keys if show_light or k != 'light')
    drawn = (*keys, *([total] if total else []))
    for label, c in curves.items():
        missing = [k for k in drawn if k not in c]
        if missing:
            raise ValueError(f"curve {label!r} has no {missing} (track groups: compute with nucleus='groups'; "
                             f"nuclides: nucleus='all').")
    if style == 'auto':
        style = 'stack' if len(curves) == 1 else 'lines' if len(curves) <= 3 else 'panels'
    if style not in _LAYOUTS:
        raise ValueError(f"style must be 'auto' or one of {tuple(_LAYOUTS)}, got {style!r}.")
    if ratio_to is not None:
        if ratio_to not in curves:
            raise ValueError(f"ratio_to={ratio_to!r} is not one of the curves {list(curves)}.")
        if style == 'stack':
            raise ValueError("ratio_to works with style 'lines' or 'panels', not 'stack'.")
    shown = [label for label in curves if label != ratio_to]
    single = style == 'lines' and len(shown) == 1
    if observable not in _AXES:
        raise ValueError(f"observable must be one of {tuple(_AXES)}, got {observable!r}.")
    axis = _AXES[observable]
    widths = np.diff(x_bins)
    scale = (lambda v: np.asarray(v, dtype=float) / widths) if per_nm else (lambda v: np.asarray(v, dtype=float))

    positive = x_bins[x_bins > 0][0]
    if isinstance(xlim, str) and xlim == 'auto':
        values = [scale(c[k]) for c in curves.values() for k in drawn]
        peak = max(float(np.max(v)) for v in values)
        last = max((int(np.nonzero(v > 1e-6 * peak)[0][-1]) for v in values if np.any(v > 1e-6 * peak)),
                   default=len(widths) - 1)
        x_end = x_bins[last + 1]
        logx = (axis['logx'] or x_end / positive > LOGX_SPAN) if logx is None else logx
        x_lo = positive if logx else x_bins[0]
        x_hi = min(x_bins[-1], x_end * 1.5 if logx else x_end + 0.1 * (x_end - x_lo))
    else:
        x_lo, x_hi = (x_bins[0], x_bins[-1]) if xlim is None else xlim
        logx = (axis['logx'] or x_hi / max(x_lo, positive) > LOGX_SPAN) if logx is None else logx
        x_lo = max(x_lo, positive) if logx else x_lo

    if single:
        import matplotlib
        cmap = matplotlib.colormaps[SINGLE_CURVE_COLORMAP]
        families = {shown[0]: {**{k: cmap(i % cmap.N) for i, k in enumerate(keys)},
                               'total': SINGLE_CURVE_TOTAL_COLOR}}
    else:
        families = {label: group_colors(colormaps[i % len(colormaps)], keys) for i, label in enumerate(shown)}
    if ylabel is None:
        ylabel = (f"Ratio to {ratio_to}" if ratio_to is not None else
                  f"{axis['counts']} / mg / {axis['unit'] if per_nm else 'bin'}")
    return {
        'curves': curves, 'x_bins': x_bins, 'style': style, 'keys': keys, 'total': total, 'drawn': drawn,
        'scale': scale, 'ratio_to': ratio_to, 'shown': shown, 'single': single, 'families': families,
        'labels': {k: GROUP_LABELS.get(k, k) for k in drawn},
        'xlim': (x_lo, x_hi), 'logx': logx,
        'logy': (style != 'stack' and ratio_to is None) if logy is None else logy,
        'xlabel': axis['xlabel'] if xlabel is None else xlabel, 'ylabel': ylabel,
    }


def _values(p, label, key):
    """What is drawn for one component of one curve: its counts (scaled, see per_nm), or with ratio_to its ratio to
    the reference curve (NaN where the reference is below RATIO_MIN_FRACTION of its maximum). Called by the layouts."""
    values = p['scale'](p['curves'][label][key])
    if p['ratio_to'] is None:
        return values
    ref = p['scale'](p['curves'][p['ratio_to']][key])
    keep = ref > RATIO_MIN_FRACTION * np.max(ref) if np.max(ref) > 0 else np.zeros(ref.shape, dtype=bool)
    return np.where(keep, values / np.where(keep, ref, 1.0), np.nan)


def _baseline(p):
    """Baseline of the step lines: 0 (vertical edges down at the ends of a spectrum) for counts, none for ratios."""
    return None if p['ratio_to'] is not None else 0


def _finish_axes(a, p, top, y_decades):
    """Scales and limits of one axes; with a log y axis and y_decades, y from top / 10**y_decades to 2 * top, top
    being the largest value drawn on it. With ratio_to, a dashed line at 1 for the reference."""
    a.set_xscale('log' if p['logx'] else 'linear')
    a.set_yscale('log' if p['logy'] else 'linear')
    a.set_xlim(*p['xlim'])
    if p['ratio_to'] is not None:
        a.axhline(1.0, color='grey', linestyle='--', linewidth=1.0, label=p['ratio_to'])
    if p['logy'] and y_decades and top > 0:
        a.set_ylim(top * 10 ** (-y_decades), top * 2)


def _plot_stack(p, ax, y_decades, legend):
    """'stack' layout: the components of each curve stacked as filled areas (first at the bottom), the total as an
    outline."""
    import matplotlib.pyplot as plt
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    x, top = p['x_bins'], 0.0
    for label in p['shown']:
        c = p['curves'][label]
        colors = p['families'][label]
        bottom = np.zeros(len(x) - 1)
        for k in p['keys']:
            upper = bottom + p['scale'](c[k])
            ax.stairs(upper, x, baseline=bottom, fill=True, color=colors[k], linewidth=0,
                      label=f"{label} {p['labels'][k]}".strip())
            bottom = upper
        top = max(top, float(np.max(bottom)))
        if p['total']:
            total = p['scale'](c[p['total']])
            ax.stairs(total, x, color=colors['total'], linewidth=1.2,
                      label=f"{label} {p['labels'][p['total']]}".strip())
            top = max(top, float(np.max(total)))
    _finish_axes(ax, p, top, y_decades)
    ax.set_xlabel(p['xlabel'])
    ax.set_ylabel(p['ylabel'])
    if legend:
        ax.legend(frameon=False, fontsize=8)
    return fig, ax


def _plot_lines(p, ax, y_decades, legend):
    """'lines' layout: every component of every curve as a line, the total as a thick line. Several curves: shade
    and line style = component, colour family = curve, two legends (curves, components). One curve: a colour per
    component and one legend."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4.5))
    x, keys, top = p['x_bins'], p['keys'], 0.0
    for label in p['shown']:
        colors = p['families'][label]
        for k in keys:
            values = _values(p, label, k)
            ax.stairs(values, x, color=colors[k], linestyle='-' if p['single'] else _linestyle(k, keys),
                      linewidth=1.2, label=p['labels'][k] if p['single'] else None, baseline=_baseline(p))
            top = max(top, float(np.nanmax(values, initial=0.0)))
        if p['total']:
            values = _values(p, label, p['total'])
            ax.stairs(values, x, color=colors['total'], linewidth=2.4,
                      label=p['labels'][p['total']] if p['single'] else None, baseline=_baseline(p))
            top = max(top, float(np.nanmax(values, initial=0.0)))
    _finish_axes(ax, p, top, y_decades)
    ax.set_xlabel(p['xlabel'])
    ax.set_ylabel(p['ylabel'])
    if p['single'] and p['shown'][0]:
        ax.set_title(p['shown'][0])
    if not legend:
        return fig, ax
    if p['single']:
        handles, labels = ax.get_legend_handles_labels()
        order = sorted(range(len(labels)), key=lambda i: labels[i] != p['labels'].get(p['total']))
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=8,
                  loc='upper left', bbox_to_anchor=(1.01, 1.0))
        return fig, ax
    curve_handles = [Patch(color=p['families'][label][keys[0]], label=label or 'spectrum') for label in p['shown']]
    if p['ratio_to'] is not None:
        curve_handles.append(Line2D([], [], color='grey', linestyle='--', label=p['ratio_to']))
    grey = group_colors('Greys', keys)
    key_handles = [Line2D([], [], color=grey[k], linestyle=_linestyle(k, keys), linewidth=1.2,
                          label=p['labels'][k]) for k in keys]
    if p['total']:
        key_handles.insert(0, Line2D([], [], color=grey['total'], linewidth=2.4, label=p['labels'][p['total']]))
    first = ax.legend(handles=curve_handles, frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=key_handles, frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(1.01, 0.0))
    return fig, ax


def _plot_panels(p, ax, y_decades, legend):
    """'panels' layout: one panel per component (the total first), every curve in its own colour family in each;
    x labels on the lowest visible panel of each column. ax is ignored (a new figure)."""
    import matplotlib.pyplot as plt
    shown = ([p['total']] if p['total'] else []) + list(p['keys'])
    n = len(shown)
    ncols = 2 if n > 2 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows), sharex=True, squeeze=False)
    for i, (a, k) in enumerate(zip(axes.flat, shown)):
        top = 0.0
        for label in p['shown']:
            family = p['families'][label]
            values = _values(p, label, k)
            a.stairs(values, p['x_bins'], color=family['total'] if k == p['total'] else family[p['keys'][0]],
                     linewidth=1.4, label=label or None, baseline=_baseline(p))
            top = max(top, float(np.nanmax(values, initial=0.0)))
        a.set_title(p['labels'][k], fontsize=10)
        _finish_axes(a, p, top, y_decades)
        if i % ncols == 0:
            a.set_ylabel(p['ylabel'])
    for a in axes.flat[n:]:
        a.set_visible(False)
    for col in range(ncols):
        visible = [axes[row, col] for row in range(nrows) if row * ncols + col < n]
        if visible:
            visible[-1].set_xlabel(p['xlabel'])
            visible[-1].xaxis.set_tick_params(labelbottom=True)
    if legend and (any(p['shown']) or p['ratio_to'] is not None):
        axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes


_LAYOUTS = {'stack': _plot_stack, 'lines': _plot_lines, 'panels': _plot_panels}


def plot_track_groups(curves, x_bins, style='auto', keys=TRACK_GROUPS, total=TOTAL_NO_LIGHT_IONS, ax=None,
                      per_nm=False, logx=None, logy=None, colormaps=DEFAULT_COLORMAPS, show_light=True,
                      xlim='auto', y_decades=None, xlabel=None, ylabel=None, legend=True, observable='tracks',
                      ratio_to=None):
    """Plots track-length (or recoil-energy) spectra split into components: the track groups, or any keys.

    Works out the common settings once (_plot_setup), then draws with the layout of `style` (_plot_stack,
    _plot_lines, _plot_panels; see the module docstring).

    Args:
        curves (dict): {label: dict of counts per key} (e.g. a nucleus='groups' output), or one such dict alone.
        x_bins (np.ndarray): Bin edges of the spectra [nm, or MeV for recoil energies].
        style (str, optional): 'auto', 'stack', 'lines' or 'panels' (see the module docstring). Defaults to 'auto'.
        keys (tuple[str], optional): Components to show, in order (the first gets the darkest shade). Defaults to
            TRACK_GROUPS (heaviest first); any keys of the curves work, e.g. nuclides or species.
        total (str or None, optional): Key of the total drawn with the components, or None. Defaults to
            'total_no_light_ions'.
        ax (matplotlib Axes, optional): Axes for 'stack' and 'lines' (a new figure if None). Ignored by 'panels'.
        per_nm (bool, optional): Plot counts per unit of the x axis (counts / bin width: per nm, or per MeV for
            recoil energies) instead of counts per bin. Defaults to False.
        logx (bool, optional): Logarithmic x axis. Defaults to None: True for recoil energies, and for tracks when
            the range shown spans more than LOGX_SPAN (from the first positive bin edge).
        logy (bool, optional): Logarithmic y axis. Defaults to None: False for 'stack' and for ratios, True
            otherwise.
        colormaps (tuple, optional): Colour families of the curves, in order. Defaults to DEFAULT_COLORMAPS.
        show_light (bool, optional): Include the 'light' key if present in keys. Defaults to True.
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
        ratio_to (str, optional): Label of one of the curves: the others are drawn divided by it, component by
            component (a dashed line at 1 stands for it); 'lines' or 'panels' only. Defaults to None (counts).

    Returns:
        tuple: (fig, axes), axes being one Axes or, for 'panels', an array of them.

    Raises:
        ValueError: If style or observable is unknown, a curve lacks a key, or ratio_to is not a curve (or is used
            with 'stack').
    """
    p = _plot_setup(curves, x_bins, style, keys, total, per_nm, logx, logy, colormaps, show_light, xlim, xlabel,
                    ylabel, observable, ratio_to)
    return _LAYOUTS[p['style']](p, ax, y_decades, legend)


# --- History of a sample: flux and overburden against time ------------------------------------------------------------

def _flux_times(flux_history, t_lo, t_hi, n_times):
    """Times [kyr, flux timeline] to draw a flux history on [t_lo, t_hi]: uniform, log-spaced from t_lo (so that a
    log time axis from the start is drawn right too), and log-spaced after each event start (transients rise and
    decay much faster than the range). Called by _draw_flux."""
    span = t_hi - t_lo
    t = [np.linspace(t_lo, t_hi, n_times), t_lo + np.geomspace(span * 1e-6, span, n_times)]
    for event in flux_history.events:
        start = event['start_time_kyr']
        if start < t_hi:
            span = t_hi - max(start, t_lo)
            t.append(np.clip(start + np.geomspace(1e-3, max(span, 2e-3), 200), t_lo, t_hi))
    return np.unique(np.concatenate(t))


def _draw_flux(ax, flux_history, species, energies_gev, t_lo, t_hi, offset, relative, n_times):
    """Flux of one species at a few energies against time on ax, x = flux time + offset (0 for the flux timeline,
    total_age_kyr for a detector's local time); excursions shaded, event starts dotted. Called by plot_flux_history
    and plot_exposure_timeline. Linear y for the flux / baseline, log for the flux."""
    t = _flux_times(flux_history, t_lo, t_hi, n_times)
    energies = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    flux = flux_history.get_map(species, t, energies)[2]
    if relative:
        flux = flux / flux_history.baseline.flux(species, energies)[None, :]
    for i, e in enumerate(energies):
        ax.plot(t + offset, flux[:, i], color=f'C{i}', linewidth=1.4, label=f'{species}, {e:g} GeV')
    for k, exc in enumerate(flux_history.excursions):
        ax.axvspan(exc['start_time_kyr'] + offset, exc['end_time_kyr'] + offset, color='grey', alpha=0.25,
                   linewidth=0, label='excursion' if k == 0 else None)
    for event in flux_history.events:
        if t_lo <= event['start_time_kyr'] <= t_hi:
            ax.axvline(event['start_time_kyr'] + offset, color='k', linestyle=':', linewidth=1)
            ax.annotate(event.get('label') or event['template'].name, (event['start_time_kyr'] + offset, 1),
                        xycoords=('data', 'axes fraction'), xytext=(3, -12), textcoords='offset points',
                        fontsize='x-small')
    ax.set_yscale('linear' if relative else 'log')
    ax.set_ylabel(f'{species} flux / baseline' if relative else f'{species} flux [cm$^{{-2}}$ s$^{{-1}}$ GeV$^{{-1}}$]')


def plot_flux_history(flux_history, species='mu-', energies_gev=(1., 10., 100.), t_range_kyr=None, relative=True,
                      ax=None, n_times=2000):
    """A FluxHistory against time: the flux of one species at a few energies, its events and excursions.

    A check of a scenario before integrating: the amplitude, start and decay of each event, the excursion windows.

    Args:
        flux_history (FluxHistory): The history.
        species (str, optional): Template species. Defaults to 'mu-'.
        energies_gev (tuple, optional): Energies drawn [GeV]. Defaults to (1, 10, 100).
        t_range_kyr (tuple, optional): (start, end) on the flux timeline [kyr, 0 = present]. Defaults to None: from
            a little before the earliest event or excursion (or -100 kyr) to the present.
        relative (bool, optional): Divide by the baseline flux. Defaults to True.
        ax (matplotlib Axes, optional): Axes to draw on. Defaults to None (a new figure).
        n_times (int, optional): Uniform time points (more are added after each event start). Defaults to 2000.

    Returns:
        tuple: (fig, ax).
    """
    import matplotlib.pyplot as plt
    if t_range_kyr is None:
        starts = [e['start_time_kyr'] for e in flux_history.events] + \
                 [e['start_time_kyr'] for e in flux_history.excursions]
        t_lo = min(starts + [-100.]) * 1.1
        t_range_kyr = (t_lo, 0.)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 4))
    _draw_flux(ax, flux_history, species, energies_gev, *t_range_kyr, 0., relative, n_times)
    ax.set_xlabel('Time [kyr, 0 = present]')
    ax.set_xlim(*t_range_kyr)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_exposure_timeline(detector, flux_history=None, windows=None, species='mu-', energies_gev=(10.,),
                           relative=True, n_times=2000):
    """A sample over its exposure: the overburden (top) and the flux at the site (bottom) against the detector's
    local time (0 = start of exposure, total_age_kyr = present), with the exposure windows of the integrals.

    Shows on one axis what the integrals combine: when the sample is shielded, when events and excursions act on
    it, and which part of the history a window (exposure_window_kyr) covers. Events that start before the exposure
    are listed in the title (only their tail reaches the sample).

    Args:
        detector (Paleodetector): The sample.
        flux_history (FluxHistory, optional): Defaults to None (the detector's).
        windows (dict or list, optional): Exposure windows [kyr, local time]: {label: (start, end)} or a list of
            (start, end). Defaults to None.
        species (str, optional): Template species of the flux panel. Defaults to 'mu-'.
        energies_gev (tuple, optional): Energies of the flux panel [GeV]. Defaults to (10,).
        relative (bool, optional): Flux divided by the baseline. Defaults to True.
        n_times (int, optional): Uniform time points. Defaults to 2000.

    Returns:
        tuple: (fig, (ax_overburden, ax_flux)).
    """
    import matplotlib.pyplot as plt
    flux_history = detector.flux_history if flux_history is None else flux_history
    age = detector.total_age_kyr
    fig, (ax_ob, ax_fl) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    t = np.unique(np.concatenate([np.linspace(0., age, n_times), np.geomspace(age * 1e-6, age, n_times)]))
    overburden = np.asarray(detector.overburden_mwe(t), dtype=float)
    ax_ob.plot(t, overburden, color='k', linewidth=1.4)
    if np.all(overburden > 0):
        ax_ob.set_yscale('log')
    ax_ob.set_ylabel('Overburden [m.w.e.]')

    _draw_flux(ax_fl, flux_history, species, energies_gev, -age, 0., age, relative, n_times)
    ax_fl.set_xlabel(f'Local time [kyr] (0 = start of exposure, {age:g} = present)')
    ax_fl.set_xlim(0., age)

    if windows is not None:
        items = windows.items() if isinstance(windows, dict) else ((None, w) for w in windows)
        for k, (label, (lo, hi)) in enumerate(items):
            for a in (ax_ob, ax_fl):
                a.axvspan(lo, hi, color=f'C{k + 3}', alpha=0.12, linewidth=0,
                          label=(label or f'window {lo:g}-{hi:g} kyr') if a is ax_ob else None)
        ax_ob.legend(frameon=False, fontsize=8)
    ax_fl.legend(frameon=False, fontsize=8)
    early = [e for e in flux_history.events if e['start_time_kyr'] < -age]
    if early:
        ax_ob.set_title('; '.join(f"{e.get('label') or e['template'].name}: starts "
                                  f"{-age - e['start_time_kyr']:g} kyr before the exposure" for e in early),
                        fontsize=9)
    fig.tight_layout()
    return fig, (ax_ob, ax_fl)


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
            taken as vertical, so the panel shows shapes and crossover depths; the integrations spread the
            primaries over zenith angle with the zenith distribution of the flux templates.
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
