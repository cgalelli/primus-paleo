"""primus_paleo: simulation of paleo-detector track-length and recoil-energy spectra (PRImuS).

    import primus_paleo as pp
    pp.describe()                                      # where data and the Geant4 application are taken from
    det = pp.Paleodetector("Halite", total_age_kyr=..., overburden_history=..., flux_history=pp.FluxHistory())
    x_bins, cosmic = det.integrate_all_particles(nucleus='groups')       # {species: {group: counts}, 'total': ...}
    _, neutrinos = det.integrate_atmospheric_neutrinos(x_bins=x_bins, nucleus='groups')
    _, radiogenic = det.integrate_radiogenic(x_bins=x_bins, nucleus='groups')
    pp.plot_track_groups({'cosmic rays': cosmic['total'], 'neutrinos': neutrinos['total'],
                          'radiogenic': radiogenic['total']}, x_bins)
    e_bins, energies = det.integrate_all_particles(nucleus='all', observable='recoil_energy')   # per nuclide

Modules:
    config          where the data root and the Geant4 application are (set_data_path, PRIMUS_DATA, ...)
    detector        the Paleodetector class
    flux            cosmic-ray flux templates and histories (FluxHistory)
    geant4_runs     running the Geant4 application (materials, runs, stopping-power tables)
    observables     what is measured: etch pits, etched spectra, microscope resolution, rebinning
    plotting        spectra split by track group, diagnostics of a detector (depth profiles, secondary production)
    _core           constants, binnings, species tables, track groups, depth-table maths (internal)

Every name in the __all__ of these modules is available here (each module lists its own public names). Other
names (private helpers, the live value of DEFAULT_INTEGRATION_X_BINS_NM changed by set_integration_x_bins_nm) are
looked up in the modules at access time.
Assigning a name here (pp.X = ...) does not change the modules: edit _core.py for constants.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("primus-paleo")
except PackageNotFoundError:                                  # running from the sources without installing
    __version__ = "0+unknown"

from . import config, _core, flux, detector, geant4_runs, observables, plotting

# Every module lists its public names in __all__; those are re-exported here and make up the package's __all__.
# _MODULES is also where __getattr__ looks up the other names, in this order.
_MODULES = (_core, detector, flux, geant4_runs, observables, plotting, config)
for _module in _MODULES:
    if not hasattr(_module, '__all__'):
        raise ImportError(f"{_module.__name__} defines no __all__: list its public names there.")

from .config import *
from .flux import *
from ._core import *
from .detector import *
from .geant4_runs import *
from .observables import *
from .plotting import *

__all__ = ['config'] + [name for _module in _MODULES for name in _module.__all__]


def __getattr__(name):
    """Names not re-exported above (private helpers, DEFAULT_INTEGRATION_X_BINS_NM), read live from the modules."""
    for module in _MODULES:
        if name in vars(module):
            return vars(module)[name]
    raise AttributeError(f"module 'primus_paleo' has no attribute '{name}'")
