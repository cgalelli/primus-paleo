"""primus_paleo: simulation of paleo-detector track spectra (PRImuS).

    import primus_paleo as pp
    pp.describe()                                      # where data and the Geant4 application are taken from
    det = pp.Paleodetector("Halite", total_age_kyr=..., overburden_history=..., flux_history=pp.FluxHistory())
    x_bins, tracks = det.integrate_all_particles()

Modules:
    config          where the data root and the Geant4 application are (set_data_path, PRIMUS_DATA, ...)
    detector        the Paleodetector class
    flux            cosmic-ray flux templates and histories (FluxHistory)
    geant4_runs     running the Geant4 application (materials, runs, stopping-power tables)
    observables     etch pits, sliced/etched spectra, diagnostic profiles and plots
    _core           constants, binnings, species tables, depth-table maths, annealing laws (internal)

Every public name of these modules is available here. Other names (private helpers, the live value of
DEFAULT_INTEGRATION_X_BINS_NM changed by set_integration_x_bins_nm) are looked up in the modules at access time.
Assigning a name here (pp.X = ...) does not change the modules: edit _core.py for constants.
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("primus-paleo")
except PackageNotFoundError:                                  # running from the sources without installing
    __version__ = "0+unknown"

from . import config, _core, flux, detector, geant4_runs, observables
from .config import (set_data_path, data_path, set_geant4_path, geant4_app_path, flux_template_dir,
                     mineral_library_path, load_mineral_library, mineral_config, describe)
from .flux import FluxHistory, FluxTemplate, MCEQ_SPECIES
from ._core import *
from .detector import *
from .geant4_runs import *
from .observables import *

__all__ = (['config', 'set_data_path', 'data_path', 'set_geant4_path', 'geant4_app_path', 'flux_template_dir',
            'mineral_library_path', 'load_mineral_library', 'mineral_config', 'describe',
            'FluxHistory', 'FluxTemplate', 'MCEQ_SPECIES']
           + _core.__all__ + detector.__all__ + geant4_runs.__all__ + observables.__all__)

_MODULES = (_core, detector, flux, geant4_runs, observables, config)


def __getattr__(name):
    """Names not re-exported above (private helpers, DEFAULT_INTEGRATION_X_BINS_NM), read live from the modules."""
    for module in _MODULES:
        if name in vars(module):
            return vars(module)[name]
    raise AttributeError(f"module 'primus_paleo' has no attribute '{name}'")
