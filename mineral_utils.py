"""Paleo-detector simulation tools: front door to the modules below, so that notebooks only need

    from mineral_utils import Paleodetector, slice_spectrum_etch, run_geant4_simulation, ...

Modules:
    paleo_core      constants, binnings, species tables, fragment naming, depth-table maths, annealing laws (internal)
    paleodetector   the Paleodetector class
    geant4_runs     running the Geant4 application (materials, runs, stopping-power tables)
    observables     etch pits, sliced/etched spectra, diagnostic profiles and plots

Every public name of the four modules is available here. Other names, including the private helpers and the live value
of DEFAULT_INTEGRATION_X_BINS_NM (changed by set_integration_x_bins_nm), are looked up in the modules at access time,
e.g. mineral_utils._annealing_law. Assigning a name here (mineral_utils.X = ...) does not change the modules: edit
paleo_core for constants.
"""
import paleo_core
import paleodetector
import geant4_runs
import observables

from paleo_core import *
from paleodetector import *
from geant4_runs import *
from observables import *

_MODULES = (paleo_core, paleodetector, geant4_runs, observables)


def __getattr__(name):
    """Names not re-exported above (private helpers, DEFAULT_INTEGRATION_X_BINS_NM), read live from the modules."""
    for module in _MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module 'mineral_utils' has no attribute '{name}'")