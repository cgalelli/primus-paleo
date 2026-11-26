"""Where primus_paleo finds its data and the Geant4 application.

Repository layout (editable install, `pip install -e .` from the repository root):

    primus-paleo/
        src/primus_paleo/      the package
        geant4/                Geant4 application sources; compiled into geant4/build/Run
        analysis/              notebooks
        Data/                  the data root:
            mineral_library.yaml
            flux_data/             flux templates (FluxHistory)
            nuclear_data/          fission and binding-energy tables
            Geant4_data/<material>/<particle>/   simulation outputs (+ geometry.json)
            Geant4_data/<material>/Range/        stopping-power tables

Data root, first match wins:
    1. an explicit data_dir= argument (Paleodetector and the functions here; FluxHistory takes the template folder
       itself, template_dir=);
    2. set_data_path(...) called in the session;
    3. the environment variable PRIMUS_DATA;
    4. <repository>/Data, when the package is installed from the repository (editable install);
    5. ./Data in the current working directory.

Geant4 application folder (the folder that contains build/Run), first match wins:
    1. an explicit argument (geant4_dir=);
    2. set_geant4_path(...);
    3. the environment variable PRIMUS_GEANT4;
    4. <repository>/geant4 (whether compiled yet or not); without a repository, ./geant4.
"""
import os

__all__ = ['repository_root', 'set_data_path', 'data_path', 'set_geant4_path', 'geant4_app_path',
           'flux_template_dir', 'mineral_library_path', 'load_mineral_library', 'mineral_config', 'describe']

_DATA_PATH = None
_GEANT4_PATH = None


def repository_root():
    """The repository folder (the one with pyproject.toml above src/primus_paleo), or None if not installed from it."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return root if os.path.exists(os.path.join(root, "pyproject.toml")) else None


def set_data_path(path):
    """Sets the data root for the session (None goes back to the default search). Returns the absolute path."""
    global _DATA_PATH
    _DATA_PATH = None if path is None else os.path.abspath(os.path.expanduser(path))
    return data_path()


def data_path(data_dir=None):
    """Absolute data root: data_dir if given, else set_data_path, PRIMUS_DATA, <repository>/Data, ./Data."""
    if data_dir is not None:
        return os.path.abspath(os.path.expanduser(data_dir))
    if _DATA_PATH is not None:
        return _DATA_PATH
    env = os.environ.get("PRIMUS_DATA")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    root = repository_root()
    if root is not None and os.path.isdir(os.path.join(root, "Data")):
        return os.path.join(root, "Data")
    return os.path.abspath("Data")


def set_geant4_path(path):
    """Sets the Geant4 application folder (containing build/Run) for the session; None resets. Returns it."""
    global _GEANT4_PATH
    _GEANT4_PATH = None if path is None else os.path.abspath(os.path.expanduser(path))
    return geant4_app_path()


def geant4_app_path(geant4_dir=None):
    """Absolute folder of the Geant4 application, the one that contains build/Run (see the module docstring)."""
    if geant4_dir is not None:
        return os.path.abspath(os.path.expanduser(geant4_dir))
    if _GEANT4_PATH is not None:
        return _GEANT4_PATH
    env = os.environ.get("PRIMUS_GEANT4")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    root = repository_root()
    return os.path.join(root, "geant4") if root else os.path.abspath("geant4")


def flux_template_dir(data_dir=None):
    """Folder of the flux templates: <data root>/flux_data."""
    return os.path.join(data_path(data_dir), "flux_data")


def mineral_library_path(data_dir=None):
    """The mineral library: <data root>/mineral_library.yaml."""
    return os.path.join(data_path(data_dir), "mineral_library.yaml")


def load_mineral_library(data_dir=None):
    """All materials of the mineral library, as a dict {name: properties}."""
    import yaml
    with open(mineral_library_path(data_dir)) as f:
        return yaml.safe_load(f) or {}


def mineral_config(name, data_dir=None):
    """Properties of one material of the mineral library (the mineral_config of Paleodetector).

    Raises:
        KeyError: If the material is not in the library.
    """
    library = load_mineral_library(data_dir)
    if name not in library:
        raise KeyError(f"{name} is not in {mineral_library_path(data_dir)} (available: {sorted(library)}).")
    return library[name]


def describe():
    """Prints where data and the Geant4 application are taken from, and whether they exist."""
    d, g = data_path(), geant4_app_path()
    rows = [("data root", d), ("mineral library", mineral_library_path()), ("flux templates", flux_template_dir()),
            ("Geant4 outputs", os.path.join(d, "Geant4_data")), ("Geant4 application", os.path.join(g, "build", "Run"))]
    for label, p in rows:
        print(f"{label:<19} {p}  {'(ok)' if os.path.exists(p) else '(missing)'}")
