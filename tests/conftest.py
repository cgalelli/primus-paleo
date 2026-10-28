import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))

import primus_paleo as pp                                     # noqa: E402
from synthetic_data import make_data_root                     # noqa: E402


@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    """Synthetic data root, built once per test session."""
    return make_data_root(str(tmp_path_factory.mktemp("Data")))


@pytest.fixture
def use_data(data_root):
    """Points primus_paleo at the synthetic data root for one test, then restores the default search."""
    pp.set_data_path(data_root)
    yield data_root
    pp.set_data_path(None)


@pytest.fixture
def no_backgrounds(monkeypatch):
    """Skips the fission and radiogenic-neutron spectra (they need the real nuclear_data tables)."""
    monkeypatch.setattr(pp.Paleodetector, "_radiogenic_spectrum", lambda self: None)
    monkeypatch.setattr(pp.Paleodetector, "_alpha_n_spectrum", lambda self: None)
