import numpy as np
import pytest

import primus_paleo as pp
from synthetic_data import NU_ENERGIES_GEV, ToyFlux

HISTORY = {"initial_depth": 0.5, "end_time_continuous_kyr": 40., "rate_continuous": 2.5,
           "density_continuous_g_cm3": 2.16, "overburden_discrete": 1500., "time_discrete_kyr": 40.,
           "density_discrete_g_cm3": 2.16}


@pytest.fixture
def det(use_data, no_backgrounds):
    d = pp.Paleodetector("Halite", total_age_kyr=100., overburden_history=HISTORY, flux_history=ToyFlux())
    d.verbose = 0
    return d


def test_species_found(det):
    assert set(det.available_species()) >= {"mu-", "mu+", "neutron", "proton", "secondary_neutron"}
    assert det.available_neutrino_species() == ["nu_mu"]


@pytest.mark.parametrize("species", ["mu-", "neutron", "proton", "secondary_neutron", "secondary_proton"])
def test_spectra_finite_and_positive(det, species):
    tracks = det.integrate_particle_signal_spectrum(species=species, exposure_window_kyr=[0, 40], nucleus="total")
    assert np.all(np.isfinite(tracks)) and np.all(tracks >= 0) and tracks.sum() > 0


def test_annealing_shortens(det):
    x = det.integrate_particle_signal_spectrum(species="mu-", exposure_window_kyr=[0, 40])
    xa = det.integrate_particle_signal_spectrum(species="mu-", exposure_window_kyr=[0, 40], annealing=0.5)
    edges = pp.DEFAULT_INTEGRATION_X_BINS_NM
    mid = 0.5 * (edges[1:] + edges[:-1])
    assert np.isclose(xa.sum(), x.sum(), rtol=0.02)              # same tracks...
    assert (xa * mid).sum() / xa.sum() < (x * mid).sum() / x.sum()   # ...shorter


def test_pool_equals_single_species(det):
    kw = dict(exposure_window_kyr=[0, 40])
    _, res = det.integrate_all_particles(species_list=["mu-", "secondary_neutron"], flux_history=ToyFlux(), **kw)
    for s in ("mu-", "secondary_neutron"):
        assert np.array_equal(res[s], det.integrate_particle_signal_spectrum(species=s, **kw))


def test_neutrino_normalisation(det):
    """Direct nu_mu tracks against the hand calculation: rate per g = 4 Phi_cell counts / (N bias L rho)."""
    edges = np.linspace(0, 1e5, 2001)
    _, res = det.integrate_atmospheric_neutrinos(x_bins=edges, exposure_window_kyr=[0, 40.],
                                                 total_simulated_particles=1000., secondaries=False, nucleus="Na23")
    E = NU_ENERGIES_GEV
    w = 4 * (ToyFlux.NORM["nu_mu"] / (2 * E)) / (1000. * E * 1e8 * 2e4 * 2.16)
    expected = (10 * w).sum() * 40. / pp.KYR_PER_SECOND * 1e-3
    assert np.isclose(res["nu_mu"].sum(), expected, rtol=0.03)
