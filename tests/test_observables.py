import numpy as np

import primus_paleo as pp


def test_rebin_conserves_counts():
    old, new = np.linspace(0, 10, 11), np.linspace(0, 10, 4)
    c = np.arange(10.)
    assert np.isclose(pp.rebin_counts(c, old, new).sum(), c.sum())


def test_etch_pits():
    L = np.array([1000., 5000.])
    major, minor = pp.etch_pit_axes(L, np.full(2, np.pi / 2), np.zeros(2), 4000., 2.5)
    assert np.all(np.isfinite(major)) and major[1] >= major[0]
    shallow, _ = pp.etch_pit_axes(L, np.full(2, 0.2), np.zeros(2), 4000., 2.5)   # sin(dip) < 1/V: no opening
    assert np.all(np.isnan(shallow))


def test_slice_zero_and_reproducible():
    edges = pp.DEFAULT_INTEGRATION_X_BINS_NM
    assert pp.slice_spectrum_etch(np.zeros(len(edges) - 1), 0.02, 2.16).sum() == 0
    c = np.random.default_rng(0).random(len(edges) - 1)
    a = pp.slice_spectrum_etch(c, 0.02, 2.16, rng=np.random.default_rng(3))
    b = pp.slice_spectrum_etch(c, 0.02, 2.16, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)
