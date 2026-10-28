import warnings

import primus_paleo as pp


def test_old_imports_still_work():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import flux_history
        import mineral_utils
    assert mineral_utils.Paleodetector is pp.Paleodetector
    assert mineral_utils._annealing_law is pp._core._annealing_law
    assert flux_history.FluxHistory is pp.FluxHistory
