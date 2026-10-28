import json
import os
import stat

import numpy as np
import pytest

import primus_paleo as pp

FAKE_RUN = """#!/bin/bash
# stand-in for the Geant4 application: writes build/output/outNuclei_<i>.txt relative to its working directory
if [ "$1" == "--export-dedx" ]; then echo "E dEe dEn L" > "$5/DEDX_Z$3_A$4.txt"; exit 0; fi
cp "$1" build/last_macro.in
i=0
while read -r a b; do
  if [ "$a" == "/run/beamOn" ]; then echo "Na23 0 1.0 0 0 1" > build/output/outNuclei_$i.txt; i=$((i+1)); fi
done < "$1"
"""


@pytest.fixture
def app(tmp_path, use_data, monkeypatch):
    app_dir = tmp_path / "geant4"
    (app_dir / "build").mkdir(parents=True)
    run = app_dir / "build" / "Run"
    run.write_text(FAKE_RUN)
    run.chmod(run.stat().st_mode | stat.S_IEXEC)
    pp.set_geant4_path(app_dir)
    monkeypatch.chdir(tmp_path)                       # unrelated working directory
    yield app_dir
    pp.set_geant4_path(None)


def test_runs_are_filed_and_merged(app, use_data):
    pp.run_geant4_simulation("Halite", "mu-", 100, [0.001, 0.002], verbose=False)
    out = pp.run_geant4_simulation("Halite", "mu-", 100, [0.004], verbose=False)
    assert out == os.path.join(use_data, "Geant4_data", "Halite", "mu-")
    assert {"outNuclei_0.001000.txt", "outNuclei_0.002000.txt", "outNuclei_0.004000.txt"} <= set(os.listdir(out))
    geo = json.load(open(os.path.join(out, "geometry.json")))
    assert len(geo["n_events"]) == 3 and "primus_paleo" in geo["code_version"]


def test_neutrino_bias_in_macro_and_geometry(app, use_data):
    out = pp.run_geant4_simulation("Halite", "nu_mu", 10, [0.3], verbose=False)
    macro = open(app / "build" / "last_macro.in").read()
    bias = json.load(open(os.path.join(out, "geometry.json")))["neutrino_bias"]["0.300000"]
    assert f"/physics_lists/nu/NuNucleusBias {bias:g}" in macro
    assert macro.index("NuNucleusBias") < macro.index("/run/initialize")


def test_stopping_power_export(app, use_data):
    assert pp.export_all_stopping_powers("Halite", max_z=2, verbose=False) == []
    assert os.path.exists(os.path.join(use_data, "Geant4_data", "Halite", "Range", "DEDX_Z1_A1.txt"))


def test_suggested_bias_order_of_magnitude():
    b = pp.suggested_neutrino_bias(np.array([1., 100.]), 5000., 2.65)
    assert 1e6 < b[0] < 1e8 and b[1] < b[0]
