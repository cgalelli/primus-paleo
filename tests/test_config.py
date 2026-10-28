import os

import primus_paleo as pp
from primus_paleo import config


def test_data_root_priority(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIMUS_DATA", raising=False)
    pp.set_data_path(None)
    repo = config.repository_root()
    assert pp.data_path() == (os.path.join(repo, "Data") if repo and os.path.isdir(os.path.join(repo, "Data"))
                              else os.path.abspath("Data"))
    monkeypatch.setenv("PRIMUS_DATA", str(tmp_path / "env"))
    assert pp.data_path() == str(tmp_path / "env")
    pp.set_data_path(tmp_path / "session")
    assert pp.data_path() == str(tmp_path / "session")
    assert pp.data_path(tmp_path / "explicit") == str(tmp_path / "explicit")
    pp.set_data_path(None)
    assert pp.data_path() == str(tmp_path / "env")


def test_paths_do_not_depend_on_working_directory(use_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert pp.flux_template_dir() == os.path.join(use_data, "flux_data")
    assert pp.geant4_paths()['geant4'] == os.path.join(use_data, "Geant4_data")
    assert pp.mineral_config("Halite")["density_g_cm3"] == 2.16


def test_geant4_app_legacy_location(tmp_path, monkeypatch):
    monkeypatch.delenv("PRIMUS_GEANT4", raising=False)
    pp.set_geant4_path(None)
    legacy = tmp_path / "Data" / "Geant4_data"
    (legacy / "build").mkdir(parents=True)
    (legacy / "build" / "Run").write_text("")
    repo = config.repository_root()
    if not (repo and os.path.exists(os.path.join(repo, "geant4", "build", "Run"))):
        assert pp.geant4_app_path(data=tmp_path / "Data") == str(legacy)
    pp.set_geant4_path(tmp_path / "app")
    assert pp.geant4_app_path() == str(tmp_path / "app")
    pp.set_geant4_path(None)
