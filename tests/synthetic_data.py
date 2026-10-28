"""Small synthetic data root for the tests: a toy "Halite" with mineral and StdRock runs, range tables and neutrino
runs, written in the layout primus_paleo reads (see primus_paleo.config). Numbers are arbitrary, not physics."""
import json
import os

import numpy as np

ENERGIES_GEV = np.logspace(-3, 1, 20)          # hadron and muon runs
NU_ENERGIES_GEV = np.array([1.0, 2.0, 4.0, 8.0])
RANGE_K_MEV_PER_UM = 1.1                       # toy range tables: L = E / k, so dE/dx = k exactly


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write("%s\t%g\t%g\t%g\t%g\t%g\n" % r)


def make_data_root(root, seed=1):
    """Writes the synthetic data root in `root` and returns it."""
    rng = np.random.default_rng(seed)
    g4 = os.path.join(root, "Geant4_data")
    with open(os.path.join(root, "mineral_library.yaml"), "w") as f:
        f.write("Halite:\n  name: Halite\n  density_g_cm3: 2.16\n  nuclei: [Na, Cl]\n  stoich: [1, 1]\n"
                "  uranium_concentration_g_g: 1.0e-8\n  an_coeff_n_nalpha: 1\n")
    for sp in ("mu-", "neutron", "proton"):                    # StdRock: depth tables and secondary production
        for e in ENERGIES_GEV:
            rows = []
            for k in range(120):
                if sp == "mu-":
                    d = min(rng.exponential(e * 300 + 1), 500)
                    rows.append(("Na23", 0, 0.1, (100 - d / 2.65) * 1e3, 0, 0.0 if k % 3 == 0 else e * 1e3))
                else:
                    d = rng.exponential(1.0 if sp == "neutron" else 0.3)
                    rows.append(("Na23", 0, 0.2, (100 - d / 2.65) * 1e3, 0, 0))
                if k % 2 == 0:
                    rows.append(("neutron" if k % 4 == 0 else "proton", 0, rng.uniform(1, max(2, e * 1e3)),
                                 (100 - d / 2.65) * 1e3, 0, 0))
            _write(os.path.join(g4, "StdRock", sp, f"outNuclei_{e:.6f}.txt"), rows)
    for sp in ("mu-", "mu+", "neutron", "proton"):             # mineral runs: recoils
        for e in ENERGIES_GEV:
            rows = [(n, 0, rng.uniform(0.05, 3), 0, 0, 0.0 if (sp == "mu-" and k % 5 == 0) else 1.0)
                    for k, n in enumerate(["Na23", "proton", "alpha", "Na23"] * 15)]
            _write(os.path.join(g4, "Halite", sp, f"outNuclei_{e:.6f}.txt"), rows)
    for e in NU_ENERGIES_GEV:                                  # neutrino runs: 10 Na23 recoils + products
        rows = [("Na23", 0, float(r), 10.0 * k, e * 1e3, 0) for k, r in enumerate(np.logspace(-1, 0.7, 10))]
        rows += [("neutron", 0, 15.0, 0.0, e * 1e3, 0)] * 2 + [("mu-", 0, 300.0, 0.0, e * 1e3, 0)]
        _write(os.path.join(g4, "Halite", "nu_mu", f"outNuclei_{e:.6f}.txt"), rows)
    with open(os.path.join(g4, "Halite", "nu_mu", "geometry.json"), "w") as f:
        json.dump({"depth_reference_z_m": 100., "target_length_m": 200.,
                   "neutrino_bias": {f"{e:.6f}": e * 1e8 for e in NU_ENERGIES_GEV}}, f)
    rng_dir = os.path.join(g4, "Halite", "Range")
    os.makedirs(rng_dir, exist_ok=True)
    E = np.logspace(-3, 2, 400)
    for z, a in ((1, 1), (2, 4), (11, 23)):
        np.savetxt(os.path.join(rng_dir, f"DEDX_Z{z}_A{a}.txt"),
                   np.c_[E, np.full_like(E, RANGE_K_MEV_PER_UM), np.zeros_like(E), E / RANGE_K_MEV_PER_UM],
                   header="E dEe/dx dEn/dx L")
    return root


class ToyFlux:
    """Stand-in for FluxHistory: constant power-law fluxes, no events (keeps the tests independent of MCEq)."""
    events, excursions = [], []
    NORM = {"mu-": 1e-3, "mu+": 1.2e-3, "neutron": 5e-3, "proton": 2e-3, "nu_mu": 1e-2}

    def get_map(self, species, t, E):
        t, E = np.atleast_1d(t), np.asarray(E)
        return t, E, np.ones((t.size,) + (1,) * E.ndim) * self.NORM.get(species, 0.0) * E ** -2.0

    def flux(self, species, E, t):
        return self.NORM.get(species, 0.0) * np.asarray(E) ** -2.0 * np.ones(np.broadcast_shapes(np.shape(E), np.shape(t)))
