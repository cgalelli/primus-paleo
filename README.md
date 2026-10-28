# primus-paleo

**Simulation and analysis tools for cosmic-ray paleo-detectors**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23320945.svg)](https://doi.org/10.5281/zenodo.23320945)

This repository contains the code behind **PRImuS** (*Paleo-astroparticles Reconstructed with the Interactions of MUons in Stone*), an INFN experiment that aims to reconstruct past cosmic-ray fluxes from the nuclear-recoil tracks induced by cosmic rays in natural minerals.

It collects, in one place:

- **`primus_paleo`**, the Python package that predicts the expected track rate and track-length distribution in a mineral, given its geological history and an assumed cosmic-ray flux scenario;
- the **Geant4 application** that produces the particle-interaction simulations the package reads;
- the **analysis notebooks** behind our published phenomenological studies (Messinian halite and Chaîne des Puys olivine xenoliths);
- **OptimusPrimus**, the machine-learning pipeline for detecting tracks in microscope images, used for the experimental side of PRImuS.

## Repository contents

```
primus-paleo/
├── src/primus_paleo/   the Python package (simulation framework)
├── geant4/             Geant4 application (C++), built in geant4/build
├── analysis/           notebooks and OptimusPrimus
├── Data/               data root: mineral library, flux templates, nuclear data, Geant4 outputs
└── tests/              test suite on a small synthetic dataset
```

| Module | Description |
|---|---|
| `primus_paleo.detector` | `Paleodetector`: track-length spectra of a mineral from cosmic-ray muons, neutrons and protons (primaries and their secondaries, all generations), atmospheric neutrinos, spontaneous fission and radiogenic neutrons, through the sample's overburden and flux histories, with optional track annealing. |
| `primus_paleo.flux` | Cosmic-ray flux templates (computed with MCEq) and time-dependent flux scenarios (`FluxHistory`): baseline, supernovae, geomagnetic excursions. |
| `primus_paleo.geant4_runs` | Running the Geant4 application: materials, simulations per mineral and particle (with neutrino cross-section biasing), stopping-power tables. |
| `primus_paleo.observables` | What is measured: etch-pit openings and sliced/etched track spectra; diagnostic plots of depth profiles and secondary production. |
| `primus_paleo.config` | Where the data root and the Geant4 application are found. |
| `analysis/` | `MSC_analysis_new.ipynb` (Messinian halite), `CDP_analysis_new.ipynb` (Chaîne des Puys olivines), `OptimusPrimus.py` and `Test_optimusprimus.ipynb`, the Geant4 run notebook. |

## The simulation workflow

The simulation pipeline is organised around Jupyter notebooks that use the `primus_paleo` package. Parameters of a given run are:

- **the mineral** to be analysed (e.g. halite, olivine), taken from `Data/mineral_library.yaml`;
- **the geological history** of the sample: age, exposure window, and its overburden history (deposition and erosion phases, water columns, discrete events), which also sets its temperature history for annealing;
- **the astrophysical scenario** for the cosmic-ray flux (e.g. the standard flux, a nearby supernova, a geomagnetic excursion).

```python
import primus_paleo as pp

det = pp.Paleodetector("Halite", total_age_kyr=5970., overburden_history=history,
                       flux_history=pp.FluxHistory(events=[{"start_time_kyr": -5605., "template": {"kind": "SN", "distance_pc": 20.}}]))
x_bins, tracks = det.integrate_all_particles(exposure_window_kyr=[0., 640.])
x_bins, fission = det.integrate_fission_spectrum()
x_bins, neutrinos = det.integrate_atmospheric_neutrinos()
pits = pp.slice_spectrum_etch(tracks["total"], sample_area_cm2=0.02, sample_density_g_cm3=2.16)
```

Changing the scenario means editing the configuration, not the code. To reproduce a published result (with improvements!), open the corresponding notebook in `analysis/` and run it top to bottom.

## Data

Everything is read from and written to one data root, `Data/`, next to `src/`:

```
Data/
├── mineral_library.yaml
├── flux_data/                         flux templates
├── nuclear_data/                      fission and binding-energy tables
└── Geant4_data/
    ├── <mineral>/<particle>/          simulation outputs (outNuclei_<E>.txt) + geometry.json
    └── <mineral>/Range/               stopping-power tables
```

It is found automatically from any working directory, so notebooks can live anywhere under `analysis/`. To use another location, call `pp.set_data_path(...)` or set the environment variable `PRIMUS_DATA`; `pp.describe()` prints what is in use. Simulation outputs and flux templates are not included in the repository; they can be regenerated with the provided code (`pp.run_geant4_simulation`, `pp.export_all_stopping_powers`, `pp.FluxHistory`).

## Geant4 simulations

The Geant4 application in `geant4/` shoots primaries (muons, neutrons, protons, neutrinos, ions) into long targets of a mineral or of standard rock and records the nuclear recoils and secondary particles of their interactions. Build it once:

```bash
cd geant4 && mkdir -p build && cd build
cmake -DGeant4_DIR=$CONDA_PREFIX/lib/cmake/Geant4 .. && make -j
```

then run it from Python; the outputs are filed under `Data/Geant4_data/`:

```python
energies = np.logspace(-3, 4, 120)
pp.run_geant4_simulation("Halite", "mu-", 10000, energies[:90])
pp.run_geant4_simulation("StdRock", "nu_mu", 10000, energies[40:100], target_length_m=5000)   # biased cross-sections
pp.export_all_stopping_powers("Halite")
```

Each run records its geometry, number of events, neutrino bias and the code version that produced it in `geometry.json`.

## OptimusPrimus: track detection

`analysis/OptimusPrimus.py` is the image-analysis component of the experimental pipeline (plasma etching → optical microscopy → track detection → comparison with theory). It uses a dual U-Net architecture to identify tracks in microscope images. Main features:

- training and inference on image patches, with an option to keep patches in memory (with a RAM warning for large datasets);
- multi-GPU training through PyTorch `DataParallel`;
- safe checkpoint handling (explicit errors on missing paths, `weights_only=True` on all loads).

`analysis/Test_optimusprimus.ipynb` shows a typical end-to-end usage example. The comparison with theory uses `pp.slice_spectrum_etch`, which turns a simulated track spectrum into the etch-pit sizes seen on an etched surface.

## Installation

Clone the repository and install the package in editable mode (changes to `src/primus_paleo` are picked up without reinstalling):

```bash
git clone https://github.com/cgalelli/primus-paleo.git
cd primus-paleo
pip install -e ".[plot]"
pip install jupyter
```

Optional extras: `".[mceq]"` to build new flux templates (loading the saved ones does not need MCEq), `".[dev]"` to run the tests with `pytest`.

For **OptimusPrimus**, additionally install PyTorch following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your platform (a CUDA-enabled build is recommended for training; inference can be performed on CPU). As well as:

```bash
pip install scikit-learn scikit-image
```

Producing new simulations requires Geant4 (≥ 11.2 for the neutrino physics, e.g. from conda-forge) and, for new flux scenarios, MCEq. If only using the provided astrophysical flux scenarios and simulations, these are not necessary.

## Citation

If you use this code, please cite the software ([Zenodo, concept DOI 10.5281/zenodo.23320945](https://doi.org/10.5281/zenodo.23320945), see `CITATION.cff`) and the relevant paper(s).

**Chaîne des Puys olivine xenoliths** — *JCAP* 04 (2026) 023, [arXiv:2510.23126](https://arxiv.org/abs/2510.23126), [doi:10.1088/1475-7516/2026/04/023](https://doi.org/10.1088/1475-7516/2026/04/023):

```bibtex
@article{Galelli:2025gss,
    author        = "Galelli, Claudio and Caccianiga, Lorenzo and Apollonio, Lorenzo and Magnani, Paolo and Breton, Vincent",
    title         = "{A volcanic chronosequence as a time-resolved paleo-detector array to study the cosmic-ray flux in the late Pleistocene and Holocene}",
    journal       = "JCAP",
    volume        = "04",
    pages         = "023",
    year          = "2026",
    doi           = "10.1088/1475-7516/2026/04/023",
    eprint        = "2510.23126",
    archivePrefix = "arXiv",
    primaryClass  = "astro-ph.HE"
}
```

**Messinian halite** — *Phys. Rev. D* 110 (2024) L121301, [arXiv:2405.04908](https://arxiv.org/abs/2405.04908):

```bibtex
@article{Caccianiga:2024,
    author        = "Caccianiga, Lorenzo and others",
    title         = "{Sedimentary rocks from Mediterranean drought in the Messinian age as a probe of the past cosmic ray flux}",
    journal       = "Phys. Rev. D",
    volume        = "110",
    pages         = "L121301",
    year          = "2024",
    eprint        = "2405.04908",
    archivePrefix = "arXiv"
}
```

**The PRImuS project** — proceedings of ICRC 2025:

```bibtex
@inproceedings{Galelli:2025icrc,
    author    = "Galelli, Claudio and Apollonio, Lorenzo and Magnani, Paolo and Caccianiga, Lorenzo",
    title     = "{Probing Ancient Cosmic Ray Flux with Paleo-Detectors and the Launch of the PRI$\mu$S Project}",
    booktitle = "39th International Cosmic Ray Conference (ICRC 2025)",
    series    = "PoS",
    volume    = "ICRC2025",
    pages     = "262",
    year      = "2025"
}
```

## Acknowledgments

PRImuS is an INFN experiment funded by the CSN5 Young Scientist Grant.

## Contact

For questions, please contact Claudio Galelli: [claudio.galelli@mi.infn.it](mailto:claudio.galelli@mi.infn.it)
