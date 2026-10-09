# primus-paleo

**Simulation and analysis tools for cosmic-ray paleo-detectors**

This repository contains the Python code behind **PRImuS** (*Paleo-astroparticles Reconstructed with the Interactions of MUons in Stone*), an INFN experiment that aims to reconstruct past cosmic-ray fluxes from the nuclear-recoil tracks induced by cosmic-ray muons in natural minerals.

It collects, in one place:

- the **simulation framework** that predicts the expected track rate and track-length distribution in a mineral, given its geological history and an assumed cosmic-ray flux scenario;
- the **analysis notebooks** behind our published phenomenological studies (Messinian halite and Chaîne des Puys olivine xenoliths);
- **OptimusPrimus**, the machine-learning pipeline for detecting tracks in microscope images, used for the experimental side of PRImuS.

## Repository contents

| File | Description |
|---|---|
| `mineral_utils.py` | Core utility module: mineral properties and the computation of expected nuclear-recoil track rates. |
| `flux_history.py` | Time-dependent cosmic-ray flux scenarios (`FluxHistory`) and the overburden history of a sample (`overburden_history`). |
| `OptimusPrimus.py` | Dual U-Net (PyTorch) pipeline for track detection in microscope images. |
| `CDP_analysis_new.ipynb` | Analysis of olivine xenoliths from the Chaîne des Puys volcanic chronosequence. |
| `MSC_analysis_new.ipynb` | Analysis of halite evaporites from the Messinian salinity crisis. |
| `Test_optimusprimus.ipynb` | Functional test notebook for OptimusPrimus. |

## The simulation workflow

The simulation pipeline is organised around Jupyter notebooks that interact with the core modules `mineral_utils.py` and `flux_history.py`. Parameters of a given run are:

- **the mineral** to be analysed (e.g. halite, olivine);
- **the geological history** of the sample (age, exposure time, deposition rate, overburden);
- **the astrophysical scenario** for the cosmic-ray flux (e.g. the standard flux, or an enhanced flux from a nearby supernova).

Changing the scenario means editing the configuration, not the code. To reproduce a published result (with improvements!), open the corresponding notebook and run it top to bottom.

## OptimusPrimus: track detection

`OptimusPrimus.py` is the image-analysis component of the experimental pipeline (plasma etching → optical microscopy → track detection → comparison with theory). It uses a dual U-Net architecture to identify tracks in microscope images. Main features:

- training and inference on image patches, with an option to keep patches in memory (with a RAM warning for large datasets);
- multi-GPU training through PyTorch `DataParallel`;
- safe checkpoint handling (explicit errors on missing paths, `weights_only=True` on all loads).

`Test_optimusprimus.ipynb` shows a typical end-to-end usage example.

## Installation

Clone the repository:

```bash
git clone https://github.com/cgalelli/primus-paleo.git
cd primus-paleo
```

### Dependencies

For the **simulation notebooks**:

```bash
pip install numpy scipy matplotlib mendeleev pyyaml jupyter
```

For **OptimusPrimus**, additionally install PyTorch following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) for your platform (a CUDA-enabled build is recommended for training; inference can be performed on CPU). As well as:

```bash
pip install scikit-learn scikit-image
```

Some steps of the full simulation chain rely on external tools (MCEq, Geant4), which must be installed separately. If only using the provided astrophysical flux scenarios and minerals, these are not necessary.

## Citation

If you use this code or the associated results in your research, please cite the relevant paper(s).

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
