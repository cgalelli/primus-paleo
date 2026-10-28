# Analysis

Notebooks (Messinian halite, olivine chronosequence, Geant4 runs, ...) and the OptimusPrimus track-detection code.
They use the installed package (`import primus_paleo as pp`) and find `Data/` on their own, so they can live in
subfolders here. To read a material: `pp.mineral_config("Halite")`, or pass the name directly to `pp.Paleodetector`.
