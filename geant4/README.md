# Geant4 application

Put the C++ sources of the application here (CMakeLists.txt, the main program, src/, include/, G4DataPaths.hh.in),
and build it in `geant4/build`:

```bash
conda activate paleo
cd geant4 && mkdir -p build && cd build
cmake -DGeant4_DIR=$CONDA_PREFIX/lib/cmake/Geant4 .. && make -j
```

`primus_paleo.run_geant4_simulation` and `export_all_stopping_powers` start `build/Run` with `geant4/` as working
directory (so `build/output/` keeps working as before), write `build/materials.mac` and `build/run.in` there, and
file the results under `Data/Geant4_data/`. Until the application is moved here, the old build in
`Data/Geant4_data/build/Run` is used.
