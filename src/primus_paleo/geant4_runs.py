"""Running the Geant4 application that produces the simulations read by Paleodetector: materials macro, runs per
material and particle (with neutrino biasing), geometry sidecars and stopping-power tables.

Layout (see config for how the two roots are found):
  <data>/mineral_library.yaml                 materials (density_g_cm3, nuclei, stoich)
  <data>/Geant4_data/<material>/<particle>/   outNuclei_<E>.txt + geometry.json, read by Paleodetector
  <data>/Geant4_data/<material>/Range/        DEDX_Z<z>_A<a>.txt stopping-power tables
  <app>/build/Run                             compiled Geant4 application (<app> = <repository>/geant4, or
                                              <data>/Geant4_data in the layout before the package)
  <app>/build/materials.mac, run.in, *.log    generated macros and logs
  <app>/build/output/                         where Run writes outNuclei_<i>.txt (i = run index)
The application is always started from <app>, so the relative paths it uses internally (build/output) do not depend
on where Python runs; every path passed to it is absolute.
"""

import json
import os

import numpy as np
from mendeleev import element
from tqdm import tqdm

from . import config
from ._core import (
    NEUTRINO_SPECIES,
)

__all__ = [
    'GEANT4_HARDCODED_DENSITIES_G_CM3', 'GEANT4_DEFAULT_TARGET_LENGTH_M', 'GEANT4_TARGET_LENGTH_M',
    'GEANT4_TARGET_RADIUS_M', 'GEANT4_PRODUCTION_CUT_MM', 'GEANT4_NEUTRINO_REGION',
    'NEUTRINO_TARGET_INTERACTION_FRACTION', 'geant4_paths', 'add_material', 'geant4_environment',
    'write_materials_macro', 'material_density_g_cm3', 'suggested_neutrino_bias', 'run_geant4_simulation',
    'get_all_known_isotopes', 'export_all_stopping_powers',
]

GEANT4_HARDCODED_DENSITIES_G_CM3 = {"StdRock": 2.65, "H2O": 1.0}     # materials defined in DetectorConstruction
GEANT4_DEFAULT_TARGET_LENGTH_M = 200.
GEANT4_TARGET_LENGTH_M = {"StdRock": 1000.}       # longer StdRock targets calibrate the deep penetration
GEANT4_TARGET_RADIUS_M = 10.
GEANT4_PRODUCTION_CUT_MM = 0.005
GEANT4_NEUTRINO_REGION = "TargetRegion"           # region of the biased neutrino cross-sections (DetectorConstruction)
NEUTRINO_TARGET_INTERACTION_FRACTION = 0.05       # biased interactions per primary aimed at by suggested_neutrino_bias


def geant4_paths(data_path=None, geant4_dir=None):
    """Absolute paths of the Geant4 layout above.

    Purpose: the single place where the Geant4 folders are built, so that the run functions work from any working
    directory (notebook, script, other folder).

    Args:
        data_path (str, optional): Data root. Defaults to None (config.data_path()).
        geant4_dir (str, optional): Folder of the application, the one containing build/Run. Defaults to None
            (config.geant4_app_path()).

    Returns:
        dict: 'data', 'library' (mineral_library.yaml), 'geant4' (<data>/Geant4_data, where the results go), 'app'
        (working directory of the application), 'build', 'executable', 'materials_macro', 'output' (raw output of
        the application).
    """
    data = config.data_path(data_path)
    app = config.geant4_app_path(geant4_dir, data)
    build = os.path.join(app, "build")
    return {
        'data': data,
        'library': config.mineral_library_path(data),
        'geant4': os.path.join(data, "Geant4_data"),
        'app': app,
        'build': build,
        'executable': os.path.join(build, "Run"),
        'materials_macro': os.path.join(build, "materials.mac"),
        'output': os.path.join(build, "output"),
    }

def add_material(name, nuclei, stoich, density_g_cm3,
                 uranium_concentration_g_g=2.8e-6, an_coeff_n_alpha=1, data_path=None):
    """
    Adds a new material to mineral_library
    """
    import yaml

    file_path = config.mineral_library_path(data_path)

    with open(file_path, "r") as file:
        data = yaml.safe_load(file) or {}

    new_material = {
        "name": name,
        "nuclei": nuclei,
        "stoich": stoich,
        "uranium_concentration_g_g": uranium_concentration_g_g,
        "density_g_cm3": density_g_cm3,
        "an_coeff_n_nalpha": an_coeff_n_alpha,
    }

    data[name] = new_material

    with open(file_path, "w") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return new_material

def geant4_environment():
    """Environment for the Geant4 application, with the dataset variables (G4ENSDFSTATEDATA, G4LEDATA, ...) filled in.

    Purpose: Python started outside an activated environment (Jupyter kernel, IDE) does not have the G4*DATA
    variables, and the application then aborts ("G4ENSDFSTATEDATA environment variable must be set"). Variables
    already set are kept; the missing ones are taken from `geant4-config --datasets` (on PATH or in the bin/ of the
    running Python's environment), then from the conda activation scripts of that environment. Not needed if the
    application was built with the dataset paths compiled in (G4DataPaths.hh), but harmless.

    Returns:
        dict: Copy of os.environ with the Geant4 dataset variables added where they were missing.
    """
    import shutil
    import subprocess
    import sys
    import glob

    env = dict(os.environ)
    found = {}
    exe = shutil.which("geant4-config") or shutil.which("geant4-config", path=os.path.join(sys.prefix, "bin"))
    if exe is not None:
        try:
            out = subprocess.run([exe, "--datasets"], capture_output=True, text=True, check=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 3 and parts[1].startswith("G4") and os.path.isdir(parts[2]):
                    found[parts[1]] = parts[2]
        except (OSError, subprocess.CalledProcessError):
            pass
    scripts = sorted(glob.glob(os.path.join(sys.prefix, "etc", "conda", "activate.d", "*.sh")))
    if scripts and shutil.which("bash"):
        cmd = (f"export CONDA_PREFIX='{sys.prefix}'; " + "".join(f"source '{s}' >/dev/null 2>&1; " for s in scripts)
               + "env -0")
        try:
            out = subprocess.run(["bash", "-c", cmd], capture_output=True, check=True).stdout.decode()
            for kv in out.split("\0"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    if k.startswith("G4") and k not in found:
                        found[k] = v
        except (OSError, subprocess.CalledProcessError):
            pass
    for k, v in found.items():
        env.setdefault(k, v)
    return env


def write_materials_macro(data_path=None, yaml_path=None, out_path=None, verbose=True):
    """Writes the Geant4 macro defining every material of the mineral library.

    One /testhadr/defineMaterial command per material: "<name> <density_g_cm3> <El1> <n1> ...", stoich being relative
    atom counts (DetectorConstruction converts them to mass fractions).

    Args:
        data_path (str, optional): Data root. Defaults to None (config.data_path()).
        yaml_path (str, optional): Mineral library. Defaults to <data>/mineral_library.yaml.
        out_path (str, optional): Macro written. Defaults to <data>/Geant4_data/build/materials.mac.
        verbose (bool, optional): Print what was written. Defaults to True.

    Returns:
        list[str]: Names of the materials defined.

    Raises:
        ValueError: If a material has different numbers of nuclei and stoich entries.
    """
    import yaml

    paths = geant4_paths(data_path)
    yaml_path = paths['library'] if yaml_path is None else os.path.abspath(yaml_path)
    out_path = paths['materials_macro'] if out_path is None else os.path.abspath(out_path)
    with open(yaml_path) as f:
        library = yaml.safe_load(f)

    lines = [f"# Auto-generated from {yaml_path} -- do not edit by hand"]
    for key, m in library.items():
        if len(m["nuclei"]) != len(m["stoich"]):
            raise ValueError(f"{key}: nuclei and stoich have different lengths")
        comps = " ".join(f"{el} {n}" for el, n in zip(m["nuclei"], m["stoich"]))
        lines.append(f"/testhadr/defineMaterial {key} {m['density_g_cm3']} {comps}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    if verbose:
        print(f"Wrote {len(library)} material definitions to {out_path}")
    return sorted(library)


def material_density_g_cm3(mineral_name, data_path=None, yaml_path=None):
    """Density used for a material in Geant4 [g/cm^3], or None if unknown.

    Args:
        mineral_name (str): Material name (library key, or a material hardcoded in DetectorConstruction).
        data_path (str, optional): Data root. Defaults to None (config.data_path()).
        yaml_path (str, optional): Mineral library. Defaults to <data>/mineral_library.yaml.

    Returns:
        float or None: Density [g/cm^3].
    """
    import yaml

    if mineral_name in GEANT4_HARDCODED_DENSITIES_G_CM3:
        return GEANT4_HARDCODED_DENSITIES_G_CM3[mineral_name]
    yaml_path = geant4_paths(data_path)['library'] if yaml_path is None else yaml_path
    with open(yaml_path) as f:
        library = yaml.safe_load(f)
    entry = library.get(mineral_name)
    return float(entry["density_g_cm3"]) if entry else None


def suggested_neutrino_bias(energy_gev, target_length_m, density_g_cm3,
                            target_fraction=NEUTRINO_TARGET_INTERACTION_FRACTION):
    """Cross-section bias giving about `target_fraction` biased interactions per neutrino along the target.

    Purpose: an unbiased neutrino essentially never interacts in a Geant4 target; the cross-sections are multiplied
    by a bias inside the target region. Above ~0.1 of interactions per primary the beam is depleted along the
    target and the rate per unit length is no longer uniform (integrate_atmospheric_neutrinos reports this), so the
    bias is chosen per energy from a rough total nu-nucleon cross-section, sigma ~ 0.7e-38 cm^2 x max(E/GeV, 0.5)
    (deep-inelastic scaling, levelled off in the quasi-elastic region). Only the order of magnitude matters: the
    runs record the bias used, and the code divides by it.

    Args:
        energy_gev (float or array_like): Neutrino energy [GeV].
        target_length_m (float): Target length [m].
        density_g_cm3 (float): Target density [g/cm^3].
        target_fraction (float, optional): Biased interactions per primary aimed at. Defaults to
            NEUTRINO_TARGET_INTERACTION_FRACTION.

    Returns:
        float or np.ndarray: Bias (>= 1), rounded to one significant digit.
    """
    e = np.asarray(energy_gev, dtype=float)
    sigma_cm2 = 0.7e-38 * np.maximum(e, 0.5)
    nucleons_cm3 = density_g_cm3 * 6.022e23
    p_true = sigma_cm2 * nucleons_cm3 * target_length_m * 100.0
    bias = np.maximum(target_fraction / p_true, 1.0)
    exponent = np.floor(np.log10(bias))
    bias = np.round(bias / 10.0 ** exponent) * 10.0 ** exponent
    return float(bias) if bias.ndim == 0 else bias


def _code_version():
    """Version of primus_paleo and, when installed from a git repository, its commit (recorded with every run)."""
    import subprocess
    from importlib.metadata import PackageNotFoundError, version
    try:
        out = {"primus_paleo": version("primus-paleo")}
    except PackageNotFoundError:
        out = {"primus_paleo": "unknown"}
    root = config.repository_root()
    if root is not None:
        try:
            commit = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True,
                                   text=True, check=True).stdout.strip()
            out["git_commit"] = commit + ("-dirty" if dirty else "")
        except (OSError, subprocess.CalledProcessError):
            pass
    return out


def _update_geometry_json(path, geometry, energies_gev, n_events, biases=None, verbose=True):
    """Writes geometry.json of a run folder, merging with the runs already there.

    Several batches of energies can go into the same folder (e.g. energies[:90], then energies[90:105]); the
    per-energy entries (n_events, neutrino_bias) are merged, and a change of target geometry between batches, which
    would mix incompatible depth scales, is reported.
    """
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        for key in ("target_length_m", "target_radius_m", "density_g_cm3"):
            if key in old and old[key] is not None and geometry.get(key) is not None \
                    and not np.isclose(old[key], geometry[key]) and verbose:
                print(f"Warning: {path}: {key} changes from {old[key]} to {geometry[key]}; the runs already in the "
                      "folder were made with the old value.")

    def as_dict(value):
        return dict(value) if isinstance(value, dict) else {}

    new = dict(old, **geometry)
    n_ev = as_dict(old.get("n_events"))
    n_ev.update({f"{e:.6f}": int(n_events) for e in energies_gev})
    new["n_events"] = n_ev
    if biases is not None:
        nb = as_dict(old.get("neutrino_bias"))
        nb.update({f"{e:.6f}": float(b) for e, b in zip(energies_gev, biases)})
        new["neutrino_bias"] = nb
    with open(path, "w") as f:
        json.dump(new, f, indent=2)


def run_geant4_simulation(mineral_name, particle_type, n_events, energies_gev, ion_type=None,
                          target_length_m=None, target_radius_m=GEANT4_TARGET_RADIUS_M, neutrino_bias=None,
                          production_cut_mm=GEANT4_PRODUCTION_CUT_MM, data_path=None, geant4_dir=None, verbose=True):
    """Runs the Geant4 application for one material and primary at a list of energies, and files the output.

    For each energy E, n_events primaries are shot along the axis of the target cylinder (from its top face) and the
    application writes the recorded particles of the primary's interactions inside the target (SteppingAction). The
    files go to <data>/Geant4_data/<material>/<particle>/outNuclei_<E as %.6f>.txt, with a geometry.json read by
    Paleodetector (target length and radius, depth of the top face, density, n_events per energy and, for neutrinos,
    the bias per energy). Batches of energies can be added to an existing folder.

    Neutrinos (NEUTRINO_SPECIES, Geant4 names): the cross-sections are biased in the target region
    (/physics_lists/nu/... commands, read before /run/initialize, so one application run per bias value):
    neutrino_bias None (suggested_neutrino_bias per energy), a number, a sequence (one per energy) or a function
    f(E_gev). The folder is named after the Geant4 particle, as NEUTRINO_SPECIES.

    Ions: particle_type "ion" and ion_type = (Z, A); the folder is ion_Z<Z>_A<A>.

    Args:
        mineral_name (str): Target material (mineral library key, StdRock or H2O).
        particle_type (str): Geant4 particle name ("mu-", "proton", "neutron", "nu_mu", "anti_nu_e", "ion", ...).
        n_events (int): Primaries per energy.
        energies_gev (array_like): Kinetic energies [GeV].
        ion_type (tuple[int, int], optional): (Z, A) for particle_type "ion".
        target_length_m (float, optional): Target length [m]. Defaults to GEANT4_TARGET_LENGTH_M[mineral] or
            GEANT4_DEFAULT_TARGET_LENGTH_M.
        target_radius_m (float, optional): Target radius [m]. Defaults to GEANT4_TARGET_RADIUS_M.
        neutrino_bias (float, sequence, callable or None, optional): Neutrino cross-section bias (see above).
        production_cut_mm (float, optional): Geant4 production cut [mm]. Defaults to GEANT4_PRODUCTION_CUT_MM.
        data_path (str, optional): Data root (Paleodetector's data_path_prefix). Defaults to None (config.data_path()).
        geant4_dir (str, optional): Folder of the application (containing build/Run). Defaults to None
            (config.geant4_app_path()).
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        str: Output folder.

    Raises:
        FileNotFoundError: If the application or the mineral library is missing.
        ValueError: On invalid geometry, ion or bias arguments.
        RuntimeError: If the application fails (its error output and the log path are in the message).
    """
    import shutil
    import subprocess

    paths = geant4_paths(data_path, geant4_dir)
    if not os.path.exists(paths['executable']):
        raise FileNotFoundError(f"Geant4 application not found: {paths['executable']} (compile it in {paths['build']}).")
    energies_gev = np.atleast_1d(np.asarray(energies_gev, dtype=float))
    if target_length_m is None:
        target_length_m = GEANT4_TARGET_LENGTH_M.get(mineral_name, GEANT4_DEFAULT_TARGET_LENGTH_M)
    if target_length_m <= 0 or target_radius_m <= 0:
        raise ValueError("target_length_m and target_radius_m must be > 0")

    if particle_type == "ion":
        if ion_type is None or len(ion_type) != 2:
            raise ValueError("particle_type 'ion' needs ion_type = (Z, A)")
        folder = f"ion_Z{int(ion_type[0])}_A{int(ion_type[1])}"
    else:
        folder = particle_type
    final_output_dir = os.path.join(paths['geant4'], mineral_name, folder)
    os.makedirs(paths['output'], exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    write_materials_macro(data_path, out_path=paths['materials_macro'], verbose=False)
    density = material_density_g_cm3(mineral_name, data_path)

    # bias per energy (neutrinos only) and grouping into application runs with one bias each
    is_neutrino = particle_type in NEUTRINO_SPECIES
    if is_neutrino:
        if neutrino_bias is None:
            if density is None:
                raise ValueError(f"Unknown density of {mineral_name}: pass neutrino_bias explicitly.")
            biases = np.atleast_1d(suggested_neutrino_bias(energies_gev, target_length_m, density))
        elif callable(neutrino_bias):
            biases = np.array([float(neutrino_bias(e)) for e in energies_gev])
        else:
            biases = np.broadcast_to(np.asarray(neutrino_bias, dtype=float), energies_gev.shape).copy()
        if np.any(biases <= 0):
            raise ValueError("neutrino_bias must be > 0")
        groups, start = [], 0
        for i in range(1, energies_gev.size + 1):
            if i == energies_gev.size or biases[i] != biases[start]:
                groups.append((start, i))
                start = i
    else:
        biases = None
        groups = [(0, energies_gev.size)]

    if verbose:
        what = particle_type if particle_type != "ion" else f"ion Z={ion_type[0]} A={ion_type[1]}"
        print(f"Geant4: {mineral_name}, {what}, {energies_gev.size} energies x {n_events} events, "
              f"target {target_length_m:g} m" + (f", {len(groups)} application runs (one per bias)" if is_neutrino else ""))

    env = geant4_environment()
    macro_path = os.path.join(paths['build'], "run.in")
    log_path = os.path.join(paths['build'], f"run_{mineral_name}_{folder}.log")
    with open(log_path, "w") as log:
        for g0, g1 in groups:
            for old in os.listdir(paths['output']):            # stale files would be filed as this run's output
                if old.startswith("outNuclei_") and old.endswith(".txt"):
                    os.remove(os.path.join(paths['output'], old))

            lines = ["# Auto-generated macro file", "/control/verbose 0", "/run/verbose 1", "/tracking/verbose 0", "",
                     "# --- Materials ---", f"/control/execute {paths['materials_macro']}", "",
                     "# --- Geometry ---", f"/testhadr/TargetMat {mineral_name}",
                     f"/testhadr/TargetRadius {target_radius_m} m", f"/testhadr/TargetLength {target_length_m} m", "",
                     "# --- Physics ---", f"/run/setCut {production_cut_mm} mm"]
            if is_neutrino:
                b = biases[g0]
                lines += [f"/physics_lists/nu/NuDetectorName {GEANT4_NEUTRINO_REGION}",
                          f"/physics_lists/nu/NuNucleusBias {b:g}",
                          f"/physics_lists/nu/NuEleCcBias {b:g}",
                          f"/physics_lists/nu/NuEleNcBias {b:g}"]
            lines += ["/run/initialize", "", "# --- Run loop ---", f"/gun/particle {particle_type}"]
            if particle_type == "ion":
                lines.append(f"/gun/ion {int(ion_type[0])} {int(ion_type[1])} {int(ion_type[0])} 0.0")
            for energy in energies_gev[g0:g1]:
                lines += [f"/gun/energy {energy:.16f} GeV", f"/run/beamOn {int(n_events)}"]
            with open(macro_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            result = subprocess.run([paths['executable'], macro_path], cwd=paths['app'], env=env,
                                    capture_output=True, text=True)
            log.write(result.stdout)
            log.write(result.stderr)
            if result.returncode != 0:
                raise RuntimeError(f"Geant4 failed (exit code {result.returncode}) for {mineral_name}/{folder}; "
                                   f"full log in {log_path}.\n{result.stderr[-3000:]}")

            missing = []
            for i, energy in enumerate(energies_gev[g0:g1]):
                src = os.path.join(paths['output'], f"outNuclei_{i}.txt")
                if os.path.exists(src):
                    shutil.move(src, os.path.join(final_output_dir, f"outNuclei_{energy:.6f}.txt"))
                else:
                    missing.append(energy)
            if missing and verbose:
                print(f"Warning: no output for {len(missing)} energies (e.g. {missing[:3]} GeV); see {log_path}.")

    # geometry sidecar read by Paleodetector: the gun sits on the target top face, depth = depth_reference_z_m - z
    geometry = {
        "target_length_m": float(target_length_m),
        "target_radius_m": float(target_radius_m),
        "depth_reference_z_m": float(target_length_m) / 2.0,
        "density_g_cm3": density,
        "code_version": _code_version(),
    }
    _update_geometry_json(os.path.join(final_output_dir, "geometry.json"), geometry, energies_gev, n_events,
                          biases=biases, verbose=verbose)
    if verbose:
        print(f"Done: {final_output_dir}")
    return final_output_dir


def get_all_known_isotopes(max_z=65):
    """(Z, A) of every isotope documented in mendeleev up to max_z (the recoils whose ranges may be needed)."""
    isotopes = set()
    for z in range(1, max_z + 1):
        for iso in element(z).isotopes:
            isotopes.add((z, iso.mass_number))
    return sorted(isotopes)


def export_all_stopping_powers(mineral_name, max_z=65, data_path=None, geant4_dir=None, verbose=True):
    """Exports the dE/dx and range tables of every known isotope up to max_z in a material.

    Runs the Geant4 application in its bypass mode (--export-dedx <material> <Z> <A> <out_dir> <materials macro>)
    once per isotope; the tables DEDX_Z<z>_A<a>.txt go to <data>/Geant4_data/<material>/Range, where
    Paleodetector reads them to convert recoil energies into track lengths.

    Args:
        mineral_name (str): Material (mineral library key).
        max_z (int, optional): Highest atomic number. Defaults to 65.
        data_path (str, optional): Data root. Defaults to None (config.data_path()).
        geant4_dir (str, optional): Folder of the application (containing build/Run). Defaults to None
            (config.geant4_app_path()).
        verbose (bool, optional): Print progress and errors. Defaults to True.

    Returns:
        list[tuple[int, int]]: Isotopes whose export failed.

    Raises:
        FileNotFoundError: If the application is missing.
    """
    import subprocess

    paths = geant4_paths(data_path, geant4_dir)
    if not os.path.exists(paths['executable']):
        raise FileNotFoundError(f"Geant4 application not found: {paths['executable']} (compile it in {paths['build']}).")
    output_dir = os.path.join(paths['geant4'], mineral_name, "Range")
    os.makedirs(output_dir, exist_ok=True)
    write_materials_macro(data_path, out_path=paths['materials_macro'], verbose=False)
    env = geant4_environment()

    isotopes = get_all_known_isotopes(max_z)
    if verbose:
        print(f"Exporting stopping-power tables for {len(isotopes)} isotopes in {mineral_name}...")
    failed = []
    for Z, A in tqdm(isotopes, disable=not verbose):
        result = subprocess.run(
            [paths['executable'], "--export-dedx", mineral_name, str(Z), str(A), output_dir, paths['materials_macro']],
            cwd=paths['app'], env=env, capture_output=True, text=True,
        )
        if result.returncode != 0:
            failed.append((Z, A))
            if verbose:
                print(f"Error exporting Z={Z}, A={A}:\n{result.stderr[-1000:]}")
    if verbose:
        print(f"Done: tables in {output_dir}" + (f" ({len(failed)} failed)" if failed else ""))
    return failed
