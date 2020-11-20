import os, json
from qepsi4 import run_psi4 as _run_psi4
from qeopenfermion import save_interaction_operator, save_interaction_rdm
from zquantum.core.utils import SCHEMA_VERSION


def run_psi4(
    basis,
    method,
    reference,
    geometry,
    freeze_core=False,
    charge=0,
    multiplicity=1,
    save_hamiltonian=False,
    save_rdms=False,
    n_active_extract="None",
    n_occupied_extract="None",
    freeze_core_extract=False,
    nthreads=1,
    options="None",
    wavefunction="None",
):
    os.mkdir("/app/scr")
    os.environ["PSI_SCRATCH"] = "/app/scr"

    if n_active_extract == "None":
        n_active_extract = None
    if n_occupied_extract == "None":
        n_occupied_extract = None
    if options == "None":
        options = None
    else:
        options = json.loads(options)
    if wavefunction == "None":
        wavefunction = None

    with open(geometry) as f:
        geometry = json.load(f)

    res = _run_psi4(
        geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        method=method,
        reference=reference,
        freeze_core=freeze_core,
        save_hamiltonian=save_hamiltonian,
        save_rdms=save_rdms,
        options=options,
        n_active_extract=n_active_extract,
        n_occupied_extract=n_occupied_extract,
        freeze_core_extract=freeze_core_extract,
    )

    results = res['results']
    results["schema"] = SCHEMA_VERSION + "-energy_calc"
    with open("energycalc-results.json", "w") as f:
        f.write(json.dumps(results, indent=2))

    hamiltonian = res.get('hamiltonian', None)
    if hamiltonian is not None:
        save_interaction_operator(hamiltonian, "hamiltonian.json")

    rdms = res.get('rdms', None)
    if rdms is not None:
        save_interaction_rdm(rdms, "rdms.json")

    with open("n_alpha.txt", "w") as f:
        f.write(str(results["n_alpha"]))

    with open("n_beta.txt", "w") as f:
        f.write(str(results["n_beta"]))

    with open("n_mo.txt", "w") as f:
        f.write(str(results["n_mo"]))

    with open("n_frozen_core.txt", "w") as f:
        f.write(str(results["n_frozen_core"]))
