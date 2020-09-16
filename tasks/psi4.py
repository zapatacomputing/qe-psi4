import os, json
from qepsi4 import run_psi4
from qeopenfermion import save_interaction_operator
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
    n_active_extract="None",
    n_occupied_extract="None",
    freeze_core_extract=False,
    nthreads=1,
    n_active="None",
    options="None",
    wavefunction="None",
):
    os.mkdir("/app/scr")
    os.environ["PSI_SCRATCH"] = "/app/scr"

    with open(geometry) as f:
        geometry = json.load(f)

    results, hamiltonian = run_psi4(
        geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        method=method,
        reference="reference",
        freeze_core=freeze_core,
        n_active=n_active,
        save_hamiltonian=save_hamiltonian,
        options=options,
        n_active_extract=n_active_extract,
        n_occupied_extract=n_occupied_extract,
        freeze_core_extract=freeze_core_extract,
    )

    results["schema"] = SCHEMA_VERSION + "-energy_calc"
    with open("energycalc-results.json", "w") as f:
        f.write(json.dumps(results, indent=2))

    if save_hamiltonian:
        save_interaction_operator(hamiltonian, "hamiltonian.json")

    with open("n_alpha.txt", "w") as f:
        f.write(str(results["n_alpha"]))

    with open("n_beta.txt", "w") as f:
        f.write(str(results["n_beta"]))

    with open("n_mo.txt", "w") as f:
        f.write(str(results["n_mo"]))

    with open("n_frozen_core.txt", "w") as f:
        f.write(str(results["n_frozen_core"]))
