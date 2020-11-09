from qepsi4 import run_psi4
from openfermion import (
    jordan_wigner,
    jw_get_ground_state_at_particle_number,
    qubit_operator_sparse,
)
import numpy as np
import math
import psi4

hydrogen_geometry = {
    "sites": [
        {"species": "H", "x": 0, "y": 0, "z": 0},
        {"species": "H", "x": 0, "y": 0, "z": 1.7},
    ]
}

dilithium_geometry = {
    "sites": [
        {"species": "Li", "x": 0, "y": 0, "z": 0},
        {"species": "Li", "x": 0, "y": 0, "z": 2.67},
    ]
}


water_geometry = {
    "sites": [
        {"species": "O", "x": 0, "y": 0, "z": -0.0454827817},
        {"species": "H", "x": -0.7634678739, "y": 0, "z": 0.5512397709},
        {"species": "H", "x":  0.7634678739, "y": 0, "z": 0.5512397709},
    ]
}

def test_run_psi4():

    results, hamiltonian = run_psi4(hydrogen_geometry, save_hamiltonian=True)
    assert math.isclose(results["energy"], -0.8544322638069642)
    assert results["n_alpha"] == 1
    assert results["n_beta"] == 1
    assert results["n_mo"] == 2
    assert results["n_frozen_core"] == 0
    assert results["n_frozen_valence"] == 0

    assert hamiltonian.n_qubits == 4
    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 2)

    results_cisd, hamiltonian = run_psi4(hydrogen_geometry, method="ccsd")

    # For this system, the CCSD energy should be exact.
    assert math.isclose(energy, results_cisd["energy"], rel_tol=1e-7)

def test_get_rdms_from_psi4():

    results, hamiltonian, rdm = run_psi4(hydrogen_geometry, method='fci', basis='STO-3G', save_hamiltonian=True, save_rdms=True)
    energy_from_rdm = rdm.expectation(hamiltonian)
    assert math.isclose(results["energy"], energy_from_rdm)
    assert results["n_alpha"] == 1 and results["n_beta"] == 1 and results["n_mo"] == 2\
                    and results["n_frozen_core"] == 0 and results["n_frozen_valence"] == 0 and hamiltonian.n_qubits == 4

    assert hamiltonian.n_qubits == 4
    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 2)

    assert math.isclose(energy, energy_from_rdm)

def test_run_psi4_using_n_occupied_extract():
    results, hamiltonian = run_psi4(
        dilithium_geometry,
        method="scf",
        basis="STO-3G",
        n_active_extract=4,
        n_occupied_extract=2,
        save_hamiltonian=True,
    )

    assert hamiltonian.n_qubits == 8

    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 4)
    assert math.isclose(energy, -14.654620243980217)

def test_run_psi4_using_n_occupied_extract_with_rdms():
    results, hamiltonian, rdm = run_psi4(
        dilithium_geometry,
        method="fci",
        basis="STO-3G",
        n_active_extract=4,
        n_occupied_extract=2,
        save_hamiltonian=True,
        save_rdms=True,
    )

    assert hamiltonian.n_qubits == 8
    assert rdm.one_body_tensor.shape[0] == 8

    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 4)

    energy_from_rdm = rdm.expectation(hamiltonian)
    assert energy <= energy_from_rdm

def test_run_psi4_n_occupied_extract_inconsistent_with_n_active_extract():
    try:
        run_psi4(
            dilithium_geometry,
            method="scf",
            basis="STO-3G",
            n_active_extract=2,
            n_occupied_extract=3,
            save_hamiltonian=True,
        )
    except ValueError:
        pass
    else:
        assert False


def test_run_psi4_n_occupied_extract_inconsistent_with_num_electrons():
    try:
        run_psi4(
            hydrogen_geometry,
            method="scf",
            basis="6-31G",
            n_active_extract=4,
            n_occupied_extract=2,
            save_hamiltonian=True,
        )
    except ValueError:
        pass
    else:
        assert False


def test_run_psi4_freeze_core_extract():
    # For some reason, we must clean Psi4 before running this test if other
    # tests have already run.
    psi4.core.clean()
    results, hamiltonian = run_psi4(
        dilithium_geometry,
        method="scf",
        basis="STO-3G",
        freeze_core=True,
        freeze_core_extract=True,
        save_hamiltonian=True,
    )

    # With STO-3G, each Li atom has one 1s orbital, one 2s orbital, and three 2p orbitals. The 1s orbitals are considered core orbitals.
    assert hamiltonian.n_qubits == 2 * 2 * (1 + 3)


def test_run_psi4_freeze_core_extract_and_rdms():
    # For some reason, we must clean Psi4 before running this test if other
    # tests have already run.
    psi4.core.clean()
    results, hamiltonian, rdm = run_psi4(
        dilithium_geometry,
        method="fci",
        basis="STO-3G",
        freeze_core=True,
        freeze_core_extract=True,
        save_hamiltonian=True,
        save_rdms=True
    )

    # With STO-3G, each Li atom has one 1s orbital, one 2s orbital, and three 2p orbitals. The 1s orbitals are considered core orbitals.
    assert hamiltonian.n_qubits == 2 * 2 * (1 + 3)
    assert rdm.one_body_tensor.shape[0] == 2 * 2 * (1 + 3)

    energy_from_rdm = rdm.expectation(hamiltonian)
    assert math.isclose(results["energy"], energy_from_rdm)
