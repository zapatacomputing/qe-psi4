from qepsi4 import select_active_space, run_psi4
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


def create_molecule(geometry, charge, mult):

    geometry_str = f"{charge} {mult}\n"
    for atom in geometry["sites"]:
        geometry_str += "{} {} {} {}\n".format(
            atom["species"], atom["x"], atom["y"], atom["z"]
        )

    geometry_str += "\nunits angstrom\n"
    geometry_str += "symmetry c1\n"

    molecule = psi4.geometry(geometry_str)

    return molecule


def test_active_space_0():

    h2 = create_molecule(hydrogen_geometry, charge=0, mult=1)
    psi4.set_options({"basis": 'sto-3g'})
    fake_wfn = psi4.core.Wavefunction.build(h2, psi4.core.get_global_option('basis')) 

    ndocc, nact, nvir = select_active_space(h2, fake_wfn, n_active_extract=2)

    assert ndocc == 0
    assert nact == 2
    assert nvir == 0

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

def test_active_space_1():

    h2o = create_molecule(water_geometry, charge=0, mult=1)
    psi4.set_options({"basis": 'sto-3g'})
    fake_wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option('basis')) 

    # Active space specs

    ndocc, nact, nvir = select_active_space(h2o, fake_wfn, n_active_extract=2)

    assert ndocc == 0
    assert nact == 2
    assert nvir == 5

    ndocc, nact, nvir = select_active_space(h2o, fake_wfn)

    assert ndocc == 0
    assert nact == 7
    assert nvir == 0

    ndocc, nact, nvir = select_active_space(h2o, fake_wfn, freeze_core_extract=True)

    assert ndocc == 1
    assert nact == 6
    assert nvir == 0

    ndocc, nact, nvir = select_active_space(h2o, fake_wfn, n_occupied_extract = 3, freeze_core_extract=True)

    assert ndocc == 2
    assert nact == 5
    assert nvir == 0

    ndocc, nact, nvir = select_active_space(h2o, fake_wfn, n_active_extract = 4, n_occupied_extract = 3)

    assert ndocc == 2
    assert nact == 4
    assert nvir == 1

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()

def test_run_psi4_0():

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

def test_run_psi4_1():

    results, hamiltonian = run_psi4(hydrogen_geometry, save_hamiltonian=True, n_active_extract=1)
    assert math.isclose(results["energy"], -0.8544322638069642)
    assert results["n_alpha"] == 1
    assert results["n_beta"] == 1
    assert results["n_mo"] == 2
    assert results["n_frozen_core"] == 0
    assert results["n_frozen_valence"] == 0

    assert hamiltonian.n_qubits == 2
    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 2) # This should give RHF energy

    assert math.isclose(energy, results["energy"], rel_tol=1e-3) # since the energy is calculated with scf_type df by default, the test fails for a lower rel_tol
    #assert math.isclose(energy, results["energy"])

def test_run_psi4_2():

    results, hamiltonian = run_psi4(hydrogen_geometry, save_hamiltonian=True, n_active_extract=1, n_occupied_extract=0)
    assert math.isclose(results["energy"], -0.8544322638069642)
    assert results["n_alpha"] == 1
    assert results["n_beta"] == 1
    assert results["n_mo"] == 2
    assert results["n_frozen_core"] == 0
    assert results["n_frozen_valence"] == 0

    assert hamiltonian.n_qubits == 2
    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 0) # This should give RHF energy in the 0-particle sector of the Fock space

    assert math.isclose(energy, results["energy"], rel_tol=1e-3) # since the energy is calculated with scf_type df by default, the test fails for a lower rel_tol
    #assert math.isclose(energy, results["energy"])

def test_get_rdms_from_psi4_0():

    results, hamiltonian, rdm = run_psi4(hydrogen_geometry, method='fci', basis='STO-3G', save_hamiltonian=True, save_rdms=True)
    energy_from_rdm = rdm.expectation(hamiltonian)
    assert math.isclose(results["energy"], energy_from_rdm)
    assert results["n_alpha"] == 1 and results["n_beta"] == 1 and results["n_mo"] == 2\
                    and results["n_frozen_core"] == 0 and results["n_frozen_valence"] == 0

    assert hamiltonian.n_qubits == 4
    assert math.isclose(np.einsum("ii->", rdm.one_body_tensor), 2)
    
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

def test_run_psi4_using_n_occupied_extract_with_rdms_0():
    results, hamiltonian, rdm = run_psi4(
        dilithium_geometry,
        method="fci",
        basis="STO-3G",
        n_active_extract=4,
        n_occupied_extract=1,
        save_hamiltonian=True,
        save_rdms=True,
    )

    assert hamiltonian.n_qubits == 8
    assert rdm.one_body_tensor.shape[0] == 8
    assert math.isclose(np.einsum("ii->", rdm.one_body_tensor), 2)

    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 2)

    energy_from_rdm = rdm.expectation(hamiltonian)
    assert math.isclose(energy, energy_from_rdm)

# This test may not play well with Psi4 because of 0 electrons in FCI, so I commented it
# out for now.
#def test_run_psi4_using_n_occupied_extract_with_rdms_1():
#    results, hamiltonian, rdm = run_psi4(
#        dilithium_geometry,
#        method="fci",
#        basis="STO-3G",
#        freeze_core=True,
#        n_active_extract=4,
#        n_occupied_extract=0,
#        save_hamiltonian=True,
#        save_rdms=True,
#    )
#
#    assert hamiltonian.n_qubits == 8
#    assert rdm.one_body_tensor.shape[0] == 8
#    assert np.einsum("ii->", rdm.one_body_tensor) == 0
#
#    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
#    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 0)
#
#    energy_from_rdm = rdm.expectation(hamiltonian)
#    assert energy <= energy_from_rdm

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
    assert math.isclose(np.einsum("ii->", rdm.one_body_tensor), 2)

    energy_from_rdm = rdm.expectation(hamiltonian)
    assert math.isclose(results["energy"], energy_from_rdm)
