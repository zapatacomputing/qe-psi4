from qepsi4 import select_active_space, run_psi4
from openfermion import (
    jordan_wigner,
    jw_get_ground_state_at_particle_number,
    qubit_operator_sparse,
)
from numpy import einsum
import math
import psi4
import pytest

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
        {"species": "H", "x": 0.7634678739, "y": 0, "z": 0.5512397709},
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


class TestActiveSpace:
    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        psi4.set_options({"basis": "sto-3g"})

        yield

        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()

    @pytest.fixture
    def water_mol_and_wfn(self):
        h2o = create_molecule(water_geometry, charge=0, mult=1)
        h2o_wfn = psi4.core.Wavefunction.build(
            h2o, psi4.core.get_global_option("basis")
        )

        return h2o, h2o_wfn

    def test_active_space_hydrogen(self):
        h2 = create_molecule(hydrogen_geometry, charge=0, mult=1)
        fake_wfn = psi4.core.Wavefunction.build(
            h2, psi4.core.get_global_option("basis")
        )

        ndocc, nact, nvir = select_active_space(h2, fake_wfn, n_active_extract=2)

        assert ndocc == 0
        assert nact == 2
        assert nvir == 0

    @pytest.mark.parametrize(
        "n_active_extract,n_occupied_extract,freeze_core_extract,expected",
        [
            (2, None, False, (0, 2, 5)),
            (None, None, False, (0, 7, 0)),
            (None, None, True, (1, 6, 0)),
            (None, 3, True, (2, 5, 0)),
            (4, 3, False, (2, 4, 1)),
        ],
    )
    def test_active_space(
        self,
        water_mol_and_wfn,
        n_active_extract,
        n_occupied_extract,
        freeze_core_extract,
        expected,
    ):
        ndocc, nact, nvir = select_active_space(
            *water_mol_and_wfn,
            n_active_extract=n_active_extract,
            n_occupied_extract=n_occupied_extract,
            freeze_core_extract=freeze_core_extract,
        )

        assert (ndocc, nact, nvir) == expected


class TestVanillaRunPsi4:
    @pytest.mark.parametrize(
        "geometry,options,n_active_extract,n_occupied_extract,jw_wagner_particle_num,target_tuple",
        [
            (
                hydrogen_geometry,
                {"scf_type": "pk"},
                1,
                None,
                2,
                (-0.8543376267387818, 1, 1, 2, 0, 1, 2, None),
            ),
            (
                hydrogen_geometry,
                {"scf_type": "pk"},
                1,
                0,
                0,
                (-0.8543376267387818, 1, 1, 2, 1, 0, 2, None),
            ),
            (
                hydrogen_geometry,
                None,
                None,
                None,
                2,
                (-0.8544322638069642, 1, 1, 2, 0, 0, 4, -0.9714266904819895),
            ),
            (
                dilithium_geometry,
                None,
                4,
                2,
                4,
                (None, None, None, None, None, None, 8, -14.654620243980217),
            ),
        ],
    )
    def test_run_psi4(
        self,
        geometry,
        options,
        n_active_extract,
        n_occupied_extract,
        jw_wagner_particle_num,
        target_tuple,
    ):
        results_dict = run_psi4(
            geometry=geometry,
            save_hamiltonian=True,
            options=options,
            n_active_extract=n_active_extract,
            n_occupied_extract=n_occupied_extract,
        )

        results, hamiltonian = results_dict["results"], results_dict["hamiltonian"]

        (
            exp_energy,
            exp_alpha,
            exp_beta,
            exp_mo,
            exp_frozen_core,
            exp_frozen_valence,
            exp_n_qubits,
            extra_energy_check,
        ) = target_tuple

        assert math.isclose(results["energy"], exp_energy) if exp_energy else True
        assert results["n_alpha"] == exp_alpha if exp_alpha else True
        assert results["n_beta"] == exp_beta if exp_beta else True
        assert results["n_mo"] == exp_mo if exp_mo else True
        assert results["n_frozen_core"] == exp_frozen_core if exp_mo else True
        assert (
            results["n_frozen_valence"] == exp_frozen_valence
            if exp_frozen_valence
            else True
        )
        assert hamiltonian.n_qubits == exp_n_qubits if exp_n_qubits else True

        qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
        energy, _ = jw_get_ground_state_at_particle_number(
            qubit_operator, jw_wagner_particle_num
        )

        if extra_energy_check:
            assert math.isclose(energy, extra_energy_check, rel_tol=1e-7)
        else:
            assert math.isclose(energy, results["energy"], rel_tol=1e-7)


class TestRunPsi4WithRDMS:
    @pytest.mark.parametrize(
        "geometry,method,freeze_core,n_active_extract,n_occupied_extract,jw_wagner_particle_num,target_tuple",
        [
            (hydrogen_geometry, "fci", False, None, None, 2, (1, 1, 2, 0, 0, 4, None)),
            (
                dilithium_geometry,
                "fci",
                False,
                4,
                1,
                2,
                (None, None, None, None, None, 8, 8),
            ),
            (
                dilithium_geometry,
                "fci",
                True,
                None,
                None,
                None,
                (None, None, None, None, None, 16, 16),
            ),
            (
                dilithium_geometry,
                "scf",
                True,
                None,
                None,
                None,
                (None, None, None, None, None, 16, None),
            ),
        ],
    )
    def test_get_rdms_from_psi4(
        self,
        geometry,
        method,
        freeze_core,
        n_active_extract,
        n_occupied_extract,
        jw_wagner_particle_num,
        target_tuple,
    ):
        psi4.core.clean()

        results_dict = run_psi4(
            geometry=geometry,
            n_active_extract=n_active_extract,
            n_occupied_extract=n_occupied_extract,
            method=method,
            freeze_core=freeze_core,
            freeze_core_extract=freeze_core,
            save_hamiltonian=True,
            save_rdms=True,
        )

        results, hamiltonian, rdm = (
            results_dict["results"],
            results_dict["hamiltonian"],
            results_dict["rdms"],
        )

        energy_from_rdm = rdm.expectation(hamiltonian)

        (
            exp_alpha,
            exp_beta,
            exp_mo,
            exp_frozen_core,
            exp_frozen_valence,
            exp_n_qubits,
            exp_one_body_tensor_shape,
        ) = target_tuple

        assert math.isclose(results["energy"], energy_from_rdm)
        assert results["n_alpha"] == exp_alpha if exp_alpha else True
        assert results["n_beta"] == exp_beta if exp_beta else True
        assert results["n_mo"] == exp_mo if exp_mo else True
        assert results["n_frozen_core"] == exp_frozen_core if exp_mo else True
        assert (
            results["n_frozen_valence"] == exp_frozen_valence
            if exp_frozen_valence
            else True
        )
        assert hamiltonian.n_qubits == exp_n_qubits if exp_n_qubits else True
        assert (
            rdm.one_body_tensor.shape[0] == exp_one_body_tensor_shape
            if exp_one_body_tensor_shape
            else True
        )
        assert math.isclose(einsum("ii->", rdm.one_body_tensor), 2)

        if jw_wagner_particle_num:
            qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
            energy, _ = jw_get_ground_state_at_particle_number(
                qubit_operator, jw_wagner_particle_num
            )

            assert math.isclose(energy, energy_from_rdm)


@pytest.mark.parametrize(
    "geometry,basis,n_active_extract,n_occupied_extract",
    [
        # n_occupied_extract_inconsistent_with_n_active_extract
        (dilithium_geometry, "STO-3G", 2, 3),
        # n_occupied_extract_inconsistent_with_num_electrons
        (hydrogen_geometry, "6-31G", 4, 2),
    ],
)
def test_run_psi4_fails_with_inconsistent_input_combination(
    geometry, basis, n_active_extract, n_occupied_extract
):
    with pytest.raises(ValueError):
        run_psi4(
            geometry=geometry,
            basis=basis,
            n_active_extract=n_active_extract,
            n_occupied_extract=n_occupied_extract,
            save_hamiltonian=True,
        )


def test_run_psi4_freeze_core_extract():
    # For some reason, we must clean Psi4 before running this test if other
    # tests have already run.
    psi4.core.clean()
    results_dict = run_psi4(
        dilithium_geometry,
        method="scf",
        basis="STO-3G",
        freeze_core=True,
        freeze_core_extract=True,
        save_hamiltonian=True,
    )
    hamiltonian = results_dict["hamiltonian"]

    # With STO-3G, each Li atom has one 1s orbital, one 2s orbital, and three 2p orbitals. The 1s orbitals are considered core orbitals.
    assert hamiltonian.n_qubits == 2 * 2 * (1 + 3)


################################################################################################


# # This test may not play well with Psi4 because of 0 electrons in FCI, so I commented it
# # out for now.
# #def test_run_psi4_using_n_occupied_extract_with_rdms_1():
# #    results, hamiltonian, rdm = run_psi4(
# #        dilithium_geometry,
# #        method="fci",
# #        basis="STO-3G",
# #        freeze_core=True,
# #        n_active_extract=4,
# #        n_occupied_extract=0,
# #        save_hamiltonian=True,
# #        save_rdms=True,
# #    )
# #
# #    assert hamiltonian.n_qubits == 8
# #    assert rdm.one_body_tensor.shape[0] == 8
# #    assert np.einsum("ii->", rdm.one_body_tensor) == 0
# #
# #    qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
# #    energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 0)
# #
# #    energy_from_rdm = rdm.expectation(hamiltonian)
# #    assert energy <= energy_from_rdm
