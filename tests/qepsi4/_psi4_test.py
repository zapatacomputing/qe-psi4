# type: ignore
from qepsi4 import select_active_space, run_psi4
from openfermion import (
    jordan_wigner,
    jw_get_ground_state_at_particle_number,
    qubit_operator_sparse,
)
from numpy import NaN, einsum
import math
import psi4
import pytest
from collections import namedtuple
from typing import List

config_list = [
    "geometry",
    "n_active_extract",
    "n_occupied_extract",
    "freeze_core",
    "freeze_core_extract",
    "save_rdms",
    "method",
    "options",
]
config_list_defaults = [None] * 2 + [False] * 3 + ["scf"] + [None]
Psi4Config = namedtuple("Psi4Config", config_list, defaults=config_list_defaults,)

expected_tuple_list = [
    "exp_energy",
    "exp_alpha",
    "exp_beta",
    "exp_mo",
    "exp_frozen_core",
    "exp_frozen_valence",
    "exp_n_qubits",
    "extra_energy_check",
    "exp_one_body_tensor_shape",
]
ExpectedTuple = namedtuple(
    "ExpectedTuple", expected_tuple_list, defaults=[None] * len(expected_tuple_list)
)


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


class TestRunPsi4:
    @pytest.mark.parametrize(
        "psi4_config,jw_wagner_particle_num,expected_tuple",
        [
            (
                Psi4Config(hydrogen_geometry, 1, options={"scf_type": "pk"}),
                2,
                ExpectedTuple(-0.8543376267387818, 1, 1, 2, 0, 1, 2),
            ),
            (
                Psi4Config(hydrogen_geometry, 1, 0, options={"scf_type": "pk"}),
                0,
                ExpectedTuple(-0.8543376267387818, 1, 1, 2, 1, 0, 2),
            ),
            (
                Psi4Config(hydrogen_geometry),
                2,
                ExpectedTuple(
                    -0.8544322638069642, 1, 1, 2, 0, 0, 4, -0.9714266904819895
                ),
            ),
            (
                Psi4Config(dilithium_geometry, None, 4, 2,),
                4,
                ExpectedTuple(exp_n_qubits=8, extra_energy_check=-14.654620243980217),
            ),
            (
                Psi4Config(hydrogen_geometry, method="fci", save_rdms=True),
                2,
                ExpectedTuple(None, 1, 1, 2, 0, 0, 4),
            ),
            (
                Psi4Config(dilithium_geometry, 4, 1, method="fci", save_rdms=True),
                2,
                ExpectedTuple(exp_n_qubits=8, exp_one_body_tensor_shape=8),
            ),
            (
                Psi4Config(
                    dilithium_geometry,
                    freeze_core=True,
                    freeze_core_extract=True,
                    method="fci",
                    save_rdms=True,
                ),
                None,
                ExpectedTuple(exp_n_qubits=16, exp_one_body_tensor_shape=16),
            ),
        ],
    )
    def test_run_psi4(
        self,
        psi4_config: Psi4Config,
        jw_wagner_particle_num: int,
        expected_tuple: ExpectedTuple,
    ):
        results_dict = run_psi4(
            geometry=psi4_config.geometry,
            n_active_extract=psi4_config.n_active_extract,
            n_occupied_extract=psi4_config.n_occupied_extract,
            freeze_core=psi4_config.freeze_core,
            freeze_core_extract=psi4_config.freeze_core_extract,
            method=psi4_config.method,
            options=psi4_config.options,
            save_hamiltonian=True,
            save_rdms=psi4_config.save_rdms,
        )

        results, hamiltonian, rdm = (
            results_dict["results"],
            results_dict["hamiltonian"],
            results_dict["rdms"],
        )

        if rdm:
            energy_from_rdm = rdm.expectation(hamiltonian)

        if psi4_config.save_rdms:
            math.isclose(results["energy"], energy_from_rdm)
        else:
            assert (
                math.isclose(results["energy"], expected_tuple.exp_energy)
                if expected_tuple.exp_energy
                else True
            )
        assert (
            results["n_alpha"] == expected_tuple.exp_alpha
            if expected_tuple.exp_alpha
            else True
        )
        assert (
            results["n_beta"] == expected_tuple.exp_beta
            if expected_tuple.exp_beta
            else True
        )
        assert (
            results["n_mo"] == expected_tuple.exp_mo if expected_tuple.exp_mo else True
        )
        assert (
            results["n_frozen_core"] == expected_tuple.exp_frozen_core
            if expected_tuple.exp_frozen_core
            else True
        )
        assert (
            results["n_frozen_valence"] == expected_tuple.exp_frozen_valence
            if expected_tuple.exp_frozen_valence
            else True
        )
        assert (
            hamiltonian.n_qubits == expected_tuple.exp_n_qubits
            if expected_tuple.exp_n_qubits
            else True
        )

        if expected_tuple.exp_one_body_tensor_shape:
            assert (
                rdm.one_body_tensor.shape[0] == expected_tuple.exp_one_body_tensor_shape
            )

            assert math.isclose(einsum("ii->", rdm.one_body_tensor), 2)

        if jw_wagner_particle_num:
            qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
            energy, _ = jw_get_ground_state_at_particle_number(
                qubit_operator, jw_wagner_particle_num
            )

            if expected_tuple.extra_energy_check:
                assert math.isclose(
                    energy, expected_tuple.extra_energy_check, rel_tol=1e-7
                )
            elif psi4_config.save_rdms:
                assert math.isclose(energy, energy_from_rdm)
            else:
                assert math.isclose(energy, results["energy"], rel_tol=1e-7)


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
