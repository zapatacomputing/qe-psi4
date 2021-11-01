import pytest
from openfermion import FermionOperator
from qepsi4 import parse_amplitudes_from_psi4_ccsd, run_psi4


def get_expected_mp2_fop():
    result_fop = FermionOperator()
    result_fop += -0.0193104873 * FermionOperator("4^ 0 5^ 1")
    result_fop += -0.0193104873 * FermionOperator("5^ 1 4^ 0")
    result_fop += -0.0118067353 * FermionOperator("6^ 0 7^ 1")
    result_fop += -0.0118067353 * FermionOperator("7^ 1 6^ 0")
    result_fop += -0.00675205785 * FermionOperator("2^ 0 3^ 1")
    result_fop += -0.00675205785 * FermionOperator("3^ 1 2^ 0")
    result_fop += -0.0056262204 * FermionOperator("2^ 0 7^ 1")
    result_fop += -0.0056262204 * FermionOperator("3^ 1 6^ 0")
    result_fop += -0.0056262204 * FermionOperator("6^ 0 3^ 1")
    result_fop += -0.0056262204 * FermionOperator("7^ 1 2^ 0")

    return result_fop


def get_expected_h2_ccsd_fop():
    result_fop = FermionOperator()
    result_fop += -0.0252893416 * FermionOperator("4^ 0 5^ 1")
    result_fop += -0.0252893416 * FermionOperator("5^ 1 4^ 0")
    result_fop += -0.0142638609 * FermionOperator("6^ 0 7^ 1")
    result_fop += -0.0142638609 * FermionOperator("7^ 1 6^ 0")
    result_fop += -0.00821798305 * FermionOperator("2^ 0 3^ 1")
    result_fop += -0.00821798305 * FermionOperator("3^ 1 2^ 0")
    result_fop += -0.00783761365 * FermionOperator("2^ 0 7^ 1")
    result_fop += -0.00783761365 * FermionOperator("3^ 1 6^ 0")
    result_fop += -0.00783761365 * FermionOperator("6^ 0 3^ 1")
    result_fop += -0.00783761365 * FermionOperator("7^ 1 2^ 0")
    result_fop += -0.0047660845 * FermionOperator("4^ 0")
    result_fop += -0.0047660845 * FermionOperator("5^ 1")

    return result_fop


def get_expected_h2_ccsd_fop_n_active_extract():
    ref_fop = FermionOperator()
    ref_fop += -0.0252893416 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.0252893416 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.00821798305 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.00821798305 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.0047660845 * FermionOperator("4^ 0")
    ref_fop += -0.0047660845 * FermionOperator("5^ 1")

    return ref_fop


def get_expected_LiH_ccsd_fop_frozen_core():
    result_fop = FermionOperator()
    result_fop += -0.1071079862 * FermionOperator("2^ 0 3^ 1")
    result_fop += -0.1071079862 * FermionOperator("3^ 1 2^ 0")
    result_fop += -0.10207494545 * FermionOperator("4^ 0 5^ 1")
    result_fop += -0.10207494545 * FermionOperator("5^ 1 4^ 0")
    result_fop += -0.10207494545 * FermionOperator("6^ 0 7^ 1")
    result_fop += -0.10207494545 * FermionOperator("7^ 1 6^ 0")
    result_fop += -0.01003110065 * FermionOperator("8^ 0 9^ 1")
    result_fop += -0.01003110065 * FermionOperator("9^ 1 8^ 0")
    result_fop += -0.0037155031 * FermionOperator("2^ 0 9^ 1")
    result_fop += -0.0037155031 * FermionOperator("9^ 1 2^ 0")
    result_fop += -0.0037155031 * FermionOperator("8^ 0 3^ 1")
    result_fop += -0.0037155031 * FermionOperator("3^ 1 8^ 0")
    result_fop += -0.0603059476 * FermionOperator("2^ 0")
    result_fop += -0.0603059476 * FermionOperator("3^ 1")
    result_fop += 0.0055808378 * FermionOperator("8^ 0")
    result_fop += 0.0055808378 * FermionOperator("9^ 1")

    return result_fop


def get_expected_LiH_ccsd_fop_freeze_core_extract():
    result_fop = FermionOperator()
    result_fop += -0.1068653324 * FermionOperator("2^ 0 3^ 1")
    result_fop += -0.1068653324 * FermionOperator("3^ 1 2^ 0")
    result_fop += -0.10189758465 * FermionOperator("4^ 0 5^ 1")
    result_fop += -0.10189758465 * FermionOperator("5^ 1 4^ 0")
    result_fop += -0.10189758465 * FermionOperator("6^ 0 7^ 1")
    result_fop += -0.10189758465 * FermionOperator("7^ 1 6^ 0")
    result_fop += -0.01005847195 * FermionOperator("8^ 0 9^ 1")
    result_fop += -0.01005847195 * FermionOperator("9^ 1 8^ 0")
    result_fop += -0.00369098685 * FermionOperator("2^ 0 9^ 1")
    result_fop += -0.00369098685 * FermionOperator("9^ 1 2^ 0")
    result_fop += -0.00369098685 * FermionOperator("8^ 0 3^ 1")
    result_fop += -0.00369098685 * FermionOperator("3^ 1 8^ 0")
    result_fop += -0.0618151355 * FermionOperator("2^ 0")
    result_fop += -0.0618151355 * FermionOperator("3^ 1")
    result_fop += 0.0050395518 * FermionOperator("8^ 0")
    result_fop += 0.0050395518 * FermionOperator("9^ 1")

    return result_fop


def get_expected_LiH_ccsd_fop_active_extract_frozen_core():
    ref_fop = FermionOperator()
    ref_fop += -0.1071079862 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.1071079862 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.10207494545 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.10207494545 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.0603059476 * FermionOperator("2^ 0")
    ref_fop += -0.0603059476 * FermionOperator("3^ 1")

    return ref_fop


def get_expected_ccsd_fop_freeze_core_extract_n_active_extract():
    ref_fop = FermionOperator()
    ref_fop += -0.1068653324 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.1068653324 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.10189758465 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.10189758465 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.0618151355 * FermionOperator("2^ 0")
    ref_fop += -0.0618151355 * FermionOperator("3^ 1")

    return ref_fop


options = {"mp2_amps_print": True, "num_amps_print": 100}


@pytest.mark.parametrize(
    "expected_fop_fn,basis,output_filename,geometry,\
        n_active_extract,freeze_core_extract,freeze_core,n_frozen_amplitudes,\
            get_mp2_amplitudes",
    [  # MP2 amplitude extraction test
        (
            get_expected_mp2_fop,
            "6-31g",
            "h2_sto3g",
            {
                "sites": [
                    {"species": "H", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            None,
            False,
            False,
            0,
            True,
        ),
        # CCSD amplitude extraction test
        (
            get_expected_h2_ccsd_fop,
            "6-31g",
            "h2_sto3g",
            {
                "sites": [
                    {"species": "H", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            None,
            False,
            False,
            0,
            False,
        ),
        # CCSD amplitude extraction test with n_active_extract
        (
            get_expected_h2_ccsd_fop_n_active_extract,
            "6-31g",
            "h2_sto3g",
            {
                "sites": [
                    {"species": "H", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            3,
            False,
            False,
            0,
            False,
        ),
        # CCSD amplitude extraction test with freeze_core
        (
            get_expected_LiH_ccsd_fop_frozen_core,
            "sto-3g",
            "lih_sto3g",
            {
                "sites": [
                    {"species": "Li", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            None,
            False,
            True,
            0,
            False,
        ),
        # CCSD amplitude extraction test with freeze_core and n_active_extract
        (
            get_expected_LiH_ccsd_fop_active_extract_frozen_core,
            "sto-3g",
            "lih_sto3g",
            {
                "sites": [
                    {"species": "Li", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            3,
            False,
            True,
            0,
            False,
        ),
        # CCSD amplitude extraction test with freeze_core_extract
        (
            get_expected_LiH_ccsd_fop_freeze_core_extract,
            "sto-3g",
            "lih_sto3g",
            {
                "sites": [
                    {"species": "Li", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            None,
            True,
            False,
            1,
            False,
        ),
        # CCSD amplitude extraction test with freeze_core_extract and n_active_extract
        (
            get_expected_ccsd_fop_freeze_core_extract_n_active_extract,
            "sto-3g",
            "lih_sto3g",
            {
                "sites": [
                    {"species": "Li", "x": 0, "y": 0, "z": 0},
                    {"species": "H", "x": 0, "y": 0, "z": 0.2},
                ]
            },
            3,
            True,
            False,
            1,
            False,
        ),
    ],
)
def test_parse_amplitudes_from_psi4_ccsd(
    expected_fop_fn,
    basis,
    output_filename,
    geometry,
    n_active_extract,
    freeze_core_extract,
    freeze_core,
    n_frozen_amplitudes,
    get_mp2_amplitudes,
):
    out = run_psi4(
        geometry=geometry,
        basis=basis,
        method="ccsd",
        freeze_core=freeze_core,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=n_active_extract,
        freeze_core_extract=freeze_core_extract,
        freeze_core=freeze_core,
        n_frozen_amplitudes=n_frozen_amplitudes,
        get_mp2_amplitudes=get_mp2_amplitudes,
    )

    expected_fop = expected_fop_fn()

    assert expected_fop == given_fop
