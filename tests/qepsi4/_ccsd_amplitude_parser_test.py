from qepsi4 import run_psi4, parse_amplitudes_from_psi4_ccsd
from openfermion import FermionOperator


def get_expected_fop():
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


### MP2 amplitude extraction test ###


def test_h2_mp2_amplitudes_parsed_correctly():
    basis = "6-31g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "h2_sto3g"

    hydrogen_geometry = {
        "sites": [
            {"species": "H", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=hydrogen_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=False,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=None,
        freeze_core_extract=False,
        freeze_core=False,
        n_frozen_amplitudes=0,
        get_mp2_amplitudes=True,
    )

    expected_fop = get_expected_fop()

    assert expected_fop == given_fop


def get_expected_ccsd_fop():
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


### CCSD amplitude extraction test ###


def test_h2_ccsd_amplitudes_parsed_correctly():
    basis = "6-31g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "h2_sto3g"

    hydrogen_geometry = {
        "sites": [
            {"species": "H", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=hydrogen_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=False,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=None,
        freeze_core_extract=False,
        freeze_core=False,
        n_frozen_amplitudes=0,
        get_mp2_amplitudes=False,
    )

    expected_fop = get_expected_ccsd_fop()

    assert expected_fop == given_fop


### CCSD amplitude extraction test with n_active_extract ###


def test_h2_ccsd_amplitudes_parsed_correctly_with_active_extract():

    ref_fop = FermionOperator()
    ref_fop += -0.0252893416 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.0252893416 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.00821798305 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.00821798305 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.0047660845 * FermionOperator("4^ 0")
    ref_fop += -0.0047660845 * FermionOperator("5^ 1")

    basis = "6-31g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "h2_sto3g"

    hydrogen_geometry = {
        "sites": [
            {"species": "H", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=hydrogen_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=False,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=3,
        freeze_core_extract=False,
        freeze_core=False,
        n_frozen_amplitudes=0,
        get_mp2_amplitudes=False,
    )

    expected_fop = ref_fop
    print("Given")
    print(given_fop)
    print("Expected")
    print(expected_fop)

    assert expected_fop == given_fop


### CCSD amplitude extraction test with freeze_core ###
def get_expected_ccsd_fop_frozen_core_LiH():
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


def test_frozen_core_lih_ccsd_amplitudes_parsed_correctly():
    basis = "sto-3g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "lih_sto3g"

    LiH_geometry = {
        "sites": [
            {"species": "Li", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=LiH_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=True,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=None,
        freeze_core_extract=False,
        freeze_core=True,
        n_frozen_amplitudes=0,
        get_mp2_amplitudes=False,
    )

    expected_fop = get_expected_ccsd_fop_frozen_core_LiH()

    assert expected_fop == given_fop


### CCSD amplitude extraction test with freeze_core and n_active_extract ###


def test_frozen_core_lih_ccsd_amplitudes_parsed_correctly_with_n_active():

    ref_fop = FermionOperator()
    ref_fop += -0.1071079862 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.1071079862 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.10207494545 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.10207494545 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.0603059476 * FermionOperator("2^ 0")
    ref_fop += -0.0603059476 * FermionOperator("3^ 1")

    basis = "sto-3g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "lih_sto3g"

    LiH_geometry = {
        "sites": [
            {"species": "Li", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=LiH_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=True,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=3,
        freeze_core_extract=False,
        freeze_core=True,
        n_frozen_amplitudes=0,
        get_mp2_amplitudes=False,
    )

    expected_fop = ref_fop

    assert expected_fop == given_fop


### CCSD amplitude extraction test with freeze_core_extract ###


def get_expected_ccsd_fop_freeze_core_extract_LiH():
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


def test_freeze_core_extract_lih_ccsd_amplitudes_parsed_correctly():
    basis = "sto-3g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "lih_sto3g"

    LiH_geometry = {
        "sites": [
            {"species": "Li", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=LiH_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=False,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=None,
        freeze_core_extract=True,
        freeze_core=False,
        n_frozen_amplitudes=1,
        get_mp2_amplitudes=False,
    )

    expected_fop = get_expected_ccsd_fop_freeze_core_extract_LiH()

    assert expected_fop == given_fop


### CCSD amplitude extraction test with freeze_core_extract and n_active_extract ###


def test_freeze_core_extract_lih_ccsd_amplitudes_parsed_correctly_with_n_active():

    ref_fop = FermionOperator()
    ref_fop += -0.1068653324 * FermionOperator("2^ 0 3^ 1")
    ref_fop += -0.1068653324 * FermionOperator("3^ 1 2^ 0")
    ref_fop += -0.10189758465 * FermionOperator("4^ 0 5^ 1")
    ref_fop += -0.10189758465 * FermionOperator("5^ 1 4^ 0")
    ref_fop += -0.0618151355 * FermionOperator("2^ 0")
    ref_fop += -0.0618151355 * FermionOperator("3^ 1")

    basis = "sto-3g"

    options = {"mp2_amps_print": True, "num_amps_print": 100}

    output_filename = "lih_sto3g"

    LiH_geometry = {
        "sites": [
            {"species": "Li", "x": 0, "y": 0, "z": 0},
            {"species": "H", "x": 0, "y": 0, "z": 0.2},
        ]
    }

    out = run_psi4(
        geometry=LiH_geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        method="ccsd",
        reference="rhf",
        freeze_core=False,
        save_hamiltonian=True,
        options=options,
        n_active_extract=None,
        n_occupied_extract=None,
        freeze_core_extract=False,
        save_rdms=False,
        output_filename=output_filename,
        return_wfn=True,
    )

    _, wfn = out

    given_fop = parse_amplitudes_from_psi4_ccsd(
        wfn=wfn,
        psi_filename="{0}.dat".format(output_filename),
        n_active_extract=3,
        freeze_core_extract=True,
        freeze_core=False,
        n_frozen_amplitudes=1,
        get_mp2_amplitudes=False,
    )

    expected_fop = ref_fop

    assert expected_fop == given_fop