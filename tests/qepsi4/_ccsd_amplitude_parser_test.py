from qepsi4 import run_psi4, parse_amplitudes_from_psi4_ccsd
from openfermion import FermionOperator


def get_expected_fop():
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

    return result_fop


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
        get_mp2_amplitudes=True,
    )

    expected_fop = get_expected_fop()

    assert expected_fop == given_fop
