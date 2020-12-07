import psi4
from psi4 import qcel as qc  # QCelemental (https://github.com/MolSSI/QCElemental)
import numpy as np
from openfermion import (
    InteractionOperator,
    general_basis_change,
    MolecularData,
    InteractionRDM,
)
from openfermion.config import EQ_TOLERANCE
from typing import Tuple, Dict
import warnings


def select_active_space(
    mol, wfn, n_active_extract=None, n_occupied_extract=None, freeze_core_extract=False
):
    """ Helper function to calculate the number of doubly occupied, active and virtual orbitals when extracting Hamiltonian/RDM 
    Args:
        mol (psi4.core.Molecule): Psi4 object containing information about the molecule
            including geometry, charge, symmetry and spin multiplicity
        wfn (psi4.core.Wavefunction): Psi4 Wavefunction object; contains basis set info
        n_active_extract (int): number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals.
        n_occupied_extract (int): number of occupied molecular orbitals to
            include in the saved Hamiltonian. Must be less than or equal to
            n_active_extract. If None, all occupied orbitals are included,
            except the core orbitals if freeze_core_extract is set to True.
        freeze_core_extract (bool): If True, frozen core orbitals will always be
            doubly occupied in the saved Hamiltonian. Ignored if
            n_occupied_extract is not None.
    """
    # Check the input parameters (should we add more checks?)
    if n_active_extract is not None:
        if wfn.nmo() < n_active_extract:
            raise ValueError(
                f"Number of active orbitals exceeds the number of basis functions."
            )
        if n_occupied_extract is not None:
            if n_occupied_extract > n_active_extract:
                raise ValueError(
                    f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than total number of molecular orbitals to extract ({n_active_extract})."
                )
    if n_occupied_extract is not None:
        if wfn.nalpha() != wfn.nbeta():
            raise ValueError(
                f"Requesting a number of occupied molecular orbitals not supported when number of alpha and beta electrons is unequal."
            )
        if n_occupied_extract > wfn.nalpha():
            raise ValueError(
                f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than number of occupied molecular orbitals ({wfn.nalpha()})."
            )

    # Determine the number of core orbitals based on the molecular identity
    # Number of cores for a given row of periodic table
    chemical_core_orbitals = {
        1: 0,
        2: qc.periodictable.to_Z("He") // 2,
        3: qc.periodictable.to_Z("Ne") // 2,
        4: qc.periodictable.to_Z("Ar") // 2,
    }

    nfrzc = 0  # number of orbitals to freeze if freeze_core_extract = True
    for i in range(mol.natom()):
        atom = mol.label(i)
        ncore = chemical_core_orbitals.get(qc.periodictable.to_period(atom), None)
        if ncore is None:
            raise ValueError(
                f"Core definitions were not implemented for the elements beyond the 4th row of periodic table but the molecule contains {atom}."
            )
        nfrzc += ncore

    n_core_extract = 0
    if freeze_core_extract and n_occupied_extract is None:
        n_core_extract = nfrzc
    elif (
        n_occupied_extract is not None
    ):  # n_occupied_extract overrides freeze_core_extract=True
        n_core_extract = wfn.nalpha() - n_occupied_extract

    if n_active_extract is not None:
        if n_core_extract + n_active_extract > wfn.nmo():
            raise ValueError(
                f"Active space dimension ({n_active_extract}) is inconsistent with the number of doubly occupied oribtals ({n_core_extract}) and basis set size ({wfn.nmo()})."
            )
        else:
            n_frozen_virtuals = wfn.nmo() - n_core_extract - n_active_extract
    else:
        n_active_extract = wfn.nmo() - n_core_extract
        n_frozen_virtuals = 0
        assert n_active_extract > 0

    return n_core_extract, n_active_extract, n_frozen_virtuals


def run_psi4(
    geometry: dict,
    basis: str = "STO-3G",
    multiplicity: int = 1,
    charge: int = 0,
    method: str = "scf",
    reference: str = "rhf",
    freeze_core: bool = False,
    save_hamiltonian: bool = False,
    options: dict = None,
    n_active_extract: int = None,
    n_occupied_extract: int = None,
    freeze_core_extract: bool = False,
    save_rdms: bool = False,
):
    """Generate an input file in the Psi4 python domain-specific language for
    a molecule.
    
    Args:
        geometry: a dictionary containing the molecule geometry.
        basis: which basis set to use
        multiplicity: spin multiplicity
        charge: charge of the molecule
        method: which calculation method to use
        reference: which reference wavefunction to use. fno- energy methods are only compatible with RHF
        freeze_core: Whether to freeze occupied core orbitals
        save_hamiltonian: whether to save the Hamiltonian to a file. If True, symmetry will be disabled.
        options: additional commands to be passed to Psi4
        n_active_extract: number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals.
        n_occupied_extract: number of occupied molecular orbitals to
            include in the saved Hamiltonian. Must be less than or equal to
            n_active_extract. If None, all occupied orbitals are included,
            except the core orbitals if freeze_core_extract is set to True.
        freeze_core_extract: If True, frozen core orbitals will always be
            doubly occupied in the saved Hamiltonian. Ignored if
            n_occupied_extract is not None.
        save_rdms: If True, save 1- and 2-RDMs
        
    Returns:
        results_dict: Python dictionary with the results of the calculation (dict), Hamiltonian
            (openfermion.ops.InteractionOperator), and 1- and 2-RDMs (openfermion.op.InteractionRDM)
    """

    if save_rdms and not method in ["fci", "cis", "cisd", "cisdt", "cisdtq"]:
        print(f"run_psi4 was called with method={method}")
        warnings.warn(
            "RDM calculation can only be performed for Configuration Interaction methods. save_rdms option will be set to False."
        )
        save_rdms = False

    geometry_str = f"{charge} {multiplicity}\n"
    for atom in geometry["sites"]:
        geometry_str += "{} {} {} {}\n".format(
            atom["species"], atom["x"], atom["y"], atom["z"]
        )

    geometry_str += "\nunits angstrom\n"
    c1_sym = save_hamiltonian or save_rdms
    if c1_sym:
        geometry_str += "symmetry c1\n"

    molecule = psi4.geometry(geometry_str)

    combined_options = {
        "reference": reference,
        "basis": basis,
        "freeze_core": freeze_core,
    }
    if options:
        combined_options.update(options)
    psi4.set_options(combined_options)

    # Create a fake wave function and use it to select the active space based on the input parameters
    fake_wfn = psi4.core.Wavefunction.build(
        molecule, psi4.core.get_global_option("basis")
    )
    ndocc, nact, n_frozen_vir = select_active_space(
        molecule,
        fake_wfn,
        n_active_extract=n_active_extract,
        n_occupied_extract=n_occupied_extract,
        freeze_core_extract=freeze_core_extract,
    )

    if ndocc != 0:
        psi4.set_options({"num_frozen_docc": ndocc})
    if n_frozen_vir != 0:
        assert (
            molecule.point_group().symbol() == "c1"
        )  # otherwise frozen_uocc should be an array specifying number of orbitals per irrep
        psi4.set_options({"frozen_uocc": [n_frozen_vir]})

    if (
        method == "fci"
        or method == "cis"
        or method == "cisd"
        or method == "cisdt"
        or method == "cisdtq"
    ):
        psi4.set_options({"qc_module": "detci"})
        if save_rdms:
            psi4.set_options({"opdm": True, "tpdm": True})

    energy, wavefunction = psi4.energy(method, return_wfn=True)

    # Perform a sanity check to make sure that the active space was selected correctly
    assert wavefunction.nmo() == (ndocc + nact + n_frozen_vir)
    # if method != 'scf':
    #    assert wavefunction.fzvpi().sum() == nvir

    results_dict = {}

    results = {
        "energy": energy,
        "n_alpha": wavefunction.nalpha(),
        "n_beta": wavefunction.nbeta(),
        "n_mo": wavefunction.nmo(),
        "n_frozen_core": wavefunction.nfrzc(),
        "n_frozen_valence": wavefunction.frzvpi().sum(),
    }

    hamiltonian = None
    if save_hamiltonian:
        mints = psi4.core.MintsHelper(wavefunction.basisset())
        hamiltonian = get_ham_from_psi4(
            wavefunction,
            mints,
            ndocc=ndocc,
            nact=nact,
            nuclear_repulsion_energy=molecule.nuclear_repulsion_energy(),
        )

    rdm = None
    if save_rdms:
        rdm = get_rdms_from_psi4(wavefunction, ndocc=ndocc, nact=nact)

    results_dict["results"] = results
    results_dict["hamiltonian"] = hamiltonian
    results_dict["rdms"] = rdm
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.core.clean_variables()
    return results_dict


def get_ham_from_psi4(
    wfn, mints, ndocc=None, nact=None, nuclear_repulsion_energy=0,
):
    """Get a molecular Hamiltonian from a Psi4 calculation.

    Args:
        wfn (psi4.core.Wavefunction): Psi4 wavefunction object
        mints (psi4.core.MintsHelper): Psi4 molecular integrals helper
        ndocc (int): number of doubly occupied molecular orbitals to
            include in the saved Hamiltonian. 
        nact (int): number of active molecular orbitals to include in the
            saved Hamiltonian. 
        nuclear_repulsion_energy (float): The ion-ion interaction energy.
    
    Returns:
        hamiltonian (openfermion.ops.InteractionOperator): the electronic
            Hamiltonian.
    """

    assert wfn.same_a_b_orbs(), (
        "Extraction of Hamiltonian from wavefunction"
        + "with different alpha and beta orbitals not yet supported :("
    )

    orbitals = wfn.Ca().to_array(dense=True)

    if nact is None and ndocc is None:
        trf_mat = orbitals
        ndocc = 0
        nact = orbitals.shape[1]
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )
    elif nact is not None and ndocc is None:
        assert nact <= orbitals.shape[1]
        ndocc = 0
        trf_mat = orbitals[:, :nact]
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )
    elif ndocc is not None and nact is None:
        assert ndocc <= orbitals.shape[1]
        nact = orbitals.shape[1] - ndocc
        trf_mat = orbitals
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )
    else:
        assert orbitals.shape[1] >= nact + ndocc
        trf_mat = orbitals[:, : nact + ndocc]

    # Note: code refactored to use Psi4 integral-transformation routines
    # no more storing the whole two-electron integral tensor when only an
    # active space is needed

    one_body_integrals = general_basis_change(
        np.asarray(mints.ao_kinetic()), orbitals, (1, 0)
    )
    one_body_integrals += general_basis_change(
        np.asarray(mints.ao_potential()), orbitals, (1, 0)
    )

    # Build the transformation matrices, i.e. the orbitals for which
    # we want the integrals, as Psi4.core.Matrix objects

    trf_mat = psi4.core.Matrix.from_array(trf_mat)

    two_body_integrals = np.asarray(mints.mo_eri(trf_mat, trf_mat, trf_mat, trf_mat))
    n_orbitals = trf_mat.shape[1]
    two_body_integrals.reshape((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    two_body_integrals = np.einsum("psqr", two_body_integrals)

    # Truncate
    one_body_integrals[np.absolute(one_body_integrals) < EQ_TOLERANCE] = 0.0
    two_body_integrals[np.absolute(two_body_integrals) < EQ_TOLERANCE] = 0.0

    occupied_indices = range(ndocc)
    active_indices = range(ndocc, ndocc + nact)

    # In order to keep the MolecularData class happy, we need a 'valid' molecule
    molecular_data = MolecularData(
        geometry=[("H", (0, 0, 0))], basis="", multiplicity=2
    )

    molecular_data.one_body_integrals = one_body_integrals
    molecular_data.two_body_integrals = two_body_integrals
    molecular_data.nuclear_repulsion = nuclear_repulsion_energy
    hamiltonian = molecular_data.get_molecular_hamiltonian(
        occupied_indices, active_indices
    )

    return hamiltonian


def get_rdms_from_psi4(wfn, ndocc=None, nact=None):

    """Get 1- and 2-RDMs from a Psi4 calculation.

    Args:
        wfn (psi4.core.Wavefunction): Psi4 wavefunction object
        ndocc (int): number of doubly occupied molecular orbitals to
            exclude from the saved RDM. 
        nact (int): number of active molecular orbitals to include in the
            saved RDM. 
    Returns:
        rdm (openfermion.ops.InteractionRDM): an openfermion object storing
            1- and 2-RDMs. 
    """

    if nact is None and ndocc is None:
        ndocc = 0
        nact = wfn.Ca().to_array(dense=True).shape[1]
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )
    elif nact is not None and ndocc is None:
        assert nact <= wfn.Ca().to_array(dense=True).shape[1]
        ndocc = 0
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )
    elif ndocc is not None and nact is None:
        assert ndocc <= wfn.Ca().to_array(dense=True).shape[1]
        nact = wfn.Ca().to_array(dense=True).shape[1] - ndocc
        print(
            f"Active space selection options were reset to: ndocc = {ndocc} and  nact = {nact}"
        )

    one_rdm_a = np.array(wfn.get_opdm(0, 0, "A", True)).reshape(wfn.nmo(), wfn.nmo())
    one_rdm_b = np.array(wfn.get_opdm(0, 0, "B", True)).reshape(wfn.nmo(), wfn.nmo())

    # Get 2-RDM from CI calculation.

    assert nact == wfn.nmo() - wfn.nfrzc() - wfn.frzvpi().sum()
    assert ndocc == wfn.nfrzc()

    two_rdm_aa = np.array(wfn.get_tpdm("AA", False)).reshape(nact, nact, nact, nact)
    two_rdm_ab = np.array(wfn.get_tpdm("AB", False)).reshape(nact, nact, nact, nact)
    two_rdm_bb = np.array(wfn.get_tpdm("BB", False)).reshape(nact, nact, nact, nact)

    # adjusting 1-RDM
    # Indices of active molecular orbitals
    active_indices = range(ndocc, ndocc + nact)
    one_rdm_a = one_rdm_a[np.ix_(active_indices, active_indices)]
    one_rdm_b = one_rdm_b[np.ix_(active_indices, active_indices)]

    one_rdm, two_rdm = unpack_spatial_rdm(
        one_rdm_a, one_rdm_b, two_rdm_aa, two_rdm_ab, two_rdm_bb
    )

    rdms = InteractionRDM(one_rdm, two_rdm)

    return rdms


def unpack_spatial_rdm(one_rdm_a, one_rdm_b, two_rdm_aa, two_rdm_ab, two_rdm_bb):
    r"""
    Convert from spin compact spatial format to spin-orbital format for RDM.
    Note: the compact 2-RDM is stored as follows where A/B are spin up/down:
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, A}^\dagger a_{q, A} a_{s, A} |>
      for 'AA'/'BB' spins.
    RDM[pqrs] = <| a_{p, A}^\dagger a_{r, B}^\dagger a_{q, B} a_{s, A} |>
      for 'AB' spins.
    Args:
        one_rdm_a: 2-index numpy array storing alpha spin
            sector of 1-electron reduced density matrix.
        one_rdm_b: 2-index numpy array storing beta spin
            sector of 1-electron reduced density matrix.
        two_rdm_aa: 4-index numpy array storing alpha-alpha spin
            sector of 2-electron reduced density matrix.
        two_rdm_ab: 4-index numpy array storing alpha-beta spin
            sector of 2-electron reduced density matrix.
        two_rdm_bb: 4-index numpy array storing beta-beta spin
            sector of 2-electron reduced density matrix.
    Returns:
        one_rdm: 2-index numpy array storing 1-electron density matrix
            in full spin-orbital space.
        two_rdm: 4-index numpy array storing 2-electron density matrix
            in full spin-orbital space.
    """
    # Initialize RDMs.
    n_orbitals = one_rdm_a.shape[0]
    n_qubits = 2 * n_orbitals
    one_rdm = np.zeros((n_qubits, n_qubits))
    two_rdm = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits))

    # Unpack compact representation.
    for p in range(n_orbitals):
        for q in range(n_orbitals):

            # Populate 1-RDM.
            one_rdm[2 * p, 2 * q] = one_rdm_a[p, q]
            one_rdm[2 * p + 1, 2 * q + 1] = one_rdm_b[p, q]

            # Continue looping to unpack 2-RDM.
            for r in range(n_orbitals):
                for s in range(n_orbitals):

                    # Handle case of same spin.
                    two_rdm[2 * p, 2 * q, 2 * r, 2 * s] = two_rdm_aa[p, r, q, s]
                    two_rdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = two_rdm_bb[
                        p, r, q, s
                    ]

                    # Handle case of mixed spin.
                    two_rdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = two_rdm_ab[p, r, q, s]
                    two_rdm[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        -1.0 * two_rdm_ab[p, s, q, r]
                    )
                    two_rdm[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = two_rdm_ab[q, s, p, r]
                    two_rdm[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        -1.0 * two_rdm_ab[q, r, p, s]
                    )

    # Map to physicist notation and return.
    two_rdm = np.einsum("pqsr", two_rdm)
    return one_rdm, two_rdm
