import psi4
import numpy as np
from openfermion import InteractionOperator, general_basis_change, MolecularData, InteractionRDM
from openfermion.config import EQ_TOLERANCE


def run_psi4(
    geometry,
    basis="STO-3G",
    multiplicity=1,
    charge=0,
    method="scf",
    reference="rhf",
    freeze_core=False,
    n_active=None,
    save_hamiltonian=False,
    options=None,
    n_active_extract=None,
    n_occupied_extract=None,
    freeze_core_extract=False,
    save_rdms=False,
):
    """Generate an input file in the Psi4 python domain-specific language for
    a molecule.
    
    Args:
        geometry (dict): a dictionary containing the molecule geometry.
        basis (str): which basis set to use
        multiplicity (int): spin multiplicity
        charge (int): charge of the molecule
        method (str): which calculation method to use
        reference (str): which reference wavefunction to use. fno- energy methods are only compatible with RHF
        freeze_core (bool): Whether to freeze occupied core orbitals
        n_active (int): Number of active orbitals. Freeze virtual orbitals that do not fit on the qubits.
        save_hamiltonian (bool): whether to save the Hamiltonian to a file. If True, symmetry will be disabled.
        options (dict): additional commands to be passed to Psi4
        n_active_extract (int): number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals.
        n_occupied_extract (int): number of occupied molecular orbitals to
            include in the saved Hamiltonian. Must be less than or equal to
            n_active_extract. If None, all occupied orbitals are included,
            except the core orbitals if freeze_core_extract is set to True.
        freeze_core_extract (bool): If True, frozen core orbitals will always be
            doubly occupied in the saved Hamiltonian. Ignored if
            n_occupied_extract is not None.
        save_rdms (bool): If True, save 1- and 2-RDMs
        
    Returns:
        tuple: The results of the calculation (dict) and the Hamiltonian
            (openfermion.InteractionOperator).
    """

    if n_active_extract is not None and n_occupied_extract is not None:
        if n_occupied_extract > n_active_extract:
            raise ValueError(
                f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than total number of molecular orbitals to extract ({n_active_extract})."
            )

    if save_rdms and not method in ['fci', 'cis', 'cisd', 'cisdt', 'cisdtq']:
        print(f"run_psi4 was called with method={method}")
        raise Warning(
            f"RDM calculation can only be performed for Configuration Interaction methods. save_rdms option will be ignored."
        )

    geometry_str = f"{charge} {multiplicity}\n"
    for atom in geometry["sites"]:
        geometry_str += "{} {} {} {}\n".format(
            atom["species"], atom["x"], atom["y"], atom["z"]
        )

    geometry_str += "\nunits angstrom\n"
    c1_sym = save_hamiltonian or n_active
    if c1_sym:
        geometry_str += "symmetry c1\n"

    molecule = psi4.geometry(geometry_str)
    psi4.set_options(
        {"reference": reference, "basis": basis, "freeze_core": freeze_core}
    )
    if method == 'fci' or method == 'cis' or method == 'cisd' or method == 'cisdt' or method == 'cisdtq':
        psi4.set_options({'qc_module' : 'detci'})
        if save_rdms:
            psi4.set_options({'opdm' : True, 'tpdm' : True})

    energy, wavefunction = psi4.energy(method, return_wfn=True)

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
        orbitals = None
        if n_active_extract is not None:
            orbitals = wavefunction.Ca().to_array(dense=True)
            n_orbitals = n_active_extract
            if n_occupied_extract is not None:
                if wavefunction.nalpha() != wavefunction.nbeta():
                    raise ValueError(
                        f"Requesting a number of occupied molecular orbitals not supported when number of alpha and beta electrons is unequal."
                    )
                if n_occupied_extract > wavefunction.nalpha():
                    raise ValueError(
                        f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than number of occupied molecular orbitals ({wavefunction.nalpha()})."
                    )
                n_orbitals += wavefunction.nalpha() - n_occupied_extract
            elif freeze_core_extract:
                n_orbitals += wavefunction.nfrzc()
            orbitals = orbitals[:, :n_orbitals]
            orbitals = psi4.core.Matrix.from_array(orbitals)
        hamiltonian = get_ham_from_psi4(
            wavefunction,
            mints,
            n_active_extract=n_active_extract,
            n_occupied_extract=n_occupied_extract,
            freeze_core_extract=freeze_core_extract,
            orbs=orbitals,
            nuclear_repulsion_energy=molecule.nuclear_repulsion_energy(),
        )

    if save_rdms:
        rdm = get_rdms_from_psi4(wavefunction, freeze_core, n_active_extract, n_occupied_extract, freeze_core_extract)
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        return results, hamiltonian, rdm
    else:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.clean_variables()
        return results, hamiltonian

def get_ham_from_psi4(
    wfn,
    mints,
    n_active_extract=None,
    n_occupied_extract=None,
    freeze_core_extract=False,
    orbs=None,
    nuclear_repulsion_energy=0,
):
    """Get a molecular Hamiltonian from a Psi4 calculation.

    Args:
        wfn (psi4.core.Wavefunction): Psi4 wavefunction object
        mints (psi4.core.MintsHelper): Psi4 molecular integrals helper
        n_active_extract (int): number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals, else you must provide
            active orbitals in orbs.
        n_occupied_extract (int): number of occupied molecular orbitals to
            include in the saved Hamiltonian. Must be less than or equal to
            n_active_extract. If None, all occupied orbitals are included,
            except the core orbitals if freeze_core_extract is set to True.
        freeze_core_extract (bool): If True, frozen core orbitals will always be
            doubly occupied in the saved Hamiltonian. Ignored if
            n_occupied_extract is not None.
        orbs (psi4.core.Matrix): Psi4 orbitals for active space transformations. Must
            include all occupied (also core in all cases).
        nuclear_repulsion_energy (float): The ion-ion interaction energy.
    
    Returns:
        hamiltonian (openfermion.ops.InteractionOperator): the electronic
            Hamiltonian.
    """

    assert wfn.same_a_b_orbs(), (
        "Extraction of Hamiltonian from wavefunction"
        + "with different alpha and beta orbitals not yet supported :("
    )

    # Note: code refactored to use Psi4 integral-transformation routines
    # no more storing the whole two-electron integral tensor when only an
    # active space is needed

    orbitals = wfn.Ca().to_array(dense=True)
    one_body_integrals = general_basis_change(
        np.asarray(mints.ao_kinetic()), orbitals, (1, 0)
    )
    one_body_integrals += general_basis_change(
        np.asarray(mints.ao_potential()), orbitals, (1, 0)
    )

    # Build the transformation matrices, i.e. the orbitals for which
    # we want the integrals, as Psi4.core.Matrix objects
    n_core_extract = 0
    if freeze_core_extract and n_occupied_extract is None:
        n_core_extract = wfn.nfrzc()
    elif n_occupied_extract is not None:
        if wfn.nalpha() != wfn.nbeta():
            raise ValueError(
                f"Requesting a number of occupied molecular orbitals not supported when number of alpha and beta electrons is unequal."
            )
        if n_occupied_extract > wfn.nalpha():
            raise ValueError(
                f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than number of occupied molecular orbitals ({wfn.nalpha()})."
            )

        n_core_extract = wfn.nalpha() - n_occupied_extract

    if n_active_extract is None:
        trf_mat = wfn.Ca()
        n_active_extract = wfn.nmo() - n_core_extract # - wfn.nfzvpi.sum() ?
    else:
        # If orbs is given, it allows us to perform the two-electron integrals
        # transformation only in the space of active orbitals. Otherwise, we
        # transform all orbitals and filter them out in the get_ham_from_integrals
        # function
        if orbs is None: # If n_acitve_extract is not None - how could possibly be orbs == None?
            trf_mat = wfn.Ca()
        else:
            assert (
                orbs.to_array(dense=True).shape[1] == n_active_extract + n_core_extract
            )
            trf_mat = orbs

    two_body_integrals = np.asarray(mints.mo_eri(trf_mat, trf_mat, trf_mat, trf_mat))
    n_orbitals = trf_mat.shape[1]
    two_body_integrals.reshape((n_orbitals, n_orbitals, n_orbitals, n_orbitals))
    two_body_integrals = np.einsum("psqr", two_body_integrals)

    # Truncate
    one_body_integrals[np.absolute(one_body_integrals) < EQ_TOLERANCE] = 0.0
    two_body_integrals[np.absolute(two_body_integrals) < EQ_TOLERANCE] = 0.0

    if n_active_extract is None and not freeze_core_extract:
        occupied_indices = None
        active_indices = None
    else:
        # Indices of occupied molecular orbitals
        occupied_indices = range(n_core_extract)

        # Indices of active molecular orbitals
        active_indices = range(n_core_extract, n_core_extract + n_active_extract)

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

def get_rdms_from_psi4(wfn, 
    freeze_core,
    n_active_extract=None, 
    n_occupied_extract=None,
    freeze_core_extract=False):
    """
    Args:
        wfn (psi4.core.Wavefunction): Psi4 wavefunction object
        freeze_core (bool): Whether to freeze occupied core orbitals
        n_active_extract (int): number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals, else you must provide
            active orbitals in orbs.
        n_occupied_extract (int): number of occupied molecular orbitals to
            include in the saved Hamiltonian. Must be less than or equal to
            n_active_extract. If None, all occupied orbitals are included,
            except the core orbitals if freeze_core_extract is set to True.
        freeze_core_extract (bool): If True, frozen core orbitals will always be
            doubly occupied in the saved Hamiltonian. Ignored if
            n_occupied_extract is not None.
    Returns:
        rdm (openfermion.ops.InteractionRDM): an openfermion object storing
            1- and 2-RDMs. 
    """

    one_rdm_a = np.array(wfn.get_opdm(
        0, 0, 'A', True)).reshape(wfn.nmo(), wfn.nmo()) # Note: this works even if freeze_core is True
    one_rdm_b = np.array(wfn.get_opdm(
        0, 0, 'B', True)).reshape(wfn.nmo(), wfn.nmo())

    # Get 2-RDM from CI calculation.
    n_core_extract = 0
    if freeze_core_extract and n_occupied_extract is None:
        n_core_extract = wfn.nfrzc() # This logic implies that cores are always the lowest energy orbitals.
    elif n_occupied_extract is not None:
        if wfn.nalpha() != wfn.nbeta():
            raise ValueError(
                f"Requesting a number of occupied molecular orbitals not supported when number of alpha and beta electrons is unequal."
            )
        if n_occupied_extract > wfn.nalpha():
            raise ValueError(
                f"Number of occupied molecular orbitals to extract ({n_occupied_extract}) is larger than number of occupied molecular orbitals ({wfn.nalpha()})."
            )

        n_core_extract = wfn.nalpha() - n_occupied_extract

    if n_active_extract is None:
        n_active_extract = wfn.nmo() - n_core_extract
    
    nfnmo = wfn.nmo() - wfn.frzcpi().sum() - wfn.frzvpi().sum() # I wouldn't work otherwise for 2-RDMs

    two_rdm_aa = np.array(wfn.get_tpdm(
        'AA', False)).reshape(nfnmo, nfnmo,
                                nfnmo, nfnmo)
    two_rdm_ab = np.array(wfn.get_tpdm(
        'AB', False)).reshape(nfnmo, nfnmo,
                                nfnmo, nfnmo)
    two_rdm_bb = np.array(wfn.get_tpdm(
        'BB', False)).reshape(nfnmo, nfnmo,
                                nfnmo, nfnmo)

    # Indices of active molecular orbitals
    active_indices = range(n_core_extract, n_core_extract + n_active_extract)

    # adjusting 1-RDM
    one_rdm_a = one_rdm_a[np.ix_(active_indices, active_indices)]
    one_rdm_b = one_rdm_b[np.ix_(active_indices, active_indices)]
    
    if freeze_core:
        active_indices = range(n_active_extract) # Need to double check if this actually works
    # If one also requested freeze_core_extract => we need to construct and save RHF RDMs for the core

    # adjusting 2-RDM
    two_rdm_aa = two_rdm_aa[np.ix_(active_indices,
                                  active_indices,
                                  active_indices,
                                  active_indices)]
    two_rdm_ab = two_rdm_ab[np.ix_(active_indices,
                                  active_indices,
                                  active_indices,
                                  active_indices)]
    two_rdm_bb = two_rdm_bb[np.ix_(active_indices,
                                   active_indices,
                                   active_indices,
                                   active_indices)]

    one_rdm, two_rdm = unpack_spatial_rdm(
    one_rdm_a, one_rdm_b, two_rdm_aa,
    two_rdm_ab, two_rdm_bb)

    rdms = InteractionRDM(one_rdm, two_rdm)

    return rdms

def unpack_spatial_rdm(one_rdm_a,
                       one_rdm_b,
                       two_rdm_aa,
                       two_rdm_ab,
                       two_rdm_bb):
    r"""
    Covert from spin compact spatial format to spin-orbital format for RDM.
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
    two_rdm = np.zeros((n_qubits, n_qubits,
                           n_qubits, n_qubits))

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
                    two_rdm[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_rdm_aa[p, r, q, s])
                    two_rdm[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (
                        two_rdm_bb[p, r, q, s])

                    # Handle case of mixed spin.
                    two_rdm[2 * p, 2 * q + 1, 2 * r, 2 * s + 1] = (
                        two_rdm_ab[p, r, q, s])
                    two_rdm[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        -1. * two_rdm_ab[p, s, q, r])
                    two_rdm[2 * p + 1, 2 * q, 2 * r + 1, 2 * s] = (
                        two_rdm_ab[q, s, p, r])
                    two_rdm[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        -1. * two_rdm_ab[q, r, p, s])

    # Map to physicist notation and return.
    two_rdm = np.einsum('pqsr', two_rdm)
    return one_rdm, two_rdm
