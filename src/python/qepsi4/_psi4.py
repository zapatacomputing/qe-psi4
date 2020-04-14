import psi4
import numpy as np
from openfermion import InteractionOperator, general_basis_change, MolecularData
from openfermion.config import EQ_TOLERANCE

def run_psi4(geometry, basis='STO-3G', multiplicity=1, charge=0,
        method='scf', reference='rhf',
        freeze_core=False, n_active=None,
        save_hamiltonian=False, options=None,
        n_active_extract=None, freeze_core_extract=False):
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
        freeze_core_extract (bool): whether to freeze core orbitals as doubly
            occupied in the saved Hamiltonian. When saving UCCSD/MP2 amplitudes, use this option
            to freeze amplitudes. It will only work as expected if freeze_core is False and the 
            wavefunction does not have frozen cores from another source
        
    Returns:
        tuple: The results of the calculation (dict) and the Hamiltonian
            (openfermion.InteractionOperator).
    """

    geometry_str = f'{charge} {multiplicity}\n'
    for atom in geometry['sites']:
        geometry_str += '{} {} {} {}\n'.format(atom['species'],
                                               atom['x'],
                                               atom['y'],
                                               atom['z'])

    geometry_str += '\nunits angstrom\n'
    c1_sym = save_hamiltonian or n_active
    if c1_sym:
        geometry_str += 'symmetry c1\n'
    
    molecule = psi4.geometry(geometry_str)
    psi4.set_options({'reference': reference, 'basis': basis})
    energy, wavefunction = psi4.energy(method, return_wfn=True)

    results = {
        'energy': energy,
        'n_alpha': wavefunction.nalpha(),
        'n_beta': wavefunction.nbeta(),
        'n_mo': wavefunction.nmo(),
        'n_frozen_core': wavefunction.nfrzc(),
        'n_frozen_valence': wavefunction.frzvpi().sum()
    }

    hamiltonian = None
    if save_hamiltonian:
        mints = psi4.core.MintsHelper(wavefunction.basisset())
        orbitals = None
        if n_active_extract is not None:
            orbitals = wavefunction.Ca().to_array(dense=True)
            n_orbitals = n_active_extract
            if freeze_core_extract:
                n_orbitals += wavefunction.nfrzc()
            orbitals = orbitals[:, :n_orbitals]
            orbitals = psi4.core.Matrix.from_array(orbitals)
        hamiltonian = get_ham_from_psi4(wavefunction,
                                        mints,
                                        n_active_extract,
                                        freeze_core_extract,
                                        orbitals,
                                        molecule.nuclear_repulsion_energy())

    return results, hamiltonian

def get_ham_from_psi4(wfn, mints, n_active_extract=None, freeze_core_extract=False, 
                      orbs=None, nuclear_repulsion_energy=0):
    """Get a molecular Hamiltonian from a Psi4 calculation.

    Args:
        wfn (psi4.core.Wavefunction): Psi4 wavefunction object
        mints (psi4.core.MintsHelper): Psi4 molecular integrals helper
        n_active_extract (int): number of molecular orbitals to include in the
            saved Hamiltonian. If None, includes all orbitals, else you must provide
            active orbitals in orbs.
        freeze_core_extract (bool): whether to freeze core orbitals as doubly
            occupied in the saved Hamiltonian.
        orbs (psi4.core.Matrix): Psi4 orbitals for active space transformations. Must
            include all occupied (also core in all cases).
        nuclear_repulsion_energy (float): The ion-ion interaction energy.
    
    Returns:
        hamiltonian (openfermion.ops.InteractionOperator): the electronic
            Hamiltonian. Note that the ion-ion electrostatic energy is not
            included.
    """

    assert wfn.same_a_b_orbs(), "Extraction of Hamiltonian from wavefunction" + \
        "with different alpha and beta orbitals not yet supported :("

    # Note: code refactored to use Psi4 integral-transformation routines
    # no more storing the whole two-electron integral tensor when only an
    # active space is needed

    orbitals = wfn.Ca().to_array(dense=True)
    one_body_integrals = general_basis_change(
        np.asarray(mints.ao_kinetic()), orbitals, (1, 0))
    one_body_integrals += general_basis_change(
        np.asarray(mints.ao_potential()), orbitals, (1, 0))

    # Build the transformation matrices, i.e. the orbitals for which
    # we want the integrals, as Psi4.core.Matrix objects
    n_core_extract = 0
    if freeze_core_extract:
        n_core_extract = wfn.nfrzc()
    if n_active_extract is None:
        trf_mat = wfn.Ca()
        n_active_extract = wfn.nmo() - n_core_extract
    else:
    # If orbs is given, it allows us to perform the two-electron integrals
    # transformation only in the space of active orbitals. Otherwise, we
    # transform all orbitals and filter them out in the get_ham_from_integrals
    # function
        if orbs is None:
            trf_mat = wfn.Ca()
        else:
            assert(orbs.to_array(dense=True).shape[1] == n_active_extract + n_core_extract)
            trf_mat = orbs

    two_body_integrals = np.asarray(mints.mo_eri(trf_mat, trf_mat, trf_mat, trf_mat))
    n_orbitals = trf_mat.shape[1]
    two_body_integrals.reshape((n_orbitals, n_orbitals,
                                n_orbitals, n_orbitals))
    two_body_integrals = np.einsum('psqr', two_body_integrals)

    # Truncate
    one_body_integrals[np.absolute(one_body_integrals) < EQ_TOLERANCE] = 0.
    two_body_integrals[np.absolute(two_body_integrals) < EQ_TOLERANCE] = 0.

    if n_active_extract is None and not freeze_core_extract:
        occupied_indices = None
        active_indices = None    
    else:
        # Indices of occupied molecular orbitals
        occupied_indices = range(n_core_extract)

        # Indices of active molecular orbitals
        active_indices = range(n_core_extract, n_core_extract + n_active_extract)

    # In order to keep the MolecularData class happy, we need a 'valid' molecule
    molecular_data = MolecularData(geometry=[('H', (0, 0, 0))],
                                   basis='',
                                   multiplicity=2)

    molecular_data.one_body_integrals = one_body_integrals
    molecular_data.two_body_integrals = two_body_integrals
    molecular_data.nuclear_repulsion = nuclear_repulsion_energy
    hamiltonian = molecular_data.get_molecular_hamiltonian(occupied_indices,
                                            active_indices)

    return(hamiltonian)
