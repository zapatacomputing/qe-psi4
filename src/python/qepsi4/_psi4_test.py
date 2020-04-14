import unittest
from qepsi4 import run_psi4
from openfermion import (jordan_wigner, jw_get_ground_state_at_particle_number,
                         qubit_operator_sparse)

class TestChem(unittest.TestCase):

    def test_run_psi4(self):

        geometry = {"sites": [
            {'species': 'H', 'x': 0, 'y': 0, 'z': 0},
            {'species': 'H', 'x': 0, 'y': 0, 'z': 1.7}
        ]}

        results, hamiltonian = run_psi4(geometry, save_hamiltonian=True)
        self.assertAlmostEqual(results['energy'], -0.8544322638069642)
        self.assertEqual(results['n_alpha'], 1)
        self.assertEqual(results['n_beta'], 1)
        self.assertEqual(results['n_mo'], 2)
        self.assertEqual(results['n_frozen_core'], 0)
        self.assertEqual(results['n_frozen_valence'], 0)

        self.assertEqual(hamiltonian.n_qubits, 4)
        qubit_operator = qubit_operator_sparse(jordan_wigner(hamiltonian))
        energy, state = jw_get_ground_state_at_particle_number(qubit_operator, 2)
        
        results_cisd, hamiltonian = run_psi4(geometry, method='ccsd')

        # For this system, the CCSD energy should be exact.
        self.assertAlmostEqual(energy, results_cisd['energy'])
