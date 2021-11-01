############################################################################
#   Copyright 2017 The OpenFermion Developers
#   Modifications copyright 2021 Zapata Computing, Inc. for compatibility reasons.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
############################################################################

from typing import Callable, List

import numpy as np
from openfermion import FermionOperator

"""
Original function:
https://github.com/quantumlib/OpenFermion-Psi4/blob/master/openfermionpsi4/_psi4_conversion_functions.py#L97
"""  # noqa: E501


def parse_amplitudes_from_psi4_ccsd(
    wfn,
    psi_filename,
    n_active_extract=None,
    freeze_core_extract=False,
    freeze_core=False,
    n_frozen_amplitudes=0,
    get_mp2_amplitudes=False,
):
    """Parse coupled cluster singles and doubles amplitudes from psi4 file.
    Args:
      wfn (psi4.core.Wavefunction): Psi4 wavefunction object
      psi_filename(str): Filename of psi4 output file
      n_active_extract (int): number of molecular orbitals to include in the
            saved amplitudes. If None, includes all orbitals.
      freeze_core_extract (bool): whether to freeze core orbitals as doubly
            occupied in the saved Amplitudes. Only works if the wavefunction
            does not contain frozen cores already and freeze_core is False.
      freeze_core (bool): Must be True if the wavefunction wfn contains frozen
                          core orbitals, otherwise amplitudes are not extracted
                          correctly.
      get_mp2_amplitudes (bool): whether to get the mp2 initial amplitudes instead
        of the CCSD amplitudes.
      n_frozen_amplitudes (int): number of molecular core orbitals to exclude in
        the saved amplitudes. If None, includes all occupied orbitals. Requires
        freeze_core False and freeze_core_extract True.
    Returns:
      molecule(InteractionOperator): Molecular Operator instance holding ccsd
        amplitudes
    """
    output_buffer = [line for line in open(psi_filename)]
    end_read = len(output_buffer)

    # the reasoning here is that where freeze_core = True
    # the Psi4 CCSD routine reset the orbitals number
    # making the first active 0, when printing the amplitudes
    if freeze_core:
        n_frozen_so = 2 * wfn.nfrzc()
        n_frozen = wfn.nfrzc()
        n_frozen_amplitudes = 0
    else:
        if freeze_core_extract:
            n_frozen_so = 2 * n_frozen_amplitudes
            n_frozen = n_frozen_amplitudes
        else:
            n_frozen_so = 0
            n_frozen = 0
            n_frozen_amplitudes = 0

    # define active window
    if n_active_extract is None:
        n_spin_orbitals = 2 * wfn.nmo() - n_frozen_so
    else:
        n_spin_orbitals = 2 * n_active_extract

    # adjust number of electrons based on frozen orbitals
    n_alpha_electrons = wfn.nalpha() - n_frozen
    n_beta_electrons = wfn.nbeta() - n_frozen

    # Find Start Indices
    if get_mp2_amplitudes:
        for i, line in enumerate(output_buffer):
            if "MP2 correlation energy" in line:
                mp2_line = i
                break
    else:
        mp2_line = None

    for i, line in enumerate(output_buffer):
        if "Solving CC Amplitude Equations" in line:
            ccsd_line = i
            break

    (
        T1IA_index,
        T1ia_index,
        T2IJAB_index,
        T2ijab_index,
        T2IjAb_index,
    ) = _extract_T_indices(output_buffer, get_mp2_amplitudes, mp2_line, ccsd_line)

    # Determine if calculation is restricted / closed shell or otherwise
    restricted = T1ia_index is None and T2ijab_index is None

    # Define local helper routines for clear indexing of orbitals
    # the number of frozen orbitals determine the zero index (frozen
    # orbitals become negative, then excluded). Then the extend of the
    # system is given by the active window
    def alpha_occupied(i):
        return 2 * (i - n_frozen_amplitudes)

    def alpha_unoccupied(i):
        return 2 * (i + n_alpha_electrons)

    def beta_occupied(i):
        return 2 * (i - n_frozen_amplitudes) + 1

    def beta_unoccupied(i):
        return 2 * (i + n_beta_electrons) + 1

    # list to store operators + amplitudes
    single_amps = []
    double_amps = []

    # Read T1's
    if get_mp2_amplitudes:
        # Make sure we stop reading before CCSD amplitudes
        end_read = ccsd_line

        # Generate all possible singlet excitations
        for i in range(n_alpha_electrons - 1, n_alpha_electrons):
            single_amps.append([[alpha_unoccupied(0), alpha_occupied(i)], 0.0])
            single_amps.append([[beta_unoccupied(0), beta_occupied(i)], 0.0])

    else:
        # read single excitations
        if T1IA_index is not None:
            ret = _extract_amplitudes_from_T1_lines(
                output_buffer[T1IA_index + 1 : end_read],
                n_spin_orbitals,
                restricted,
                fn_list=[alpha_unoccupied, alpha_occupied],
                restricted_fn_list=[beta_unoccupied, beta_occupied],
            )

            single_amps.extend(ret)

        if T1ia_index is not None:
            ret = _extract_amplitudes_from_T1_lines(
                output_buffer[T1IA_index + 1 : end_read],
                n_spin_orbitals,
                restricted,
                fn_list=[beta_unoccupied, beta_occupied],
            )

            single_amps.extend(ret)

    # Read T2's
    if T2IJAB_index is not None:
        ret = _extract_amplitudes_from_T1_lines(
            output_buffer[T2IJAB_index + 1 : end_read],
            n_spin_orbitals,
            restricted,
            fn_list=[alpha_unoccupied, alpha_occupied] * 2,
            restricted_fn_list=[beta_unoccupied, beta_occupied] * 2,
        )

        double_amps.extend(ret)

    if T2ijab_index is not None:
        ret = _extract_amplitudes_from_T1_lines(
            output_buffer[T2ijab_index + 1 : end_read],
            n_spin_orbitals,
            restricted,
            fn_list=[beta_unoccupied, beta_occupied] * 2,
        )

        double_amps.extend(ret)

    if T2IjAb_index is not None:
        ret = _extract_amplitudes_from_T1_lines(
            output_buffer[T2IjAb_index + 1 : end_read],
            n_spin_orbitals,
            restricted,
            fn_list=[alpha_unoccupied, alpha_occupied, beta_unoccupied, beta_occupied],
            restricted_fn_list=[
                beta_unoccupied,
                beta_occupied,
                alpha_unoccupied,
                alpha_occupied,
            ],
        )

        double_amps.extend(ret)

    # sort amplitudes
    single_amps.sort(key=lambda x: abs(x[1]), reverse=True)
    double_amps.sort(key=lambda x: abs(x[1]), reverse=True)

    # separate operators and initial guesses
    parameters_single = [x[1] for x in single_amps]
    parameters_double = [x[1] for x in double_amps]
    single_ops = [x[0] for x in single_amps]
    double_ops = [x[0] for x in double_amps]

    print("single_ops", single_ops)
    print("double_ops", double_ops)

    # Generate Fermion Operator
    s_generator = FermionOperator()
    d_generator = FermionOperator()

    # Add single excitations
    for n, [i, j] in enumerate(single_ops):
        i, j = int(i), int(j)
        s_generator += FermionOperator(((i, 1), (j, 0)), parameters_single[n])

    # Add double excitations
    for m, [i, j, k, l] in enumerate(double_ops):  # noqa: E741
        i, j, k, l = int(i), int(j), int(k), int(l)  # noqa: E741
        d_generator += FermionOperator(
            ((i, 1), (j, 0), (k, 1), (l, 0)), parameters_double[m]
        )

    # order of single and double excitation operators is fixed
    fermion_generator = s_generator + d_generator

    return fermion_generator


def _extract_T_indices(
    output_buffer: List[str], get_mp2_amplitudes: bool, mp2_line: int, ccsd_line: int
):
    T1IA_index = T1ia_index = T2IJAB_index = T2ijab_index = T2IjAb_index = None

    is_reading = False
    for i, line in enumerate(output_buffer):
        if (get_mp2_amplitudes and i > mp2_line and i < ccsd_line) or (
            i > ccsd_line and not get_mp2_amplitudes
        ):
            is_reading = True
        else:
            is_reading = False

        if is_reading:
            if "Largest TIA Amplitudes:" in line:
                T1IA_index = i

            elif "Largest Tia Amplitudes:" in line:
                T1ia_index = i

            elif "Largest TIJAB Amplitudes:" in line:
                T2IJAB_index = i

            elif "Largest Tijab Amplitudes:" in line:
                T2ijab_index = i

            elif "Largest TIjAb Amplitudes:" in line:
                T2IjAb_index = i

    return (
        T1IA_index,
        T1ia_index,
        T2IJAB_index,
        T2ijab_index,
        T2IjAb_index,
    )


def _functions_to_list(
    list_of_fn: List[Callable], list_of_params: List[int], ivals: List[str]
):
    res = []
    for fn, param in zip(list_of_fn, list_of_params):
        res.append(fn(int(ivals[param])))

    return res


def _extract_amplitudes_from_T1_lines(
    buffer: List[str],
    n_spin_orbitals: int,
    restricted: bool,
    fn_list: List[Callable],
    restricted_fn_list: List[Callable] = [],
):
    ret = []

    param_list = [1, 0] if len(fn_list) == 2 else [2, 0, 3, 1]

    for line in buffer:
        ivals = line.split()
        if not ivals:
            break

        op = _functions_to_list(fn_list, param_list, ivals)

        if np.prod(np.array(op) < n_spin_orbitals) and np.prod(np.array(op) >= 0):
            ret.append([op, float(ivals[len(fn_list)]) / float(len(fn_list) / 2)])

            if restricted:
                op = _functions_to_list(restricted_fn_list, param_list, ivals)

                ret.append([op, float(ivals[len(fn_list)]) / float(len(fn_list) / 2)])

    return ret
