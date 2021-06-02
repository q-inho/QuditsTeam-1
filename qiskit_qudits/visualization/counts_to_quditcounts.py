# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
#
# (C) Copyright 2021 Hoang Van Do, Tim Alexis Körner, Inho Choi, Timothé Presles and Élie Gouzien.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Converts resulting count dict list of the qasm_simulator backend from qiskit Aer."""

from qiskit.exceptions import QiskitError

from qiskit_qudits.circuit.quditmeasure import QuditMeasure


def to_qudit_counts(circuit_list, count_dict_list, fillchar='#'):
    """
    Converts each qudit measurement result in the count dict list
    of the qasm_simulator from a bit string to decimal representation,
    padded with ``fillchar``.

    Args:
        circuit_list (QuditCircuit or list(QuditCircuit)):
            qudit circuits the experiments were simulated on
        count_dict_list (dict or list(dict)): return value of job.result().get_counts()
        fillchar (str): Optional. Single character for padding.

    Returns:
        dict or list(dict): new count dict or count dict list

    Raises:

    """
    if len(fillchar) != 1:
        raise QiskitError("Invalid fill character")

    if not isinstance(circuit_list, list):
        circuit_list = [circuit_list]
    if not isinstance(count_dict_list, list):
        count_dict_list = [count_dict_list]

    if len(circuit_list) != len(count_dict_list):
        raise QiskitError("Different number of circuits and experiments.")

    new_count_dict_list = []

    for circuit, count_dict in zip(circuit_list, count_dict_list):

        # contains chunks of indices for classical bits used for qudit measurements
        measured_clbit_indices = []
        for inst, qargs, cargs in circuit.data:
            if isinstance(inst, QuditMeasure):
                indices = [circuit.clbits.index(clbit) for clbit in cargs]
                if sorted(indices) != list(range(min(indices), max(indices) + 1)):
                    raise QiskitError(
                        "To interpret counts as qudit counts, all measured qudits (i.e. their "
                        "underlying qubits) must be measured in adjacent Classical bits."
                    )
                measured_clbit_indices.append(indices)

        new_count_dict = {}

        for bit_string in count_dict:
            separator_indices = [
                idx for idx, char in enumerate(next(iter(count_dict))) if char == " "
            ]
            rev_bit_list = list(reversed(bit_string.replace(" ", "")))

            for clbit_indices in measured_clbit_indices:
                value = sum(rev_bit_list[idx] * 2**num for num, idx in enumerate(clbit_indices))
                str_value = str(value)
                str_value = fillchar * (len(clbit_indices) - len(str_value)) + str_value
                rev_list_value = list(reversed(str_value))

                start = len(rev_bit_list) - max(clbit_indices)
                stop = start + len(clbit_indices)
                rev_bit_list[start:stop] = rev_list_value

            short_bit_list = list(reversed(rev_bit_list))
            for idx in separator_indices:
                short_bit_list.insert(idx, " ")
            new_bit_string = "".join(short_bit_list)

            new_count_dict[new_bit_string] = count_dict[bit_string]

    if len(new_count_dict_list) == 1:
        return new_count_dict_list[0]
    return new_count_dict_list
