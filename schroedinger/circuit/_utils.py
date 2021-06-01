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

"""Utility function collection"""

from qiskit.circuit.quantumregister import QuantumRegister


def parse_complex_index(index):
    """
    Parses a index (single number or slice or list or range) made of
    either purely imaginary or real number(s).

    Arg:
        index (int or complex or slice or list or range): Index of to be parsed.
    Returns:
        tuple(int or slice or list, bool):  real index and boolean representing if index was real

    Raises:
        TypeError: If the `index` is made of complex but not a purely imaginary integer(s).
        TypeError: If the `index` is a list with different index types.
        TypeError: If the `index` is a slice with different index types (not regarding None).
        TypeError: If the `index` is neither a int / complex integer nor a list or slice or range.
    """
    if isinstance(index, complex):
        if index.real != 0 or int(index.imag) != index.imag:
            raise TypeError("Complex indices must be purely imaginary integers.")
        return int(index.imag), False

    if isinstance(index, (int, range)):
        return index, True

    if isinstance(index, list):
        if len(index) == 0:
            return index, True

        if len(set(type(idx) for idx in index)) != 1:
            raise TypeError("Indices must either be purely imaginary integers or real integers.")

        if isinstance(index[0], complex):
            if any(idx.real != 0 or int(idx.imag) != idx.imag for idx in index):
                raise TypeError("Complex indices must be purely imaginary integers.")
            return [idx.imag for idx in index], False

        if isinstance(index[0], int):
            return index, True

        raise TypeError("Indices in a list must be real or imaginary integers.")

    if isinstance(index, slice):
        slice_types = set(
            type(idx) for idx in (index.start, index.stop, index.step) if idx is not None
        )
        if len(slice_types) != 1:
            raise TypeError("All slice indices must either have the same type or be None.")

        if any(isinstance(idx, complex) for idx in (index.start, index.stop, index.step)):
            if any(idx.real != 0 or int(idx.imag) != idx.imag
                   for idx in (index.start, index.stop, index.step) if idx is not None):
                raise TypeError("Complex slice indices must be purely imaginary integers.")

            real_slice = slice(
                *(int(idx.imag) if idx is not None else None
                  for idx in (index.start, index.stop, index.step))
            )
            return real_slice, False

        if any(isinstance(idx, int) for idx in (index.start, index.stop, index.step)):
            return index, True

        raise TypeError("Indices in a slice must be real or imaginary integers.")

    raise TypeError("Index must be of type int or slice or list or range.")


def qargs_to_indices(circuit, qargs):
    """
    Parses qargs to indices for qudits and qubits.

    Args:
        circuit (~circuit.QuditCircuit): circuit of indexed qudits and qubits
        qargs: quantum bit (qubit/qubit) representations

    Returns:
        tuple(List(int), list(int)): Tuple of list of qudit indices and list of qubit indices
    """
    qudits = []
    qubits = []

    if qargs is None:
        qudits.extend(list(range(circuit.num_qudits)))
        qudits.extend(list(range(circuit.num_single_qubits)))
    else:
        for qarg in qargs:
            if isinstance(qargs, QuantumRegister):
                if type(qargs) is not QuantumRegister:
                    qubits.extend([qarg[j * 1j] for j in range(qarg.size)])
                else:
                    qubits.extend([qarg[j] for j in range(qarg.size)])

            qarg, is_real = parse_complex_index(qarg)

            if is_real:
                bits = qubits
                num_bits = circuit.num_qubits
            else:
                bits = qudits
                num_bits = circuit.num_qudits

            if isinstance(qarg, list):
                bits.extend(qarg)
            elif isinstance(qarg, range):
                bits.extend(list(qarg))
            elif isinstance(qarg, slice):
                bits.extend(list(range(num_bits))[qarg])
            else:
                bits.append(qarg)

    return qudits, qubits
