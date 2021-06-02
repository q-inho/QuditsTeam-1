# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumcircuitdata.py from the original Qiskit-Terra code.
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
#
###############################################################################
#
# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
###############################################################################

"""
A wrapper class for the purposes of validating modifications to QuditCircuit.data
while maintaining the interface of a python list.
Qudit context is stored additionally to the qubit an classical bit context for each
instruction in the data. There are two forms of each rule in the data:
(instruction, qubits, clbits) where qubits consists of qubits of qudits and single qubits,
(instruction, qudits, qubits, clbits) where qubits are only single qubits.
The second variant can be accessed via imaginary indices, ``qd_get`` or the ``qd_iter`` method.
"""

from qiskit.circuit.quantumcircuitdata import QuantumCircuitData
from qiskit.circuit.exceptions import CircuitError

from ._utils import parse_complex_index
from .quditinstruction import QuditInstruction


class QuditCircuitData(QuantumCircuitData):
    """A wrapper class for the purposes of validating modifications to QuditCircuit.data
    while maintaining the interface of a python list. Allows access to rules of the
    form (instruction, qubits, clbits) or (instruction, qudits, qubits, clbits)."""

    @staticmethod
    def to_rule(qd_rule):
        """
        Converts a rule with qudit context to a rule without qudit context.

        Args:
            qd_rule (tuple): Tuple (instruction, qdargs, qargs, cargs),
                qdargs / qargs / cargs is a list of qudits / qubits / classical bits.

        Returns:
            Tuple: Converted rule (instruction, qargs, cargs).
        """
        if len(qd_rule) != 4:
            raise CircuitError(
                f"Invalid rule form, expected (instruction, qudits, qubits, clbits), "
                f"got {qd_rule}."
            )
        inst, qdargs, qargs, cargs = qd_rule
        return inst, [qubit for qudit in qdargs for qubit in qudit] + qargs, cargs

    def to_qd_rule(self, rule):
        """
        Converts a rule without qudit context to a rule with qudit context.

        Args:
            rule (tuple): Tuple (instruction, qargs, cargs),
                qargs / cargs is a list of qubits / classical bits.

        Returns:
            Tuple: Converted rule (instruction, qdargs, qargs, cargs).
        """
        if len(rule) != 3:
            raise CircuitError(
                f"Invalid rule form, expected (instruction, qubits, clbits), "
                f"got {rule}."
            )
        inst, qargs, cargs = rule
        qdargs, qargs = self._circuit._split_qargs(qargs)
        return inst, qdargs, qargs, cargs

    def qd_get(self, key):
        """Returns rule including qudit context at index key or a list of tuples
        with slice as key. Does not relay modifications to original data, please use
        single item modifications e.g. data[4j] = (inst, qdargs, qargs, cargs) instead."""
        if isinstance(key, int):
            return self.to_qd_rule(super().__getitem__(key))
        return [self.to_qd_rule(rule) for rule in super().__getitem__(key)]

    def __getitem__(self, key):
        """
        Supports imaginary numbers (e.g. 0j) for indexing into rules with qudit context.

        Arg:
            key (int or complex or slice or list): Index of the rule to be retrieved.
        Returns:
            Rule / list of rules with qudit context if key is a purely imaginary
            integer / slice with purely imaginary integers.
            Rule / list of rules without qudit context if key is a real
            integer / slice with real integers.

        Raises:
            TypeError: If the `index` is made of complex but not a purely imaginary integer(s).
            TypeError: If the `index` is a list with different index types.
            TypeError: If the `index` is a slice with different index types (not regarding None).
            TypeError: If the `index` is neither a int / complex integer nor a list or slice.
        """
        key, is_real = parse_complex_index(key)
        if is_real:
            return super().__getitem__(key)
        return self.qd_get(key)

    def __setitem__(self, key, value):
        """
        Supports imaginary numbers (e.g. 0j) to set rules with qudit context.

        Arg:
            key (int): Index of the rule to set.
            value (tuple): Rule containing instruction, qdargs (optional), qargs, cargs.

        Raises:
            CircuitError: If the instruction is not a QuditInstruction but `value` includes qdargs.
            CircuitError: If `value` cannot be interpreted without broadcasting.
        """
        key, is_real = parse_complex_index(key)

        if is_real:
            inst, qargs, cargs = value
            qdargs = []
        else:
            inst, qdargs, qargs, cargs = value

        if hasattr(inst, "to_qd_instruction"):
            inst = inst.to_qd_instruction()

        if isinstance(inst, QuditInstruction):
            expanded_partial_qdargs = [
                self._circuit.qdit_argument_conversion(qdarg) for qdarg in qdargs or []
            ]
            flat_partial_qdargs = [
                qdarg for mixed_qdargs in expanded_partial_qdargs for qdarg in mixed_qdargs
            ]
            expanded_mixed_qargs = [
                self._circuit.qbit_argument_conversion(
                    self._circuit._offset_qubit_representation(qarg)
                ) for qarg in qargs or []
            ]
            flat_mixed_qargs = [
                qarg for mixed_qargs in expanded_mixed_qargs for qarg in mixed_qargs
            ]
            expanded_cargs = [
                self._circuit.cbit_argument_conversion(carg) for carg in cargs or []
            ]

            qudits, single_qubits = self._circuit._split_qargs(flat_mixed_qargs)
            qdargs = qudits + flat_partial_qdargs
            qargs = single_qubits
            cargs = [carg for cargs in expanded_cargs for carg in cargs]

            inst, qargs, cargs = self.to_rule((inst, qdargs, qargs, cargs))

        elif qdargs:
            raise CircuitError(
                "Instruction argument is not an QuditInstruction, "
                "but value includes qdargs."
            )

        print(inst, qargs, cargs)
        super().__setitem__(key, (inst, qargs, cargs))

    def __repr__(self):
        """returns qudit data representation"""
        return repr(self[0j:])

    def __mul__(self, n):
        """
        Multiplies and returns data*n if `n` is real integer,
        qd_data*n if `n` is a purely imaginary integer.

        Raises:
            TypeError: If the `n` is complex but not a purely imaginary integer.
        """
        if isinstance(n, complex):
            if n.real != 0 or int(n.imag) != n.imag:
                raise TypeError("Complex factors must be purely imaginary integers.")

            return self[0j:] * int(n.imag)
        return super().__mul__(n)

    def __rmul__(self, n):
        """
        Multiplies and returns n*data if `n` is real integer,
        n*qd_data if `n` is a purely imaginary integer.

        Raises:
            TypeError: If the `n` is complex but not a purely imaginary integer.
        """
        if isinstance(n, complex):
            if n.real != 0 or int(n.imag) != n.imag:
                raise TypeError("Complex factors must be purely imaginary integers.")

            return int(n.imag) * self[0j:]
        return super().__rmul__(n)
