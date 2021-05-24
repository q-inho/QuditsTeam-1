# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of instruction.py from the original Qiskit-Terra code.
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
A qudit quantum instruction.
Qudit instructions additionally hold the number of affected qudits.
"""
import warnings
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister

from .quditregister import QuditRegister

_CUTOFF_PRECISION = 1e-10


class QuditInstruction(Instruction):
    """Qudit quantum instruction. Both definition """

    def __init__(self, name, qudit_dimensions, num_single_qubits, *args, **kwargs):
        """Create a new qudit instruction.

        Args:
            name (str): instruction name
            qudit_dimensions (list[int]): A list of int with dimensions of
                multiple qudits in order.
            args: Additional arguments passed on to the superclass Instruction.
            kwargs: Additional keyword arguments passed on to the superclass Instruction.

        Raises:
            TypeError: If ``qudit_dimensions`` has an incorrect type.
            CircuitError: If a dimension in ``qudit_dimensions`` is smaller than two.
        """
        if not isinstance(qudit_dimensions, list) or \
                any(not isinstance(dimension, int) for dimension in qudit_dimensions):
            raise TypeError(
                "qudit_dimensions must be a list of integers"
            )
        if any(d < 2 for d in qudit_dimensions):
            raise CircuitError(
                "qudit dimension must be 2 or higher"
            )

        # todo seperate qudits, ancilla qubits, unerlying qubits in Instruction
        # todo is self._qd_definition necessary? or works _definition recursivly
        self.qudit_dimensions = qudit_dimensions
        self.num_single_qubits = num_single_qubits

        # map qudits to qubits for underlying Instruction
        num_qubits = int(sum(np.ceil(np.log2(dimension)) for dimension in qudit_dimensions))
        num_qubits += num_single_qubits
        super().__init__(name, num_qubits, *args, **kwargs)

    def __eq__(self, other):
        """Two qudit instructions are the same if they have the same name,
        same qudit dimensions, same numbers of qubits and classical bits, and same params.

        Args:
            other (QuditInstruction): other instruction

        Returns:
            bool: are self and other equal.
        """
        if type(self) is not type(other) or \
                sorted(self.qudit_dimensions) != sorted(other.qudit_dimensions):
            return False
        return super().__eq__(other)

    def soft_compare(self, other: "QuditInstruction") -> bool:
        """
        Soft comparison between qudit gates. Their names, qudit dimensions, number of qubits,
        and classical bit numbers must match. The number of parameters must match. Each parameter
        is compared. If one is a ParameterExpression then it is not taken into
        account.

        Args:
            other (QuditInstruction): other instruction.

        Returns:
            bool: are self and other equal up to parameter expressions.
        """
        if type(self) is not type(other) or \
                sorted(self.qudit_dimensions) != sorted(other.qudit_dimensions):
            return False
        return super().soft_compare(other)

    def assemble(self):
        """Assemble a QasmQobjInstruction"""
        warnings.warn(
            "Assembly of QasmQobjInstruction not supported for qudits, "
            "defaulting to assembly without qudits."
        )
        return super().assemble()

    def reverse_ops(self):
        """For a composite instruction, reverse the order of sub-instructions.

        This is done by recursively reversing all sub-instructions.
        It does not invert any gate. The can handle sub-instructions of
        both QuditInstruction and Instruction type.

        Returns:
            QuditInstruction: a new instruction with sub-instructions reversed.
        """
        if not self._definition:
            return self.copy()

        reverse_inst = self.copy(name=self.name + "_reverse")
        reverse_inst.definition._data = [
            (inst[0].reverse_ops(), *inst[1:]) for inst in reversed(self._definition)
        ]
        return reverse_inst

    def inverse(self):
        """Invert this instruction.

        If the instruction is composite (i.e. has a definition),
        then its definition will be recursively inverted.

        Special instructions inheriting from QuditInstruction can
        implement their own inverse.

        Returns:
            QuditInstruction: a fresh instruction for the inverse

        Raises:
            CircuitError: If the instruction is not composite
                and an inverse has not been implemented for it.
        """
        if self.definition is None:
            raise CircuitError("inverse() not implemented for %s." % self.name)

        from . import QuditCircuit, QuditGate  # pylint: disable=cyclic-import

        if self.num_clbits:
            inverse_gate = QuditInstruction(
                name=self.name + "_dg",
                qudit_dimensions=self.qudit_dimensions,
                num_qubits=self.num_qubits,
                num_clbits=self.num_clbits,
                params=self.params.copy()
            )

        else:
            inverse_gate = QuditGate(
                name=self.name + "_dg",
                qudit_dimensions=self.qudit_dimensions,
                num_qubits=self.num_qubits,
                params=self.params.copy()
            )

        inverse_gate.definition = QuditCircuit(
            *self.definition.qdregs,
            *self.definition.qregs,
            *self.definition.cregs,
            global_phase=-self.definition.global_phase
        )
        inverse_gate.definition._data = [
            (inst[0].inverse(), *inst[1:]) for inst in reversed(self._definition)
        ]

        return inverse_gate

    def broadcast_arguments(self, qdargs, qargs, cargs):
        """
        Validation of the arguments.

        Args:
            qdargs (List): List of d-dimensional quantum bit arguments.
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.

        Yields:
            Tuple(List, List, List): A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """
        if len(qdargs) != len(self.qudit_dimensions):
            raise CircuitError(
                f"The amount of qudit arguments {len(qdargs)} does not match"
                f" the instruction expectation ({len(self.qudit_dimensions)})."
            )
        if len(qargs) != self.num_qubits:
            raise CircuitError(
                f"The amount of qubit arguments {len(qargs)} does not match"
                f" the instruction expectation ({self.num_qubits})."
            )

        #  [[qd[0], qd[1]], [q[0], q[1]], [c[0], c[1]]] -> [qd[0], q[0], c[0]], [qd[1], q[1], c[1]]
        flat_qdargs = [qdarg for sublist in qdargs for qdarg in sublist]
        flat_qargs = [qarg for sublist in qargs for qarg in sublist]
        flat_cargs = [carg for sublist in cargs for carg in sublist]
        yield flat_qdargs, flat_qargs, flat_cargs

    def _return_repeat(self, exponent):
        return QuditInstruction(
            name="%s*%s" % (self.name, exponent),
            qudit_dimensions=self.qudit_dimensions,
            num_qubits=self.num_qubits,
            num_clbits=self.num_clbits,
            params=self.params,
        )

    def repeat(self, n):
        """Creates an QuditInstruction with `gate` repeated `n` amount of times.

        Args:
            n (int): Number of times to repeat the instruction

        Returns:
            QuditInstruction: Containing the definition.

        Raises:
            CircuitError: If n < 1.
        """
        if int(n) != n or n < 1:
            raise CircuitError("Repeat can only be called with strictly positive integer.")

        n = int(n)

        instruction = self._return_repeat(n)
        qdargs = [] if not self.qudit_dimensions \
            else QuditRegister(self.qudit_dimensions, "qd")
        qargs = [] if self.num_qubits == 0 else QuantumRegister(self.num_qubits, "q")
        cargs = [] if self.num_clbits == 0 else ClassicalRegister(self.num_clbits, "c")

        from .quditcircuit import QuditCircuit  # pylint: disable=cyclic-import

        qdc = QuditCircuit()
        if qdargs:
            qdc.add_register(qdargs)
        if qargs:
            qdc.add_register(qargs)
        if cargs:
            qdc.add_register(cargs)

        # imaginary index to access qudits instead of qubits in QuditRegister
        qdc.data = [(self, qdargs[0j:], qargs[:], cargs[:])] * n
        instruction.definition = qdc
        return instruction
