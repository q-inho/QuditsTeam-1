# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of gate.py from the original Qiskit-Terra code.
#
# Author: Tim Alexis Körner
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

"""Unitary gate for qudits."""

from typing import List, Optional, Union, Tuple
from itertools import product
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError

from .quditinstruction import QuditInstruction
from .quditregister import QuditRegister


# Multiple inheritance to insure that QuditGate is an instance of both QuditInstruction and Gate
class QuditGate(QuditInstruction, Gate):
    """Unitary qudit gate."""

    def __init__(self, name: str, qudit_dimensions: List[int], num_single_qubits: int,
                 params: List, label: Optional[str] = None) -> None:
        """Create a new qudit gate.

        Args:
            name: The Qobj name of the gate.
            qudit_dimensions: A list of dimensions for each qudit, in order.
            num_single_qubits: Number of single (non-qudit) qubits.
            params: A list of parameters.
            label: An optional label for the gate.
        """
        self.qudit_dimensions = qudit_dimensions
        self.num_single_qubits = num_single_qubits

        # map qudits to qubits for underlying Gate
        num_qubits = QuditRegister(qudit_dimensions).size
        num_qubits += num_single_qubits

        # Direct superclass instantiation without super() to solve multiple inheritance issue
        # (see diamond inheritance problem). QuditInstruction instantiation will call
        # superclass Instruction directly.
        QuditInstruction.__init__(self, name, qudit_dimensions, num_single_qubits, 0, params)
        Gate.__init__(self, name, num_qubits, params, label=label)

    def power(self, exponent: float):
        """Creates a unitary gate as gate^exponent.
        The current implementation will only return a Gate, not a QuditGate."""
        super().power(exponent)

    def _return_repeat(self, exponent: float) -> 'QuditGate':
        return QuditGate(
            name="%s*%s" % (self.name, exponent),
            qudit_dimensions=self.qudit_dimensions,
            num_single_qubits=self.num_single_qubits,
            params=self.params,
            label=self.label
        )

    def control(self, num_ctrl_qubits: Optional[int] = 1, label: Optional[str] = None,
                ctrl_state: Optional[Union[int, str]] = None):
        """Return controlled version of gate. See :class:`.ControlledGate` for usage.
        This implementation will only return  a Gate, not a QuditGate."""
        return super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state
        )

    def qd_broadcast_arguments(self, qdargs: List, qargs: List,
                               cargs: List) -> Tuple[List, List, List]:
        """Validation and handling of the arguments and its relationship for qudit gates.

        At first qdargs will be cast into the form
        [[qd(d=d1), qd(d=d2), ...], [qd(d=d1), qd(d=d2), ...], ...]
        for qudit_dimensions [d1, d2, ...] with each qd being a different qudit.
        Similarly qargs will be cast into the form
        [[qb, qb, ...], [qb, qb, ...], ...]
        with each sublist having length num_single_qubits and each qb being a different qubit.

        The returned tuples are the cartesian product of these forms.

        Args:
            qdargs: List of d-dimensional quantum bit arguments.
            qargs: List of quantum bit arguments.
            cargs: List of classical bit arguments.

        Returns:
            A tuple with single arguments.

        Raises:
            CircuitError: If the input is not valid. For example, the number of
                arguments does not match the gate expectation.
        """

        if all(len(qdarg) == 1 for qdarg in qdargs) and \
                [qdarg[0].dimension for qdarg in qdargs] == self.qudit_dimensions:
            qdargs = [[qdarg[0] for qdarg in qdargs]]

        elif not (len(qdargs) != 0 and
                  all([qd.d for qd in qdarg] == self.qudit_dimensions for qdarg in qdargs)):
            raise CircuitError(f"qdargs {qdargs} can not be broadcast to match"
                               f"the expected qudit dimensions {self.qudit_dimensions}")

        if all(len(qarg) == 1 for qarg in qargs) and len(qargs) == self.num_single_qubits:
            qargs = [[qarg[0] for qarg in qargs]]

        elif not (len(qargs) != 0 and all(len(qarg) == self.num_single_qubits for qarg in qargs)):
            raise CircuitError(f"qargs {qargs} can not be broadcast "
                               f"into multiples of {self.num_single_qubits}")

        if cargs:
            raise CircuitError("cargs should be empty for gates")

        for qdarg, qarg in product(qdargs, qargs):
            yield qdarg, qarg, []
