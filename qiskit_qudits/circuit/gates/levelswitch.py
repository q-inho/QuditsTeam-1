# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
#
# Authors: Hoang Van Do, Tim Alexis Körner
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

"""
Level switch gate for qudits, can be represented by permutation matrix with a single permutation.
"""

from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumregister import QuantumRegister, AncillaRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qiskit_qudits.circuit import FlexibleQuditGate


class LevelSwitchGate(FlexibleQuditGate):
    """Level switch gate for Qudits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, first_level=0, second_level=1, label=None):
        """Create new level switch gate for a single qudit, where ``first_level``
        and ``second_level`` are interchanged."""
        if first_level >= qudit_dimensions[0] or second_level >= qudit_dimensions[0] \
                or first_level < 0 or second_level < 0:
            raise QiskitError("Levels given do not fulfill 0 <= level < dimension.")
        super().__init__("LS", qudit_dimensions, 1, [first_level, second_level], label=label)

    def _define(self):
        """gate ls()"""
        q = QuantumRegister(self.num_qubits - 1, 'q')
        a = AncillaRegister(1, 'a')
        qc = QuantumCircuit(q, a, name=self.name)

        first_on_qubits = [i for i in range(q.size) if (self.params[0] >> i) & 1 != 1]
        second_on_qubits = [i for i in range(q.size) if (self.params[1] >> i) & 1 != 1]

        def _add_controls(on_qubits):
            nonlocal qc
            if on_qubits:
                qc.x(on_qubits)
                qc.mcx(q[:], a[0])
                qc.x(on_qubits)
            else:
                qc.mcx(q[:], a[0])

        _add_controls(first_on_qubits)
        _add_controls(second_on_qubits)

        for i in range(q.size):
            if ((self.params[0] >> i) & 1) != ((self.params[1] >> i) & 1):
                qc.cx(a[0], q[i])

        # reset ancilla
        _add_controls(first_on_qubits)
        _add_controls(second_on_qubits)

        self.definition = qc

    def inverse(self):
        """Special case. Return self."""
        return LevelSwitchGate(
            self.qudit_dimensions,
            self.params[0],
            self.params[1],
            label=self.label
        )
