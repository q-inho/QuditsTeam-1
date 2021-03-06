# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
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

"""
Generalized Z gate for qudits.
"""

import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates.p import PhaseGate

from qiskit_qudits.circuit import FlexibleQuditGate


class ZDGate(FlexibleQuditGate):
    """General Z gate for Qudits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, label=None):
        """Create new general Z gate for a single qudit."""
        super().__init__("ZD", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate zd()"""
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)

        w = 2 * np.pi / self.qudit_dimensions[0]
        rules = []
        for i in range(q.size):
            rules.append(
                (PhaseGate(w * 2**(q.size - i - 1)), [q[q.size - i - 1]], [])
            )

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        """gate zddg()"""
        return ZDdgGate(self.qudit_dimensions, label=self.label)


class ZDdgGate(FlexibleQuditGate):
    """Adjoint of general Z gate for Qudits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, label=None):
        """Create new adjoint of a general Z gate for a single qudit."""
        super().__init__("ZD_dg", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate zddg()"""
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)

        w = -2 * np.pi / self.qudit_dimensions[0]
        rules = []
        for i in range(q.size):
            rules.append(
                (PhaseGate(w * 2**(q.size - i - 1)), [q[q.size - i - 1]], [])
            )

        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self):
        """gate zd()"""
        return ZDGate(self.qudit_dimensions, label=self.label)
