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
Generalized X gate for qudits, also known as X-.
"""

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister

from qiskit_qudits.circuit.quditcircuit import QuditCircuit
from qiskit_qudits.circuit.quditregister import QuditRegister
from qiskit_qudits.circuit import FlexibleQuditGate
from .qft import QFTGate, QFTdgGate
from .zd import ZDGate


class XDGate(FlexibleQuditGate):
    """General X gate for Qudits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, steps=1, label=None):
        """Create new general X gate for a single qudit."""
        self._steps = steps
        super().__init__("XD", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate xd()"""
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)

        # gates of gates are not supported, extracting subinstructions instead

        rules = []

        rules.extend(QFTGate(self.qudit_dimensions).definition[:])

        for _ in range(self._steps):
            rules.extend(ZDGate(self.qudit_dimensions).definition[:])

        rules.extend(QFTdgGate(self.qudit_dimensions).definition[:])

        for instr, _, _ in rules:
            qc._append(instr, q[:], [])
        self.definition = qc

    def inverse(self):
        """gate xddg()"""
        return XDdgGate(self.qudit_dimensions, steps=self._steps, label=self.label)


class XDdgGate(FlexibleQuditGate):
    """Adjoint of general X gate for Qudits, also known as X+."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, steps=1, label=None):
        """Create new adjoint of a general X gate for a single qudit."""
        self._steps = steps
        super().__init__("XD_dg", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate xddg()"""
        q = QuantumRegister(self.num_qubits, 'q')
        qc = QuantumCircuit(q, name=self.name)

        # gates of gates are not supported, extracting subinstructions instead

        rules = []

        rules.extend(QFTdgGate(self.qudit_dimensions).definition[:])

        for _ in range(self._steps):
            rules.extend(ZDGate(self.qudit_dimensions).definition[:])

        rules.extend(QFTGate(self.qudit_dimensions).definition[:])

        for instr, _, _ in rules:
            qc._append(instr, q[:], [])
        self.definition = qc

    def inverse(self):
        """gate xd()"""
        return XDGate(self.qudit_dimensions, steps=self._steps, label=self.label)
