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

from qiskit_qudits.circuit.quditregister import QuditRegister
from qiskit_qudits.circuit.quditcircuit import QuditCircuit
from qiskit_qudits.circuit import FlexibleQuditGate
from .qftd import QFTDGate, QFTDdgGate
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
        qd = QuditRegister(self.num_qubits, 'qd')
        qdc = QuditCircuit(qd, name=self.name)

        qd_rules = [
            (QFTDGate(self.qudit_dimensions), [qd[0j]], [], [])
        ]
        for _ in range(self._steps):
            qd_rules.append(
                (ZDGate(self.qudit_dimensions), [qd[0j]], [], [])
            )
            qdc.zd(0)
        qd_rules.append(
            (QFTDdgGate(self.qudit_dimensions), [qd[0j]], [], [])
        )

        for instr, qdargs, qargs, cargs in qd_rules:
            qdc._qd_append(instr, qargs, qargs, cargs)
        self.definition = qdc

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
        qd = QuditRegister(self.num_qubits, 'qd')
        qdc = QuditCircuit(qd, name=self.name)

        qd_rules = [
            (QFTDdgGate(self.qudit_dimensions), [qd[0j]], [], [])
        ]
        for _ in range(self._steps):
            qd_rules.append(
                (ZDGate(self.qudit_dimensions), [qd[0j]], [], [])
            )
            qdc.zd(0)
        qd_rules.append(
            (QFTDGate(self.qudit_dimensions), [qd[0j]], [], [])
        )

        for instr, qdargs, qargs, cargs in qd_rules:
            qdc._qd_append(instr, qargs, qargs, cargs)
        self.definition = qdc

    def inverse(self):
        """gate xd()"""
        return XDGate(self.qudit_dimensions, steps=self._steps, label=self.label)
