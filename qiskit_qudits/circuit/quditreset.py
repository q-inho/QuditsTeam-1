# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of reset.py from the original Qiskit-Terra code.
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

"""
Qudit reset to computational zero by resetting all underlying qubits.
While qudits could be reset by simply calling reset(qdc, qudit[:]),
QuditReset can not be properly visualized in terms of qudits.
"""

from qiskit.circuit.reset import Reset
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit

from .flexiblequditinstruction import FlexibleQuditInstruction


class QuditReset(FlexibleQuditInstruction):
    """Qudit reset."""

    num_qudits = 1

    def __init__(self, qudit_dimensions):
        """Create new qudit reset instruction."""
        super().__init__("reset", qudit_dimensions, 0, 0, [])

    def _define(self):
        """Reset each underlying qubit."""
        q = QuantumRegister(self.num_qubits)
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (Reset(), q[:], [])
        ]
        for inst, qargs, cargs in rules:
            qc._append(inst, qargs, cargs)

        self.definition = qc
