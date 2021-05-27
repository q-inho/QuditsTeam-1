# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of reset.py from the original Qiskit-Terra code.
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
this can not be properly visualized in terms of qudits.
"""
from qiskit.circuit.reset import Reset, reset

from .quditcircuit import QuditCircuit
from .flexiblequditinstruction import FlexibleQuditInstruction, flex_qd_broadcast_arguments
from .quditregister import Qudit, QuditRegister


class QuditReset(FlexibleQuditInstruction):
    """Qudit reset."""

    num_qudits = 1

    def __init__(self, qudit_dimension):
        """Create new qudit reset instruction."""
        super().__init__("qudit reset", [qudit_dimension], 0, 0, [])

    def _define(self):
        """Reset each underlying qubit."""
        qd = QuditRegister(self.qudit_dimensions[0], 'qd')
        qdc = QuditCircuit(qd, name=self.name)
        rules = [
            (Reset(), [qd[:]], [])
        ]
        for instr, qargs, cargs in rules:
            qdc._append(instr, qargs, cargs)

        self.definition = qdc

    def qd_broadcast_arguments(self, qdargs, qargs, cargs):
        for qdarg in qdargs[0]:
            yield [qdarg], [], [], []


def qd_reset(self, qdargs):
    """Reset a qudit or a qudit. Qudits are reset by resetting all underlying qubits"""
    if isinstance(qdargs, (Qudit, QuditRegister)) or \
            isinstance(qdargs, (list, tuple)) and all(isinstance(qdarg, Qudit) for qdarg in qdargs):

        for qdargs, qargs, cargs in flex_qd_broadcast_arguments(self, QuditReset, qdargs):
            qudit_dimensions = [qdarg.dimension for qdarg in qdargs]
            self.append(QuditReset(qudit_dimensions), qdargs, qargs, cargs)

    # in case qdargs are qubits
    return reset(self, qdargs)


QuditCircuit.reset = qd_reset
QuditCircuit.qd_reset = qd_reset
