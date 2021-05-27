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
from qiskit.circuit.exceptions import CircuitError

from .quditcircuit import QuditCircuit
from .quditinstruction import QuditInstruction
from .quditregister import Qudit, QuditRegister


class QuditReset(QuditInstruction):
    """Qudit reset."""

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
            yield [qdarg], [], []


def qd_reset(self, qdargs):
    """Reset a qudit or a qudit. Qudits are reset by resetting all underlying qubits"""
    if isinstance(qdargs, Qudit):
        return self.append(QuditReset(qdargs.dimension), [[qdargs]], [], [])
    if isinstance(qdargs, QuditRegister):
        qdargs = qdargs[0j:]
    if isinstance(qdargs, (list, tuple)) and all(isinstance(qdarg, Qudit) for qdarg in qdargs):
        if any(qdarg.dimension != qdargs[0].dimension for qdarg in qdargs):
            raise CircuitError(
                "Resetting a group of qudits requires all qudits to have the same dimension"
            )
        return self.append(QuditReset(qdargs[0].dimension), [qdargs], [], [])

    # in case qdargs are qubits
    return reset(self, qdargs)


QuditCircuit.reset = qd_reset
QuditCircuit.qd_reset = qd_reset
