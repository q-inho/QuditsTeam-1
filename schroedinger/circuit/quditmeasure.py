# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of measure.py from the original Qiskit-Terra code.
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
Qudit measurement by measureing all underlying qubits.
While qudits could be mesured by simply calling measure(qdc, qudit[:], cbits),
QuditMeasure can be properly visualized in terms of qudits.
"""

from qiskit.circuit.measure import Measure
from qiskit.circuit.classicalregister import ClassicalRegister

from .quditcircuit import QuditCircuit
from .flexiblequditinstruction import FlexibleQuditInstruction
from .quditregister import QuditRegister


class QuditMeasure(FlexibleQuditInstruction):
    """Qudit measurement by measuring all underlying Qubits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions):
        """Create new measurement instruction."""
        num_clbits = QuditRegister(qudit_dimensions).size
        super().__init__("measure", qudit_dimensions, 0, num_clbits, [])

    def _define(self):
        """Measure each underlying qubit."""
        qd = QuditRegister(self.qudit_dimensions, 'qd')
        c = ClassicalRegister(qd.size, 'c')
        qdc = QuditCircuit(qd, c, name=self.name)
        rules = [
            (Measure(), [qd[:]], [c[:]])
        ]
        for inst, qargs, cargs in rules:
            qdc._append(inst, qargs, cargs)

        self.definition = qdc
