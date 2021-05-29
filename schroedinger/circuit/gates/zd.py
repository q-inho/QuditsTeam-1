# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
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

from schroedinger.circuit.quditcircuit import QuditCircuit
from schroedinger.circuit.flexiblequditinstruction import FlexibleQuditGate
from schroedinger.circuit.quditregister import QuditRegister


class ZDGate(FlexibleQuditGate):
    """General Z gate for Qudits."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, label=None):
        """Create new general Z gate for a single qudit."""
        super().__init__("ZD", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate zd()"""
        qd = QuditRegister(self.qudit_dimensions, 'qd')
        qdc = QuditCircuit(qd, name=self.name)

        w = 2*np.pi/self.qudit_dimensions[0]
        for i in range(qd.size):
            qdc.rz(w * 2**(qd.size-i+1), qd[qd.size-1-i])

        self.definition = qdc
