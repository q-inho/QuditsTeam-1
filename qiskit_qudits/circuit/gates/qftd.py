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
QFT gate for qudits. Currently only powers of 2 are supported for qudit dimension.
Uses ``qiskit.circuit.library.basis_change.qft``.
"""

import numpy as np
from qiskit.circuit.library.basis_change.qft import QFT

from qiskit_qudits.circuit import FlexibleQuditGate


class QFTDGate(FlexibleQuditGate):
    """QFT gate for Qudits. Currently only powers of 2 are supported for qudit dimension."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, label=None):
        """Create new QFT gate for a single qudit.
        Currently only powers of 2 are supported for qudit dimension."""
        num_qubits = np.log2(qudit_dimensions[0])
        if int(num_qubits) != num_qubits:
            raise NotImplementedError("General QFT gate not implemented yet.")
        super().__init__("QFT", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate qftd()"""
        # for future reference for generalizations see https://arxiv.org/abs/quant-ph/0212002
        self.definition = QFT(num_qubits=self.num_qubits, name=self.name)

    def inverse(self):
        """gate qftddg()"""
        return QFTDdgGate(self.qudit_dimensions, label=self.label)


class QFTDdgGate(FlexibleQuditGate):
    """Adjoint of QFT gate for Qudits.
    Currently only powers of 2 are supported for qudit dimension."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, label=None):
        """Create new adjoint of a QFT gate for a single qudit.
        Currently only powers of 2 are supported for qudit dimension."""
        num_qubits = np.log2(qudit_dimensions[0])
        if int(num_qubits) != num_qubits:
            raise NotImplementedError("General QFT_dg gate not implemented yet.")
        super().__init__("QFT_dg", qudit_dimensions, 0, [], label=label)

    def _define(self):
        """gate qftddg()"""
        # for future reference for generalizations see https://arxiv.org/abs/quant-ph/0212002
        self.definition = QFT(num_qubits=self.num_qubits, inverse=True, name=self.name)

    def inverse(self):
        """gate qftd()"""
        return QFTDGate(self.qudit_dimensions, label=self.label)
