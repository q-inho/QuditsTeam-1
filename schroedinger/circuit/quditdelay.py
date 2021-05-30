# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of barrier.py from the original Qiskit-Terra code.
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
Delay instruction on qudits (for quditcircuit module).
"""

import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Delay

from .quditcircuit import QuditCircuit
from .flexiblequditinstruction import FlexibleQuditInstruction
from .quditregister import QuditRegister


class QuditDelay(FlexibleQuditInstruction):
    """Do nothing and just delay/wait/idle for a specified duration."""

    num_qudits = 1

    def __init__(self, qudit_dimensions, duration, unit='dt'):
        """Create new delay instruction for qudits."""
        if not isinstance(duration, (float, int)):
            raise CircuitError('Unsupported duration type.')

        if unit == 'dt' and not isinstance(duration, int):
            raise CircuitError("Integer duration is required for 'dt' unit.")

        if unit not in {'s', 'ms', 'us', 'ns', 'ps', 'dt'}:
            raise CircuitError('Unknown unit %s is specified.' % unit)

        super().__init__("delay", qudit_dimensions, 0, 0, params=[duration], unit=unit)

    def _define(self):
        """Relay delay to each underlying qubit."""
        qd = QuditRegister(self.qudit_dimensions, 'qd')
        qdc = QuditCircuit(qd, name=self.name)
        rules = [
            (Delay(self.params[0], self.unit), [qd[:]], [])
        ]
        for inst, qargs, cargs in rules:
            qdc._append(inst, qargs, cargs)

        self.definition = qdc

    def c_if(self, classical, val):
        raise CircuitError('Conditional Delay is not yet implemented.')

    @property
    def duration(self):
        """Get the duration of this delay."""
        return self.params[0]

    @duration.setter
    def duration(self, duration):
        """Set the duration of this delay."""
        self.params = [duration]
        self._define()

    def __array__(self, dtype=None):
        """Return the identity matrix."""
        return np.identity(QuditRegister(self.qudit_dimensions).size, dtype=dtype)

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the unitary matrix. This has been
        added to enable simulation without making delay a full Gate type.

        Returns:
            np.ndarray: matrix representation.
        """
        return self.__array__(dtype=complex)

    def __repr__(self):
        """Return the official string representing the delay."""
        return "%s(duration=%s[unit=%s])" % \
               (self.__class__.__name__, self.params[0], self.unit)
