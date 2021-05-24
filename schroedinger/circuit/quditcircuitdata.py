# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumcircuitdata.py from the original Qiskit-Terra code.
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

"""A wrapper class for the purposes of validating modifications to
QuditCircuit.qddata while maintaining the interface of a python list."""

from qiskit.circuit.quantumcircuitdata import QuantumCircuitData
from qiskit.circuit.exceptions import CircuitError

from .quditinstruction import QuditInstruction


class QuditCircuitData(QuantumCircuitData):
    """A wrapper class for the purposes of validating modifications to
    QuditCircuit.qddata while maintaining the interface of a python list."""

    def __getitem__(self, i):
        return self._circuit._qddata[i]

    def __setitem__(self, key, value):
        instruction, qdargs, qargs, cargs = value

        if not isinstance(instruction, QuditInstruction) and \
                hasattr(instruction, "to_qdinstruction"):
            instruction = instruction.to_qdinstruction()

            if not isinstance(instruction, QuditInstruction):
                raise CircuitError("Object is not an QuditInstruction.")

        expanded_qdargs = [self._circuit.qdit_argument_conversion(qdarg) for qdarg in qdargs or []]
        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg) for carg in cargs or []]

        broadcast_args = list(instruction.broadcast_arguments(
            expanded_qdargs, expanded_qargs, expanded_cargs)
        )

        if len(broadcast_args) > 1:
            raise CircuitError(
                "QuditCircuit.qddata modification does not support argument broadcasting."
            )

        qdargs, qargs, cargs = broadcast_args[0]

        self._circuit._check_qddups(qdargs)
        self._circuit._check_dups(qargs)
        self._circuit._check_qdargs(qdargs)
        self._circuit._check_qargs(qargs)
        self._circuit._check_cargs(cargs)

        self._circuit._data[key] = (instruction, qdargs, qargs, cargs)

        self._circuit._update_parameter_table(instruction)

    def __cast(self, other):
        return other._circuit._qddata if isinstance(other, QuditCircuitData) else other
