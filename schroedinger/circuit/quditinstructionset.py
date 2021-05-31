# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of instructionset.py from the original Qiskit-Terra code.
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
QuditInstruction collection.
"""

from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.instructionset import InstructionSet

from .quditinstruction import QuditInstruction
from .quditcircuitdata import QuditCircuitData


class QuditInstructionSet(InstructionSet):
    """QuditInstruction collection, and their contexts."""

    def __init__(self):
        """New collection of instructions acting on qudits.

        The context (qdargs, qargs and cargs that each instruction is attached to)
        is also stored separately for each instruction.
        """
        self.qdargs = []
        super().__init__()

    def add(self, instruction, qargs, cargs):
        """Add an instruction and its context (where it is attached).
        Appends an empty list to qdargs for each added instruction.

        Args:
            instruction (Instruction): Any Instruction instance.
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.
        """
        super().add(instruction, qargs, cargs)
        self.qdargs.append([])

    def qd_add(self, qudit_instruction, qdargs, qargs, cargs):
        """Add a qudit instruction and its context (where it is attached).

        Args:
            qudit_instruction (QuditInstruction): Any QuditInstruction instance.
            qdargs (List): List of d-dimensional quantum bit arguments.
            qargs (List): List of quantum bit arguments.
            cargs (List): List of classical bit arguments.
        Raises:
            CircuitError: If the instruction is not a QuditInstruction.
        """
        if not isinstance(qudit_instruction, QuditInstruction):
            raise CircuitError
        super().add(qudit_instruction, qargs, cargs)
        self.qdargs.append(qdargs)
