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
Flexible versions of QuditInstruction and QuditGate.
Subclasses should only leave the qudit_dimensions argument in constructor, i.e.
def __init__(self, qudit_dimensions): ...
Since num_qudits is a class variable, it exists before instantiation.
This allows instancing instructions in response to qudit arguments.
"""

from typing import List, Optional
from itertools import islice
from qiskit.circuit.exceptions import CircuitError

from .quditinstruction import QuditInstruction
from .quditgate import QuditGate


class FlexibleQuditInstruction(QuditInstruction):
    """Qudit instruction adjusting to qudit dimensions.
    Class variable num_qudits must be set as an integer greater than 0.
    """

    # number of qudits; must later equal length of qudit_dimensions
    num_qudits = None

    def __init__(self, name, qudit_dimensions, num_single_qubits,
                 num_clbits, params, duration=None, unit='dt'):
        """
        Flexible qudit instruction. Subclasses should only leave the qudit_dimensions
        argument in constructor, i.e. def __init__(self, qudit_dimensions): ...

        Raises:
            CircuitError: If number of qudits does not equal length of qudit_dimensions.
        """

        if not isinstance(self.num_qudits, int) or len(qudit_dimensions) != self.num_qudits:
            raise CircuitError(
                f"Number of flexible qudits ({self.num_qudits})"
                f" does not match number of qudit_dimensions {qudit_dimensions}."
            )
        super().__init__(
            name=name,
            qudit_dimensions=qudit_dimensions,
            num_single_qubits=num_single_qubits,
            num_clbits=num_clbits,
            params=params,
            duration=duration,
            unit=unit
            )

    def __eq__(self, other):
        """Compares if equal ``num_qudits`` and if equal from qudit instruction perspective."""
        if isinstance(other, FlexibleQuditInstruction) and self.num_qudits != other.num_qudits:
            return False
        return super().__eq__(other)


class FlexibleQuditGate(QuditGate):
    """Qudit gate adjusting to qudit dimensions.
    Class variable num_qudits must be set as an integer greater than 0.
    """

    # number of qudits; must later equal length of qudit_dimensions
    num_qudits = None

    def __init__(self, name: str, qudit_dimensions: List[int], num_single_qubits: int,
                 params: List, label: Optional[str] = None) -> None:
        """
        Flexible qudit gate. Subclasses should only leave the qudit_dimensions
        argument in constructor, i.e. def __init__(self, qudit_dimensions): ...

        Raises:
            CircuitError: If number of qudits does not equal length of qudit_dimensions.
        """

        if not isinstance(self.num_qudits, int) or len(qudit_dimensions) != self.num_qudits:
            raise CircuitError(
                f"Number of flexible qudits ({self.num_qudits})"
                f" does not match number of qudit_dimensions {qudit_dimensions}."
            )
        super().__init__(
            name=name,
            qudit_dimensions=qudit_dimensions,
            num_single_qubits=num_single_qubits,
            params=params,
            label=label
            )

    def __eq__(self, other):
        """Compares if equal ``num_qudits`` and if equal from qudit gate perspective."""
        if isinstance(other, FlexibleQuditGate) and self.num_qudits != other.num_qudits:
            return False
        return super().__eq__(other)


def flex_qd_broadcast_arguments(circuit, instr_class, qdargs=None, qargs=None, cargs=None,
                                num_single_qubits=None, num_clbits=None):
    """
    Broadcasts qudit arguments for flexible qudit instructions before instantiating these
    instructions (in general multiple inst. are created) and appending them with their context.
    Broadcasting only works if for each slice of qdargs to the size of the expected qudit number
    ``instr_class.num_qudits`` the resulting qudit dimensions agree with the remaining qargs
    and cargs. If instr_class(qudit_dimensions) needs more qubits or clbits than are left in
    qargs / cargs broadcasting fails.

    Args:
        circuit: (QuditCircuit): circuit for the instruction
        instr_class (type(FlexibleQuditInstruction) or type(FlexibleQuditGate)): type of instruction
        qdargs (Object): Representation of d-dimensional quantum bit arguments.
        qargs (Object): Representation of quantum bit arguments.
        cargs (Object): Representation of classical bit arguments.
        num_single_qubits (int): Optional. Fix number of single qubits used by the instruction.
        num_clbits (int): Optional. Fix number of classical bits used by the instruction.

    Raises:
        CircuitError: If the broadcast factor is not an integer greater than one.
        CircuitError: If the broadcast factor is ambiguous.
    """
    if not issubclass(instr_class, (FlexibleQuditInstruction, FlexibleQuditGate)) or \
            not isinstance(instr_class.num_qudits, int) or instr_class.num_qudits < 1:
        raise CircuitError(
            "Only flexible instructions with num_qudits > 0 can be pre-broadcast."
        )
    if issubclass(instr_class, FlexibleQuditGate):
        num_clbits = 0

    qdargs = qdargs if qdargs is not None else []
    qargs = qargs if qargs is not None else []
    cargs = cargs if cargs is not None else []

    qdargs = circuit.qdit_argument_conversion(qdargs)
    qargs = circuit.qbit_argument_conversion(circuit._offset_qubit_representation(qargs))
    cargs = circuit.cbit_argument_conversion(cargs)

    broadcast_factor = len(qdargs) / instr_class.num_qudits
    if broadcast_factor == 0 or int(broadcast_factor) != broadcast_factor:
        raise CircuitError(
            f"Number of qdargs {len(qdargs)} cannot be broadcast to expected "
            f"number of qudits {instr_class.num_qudits}."
        )

    # e.g. num_qudits = 2: [qd[0], qd[1], qd[2], ..] -> (qd[0], qd[1]), (qd[2], qd[4]), ...
    qdit = zip(*([iter(qdargs)]*instr_class.num_qudits))
    qit = iter(qargs)
    cit = iter(cargs)

    for qudits in qdit:

        qudits = list(qudits)

        # loop variables
        num_sq = num_single_qubits
        num_c = num_clbits

        if num_single_qubits is None or num_clbits is None:
            # temporary instruction to look up expected num_single_qubits and num_clbits
            temp_inst = instr_class([qd.dimension for qd in qudits])
            num_sq = temp_inst.num_single_qubits
            num_c = temp_inst.num_clbits

        qubits = [qubit for qubit in islice(qit, num_sq)]
        clbits = [qubit for qubit in islice(cit, num_c)]

        if len(qubits) != num_sq or len(clbits) != num_c:
            raise CircuitError(
                f"Not enough qudits or classical bits to broadcast with qudit dimensions "
                f"{[qdarg.dimension for qdarg in qdargs]}, given {len(qargs)} qubits "
                f"and {len(cargs)} classical bits."
            )

        yield qudits, qubits, clbits
