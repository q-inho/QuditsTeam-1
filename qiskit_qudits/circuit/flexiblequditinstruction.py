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
Flexible versions of QuditInstruction and QuditGate.
Subclasses should only leave the qudit_dimensions argument in constructor, i.e.
def __init__(self, qudit_dimensions): ...
Since num_qudits is a class variable, it exists before instantiation.
This allows instancing instructions in response to qudit arguments.
"""

from typing import List, Optional
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


def flex_qd_broadcast_arguments(circuit, instr_class, qdargs=None, qargs=None, cargs=None,
                                num_single_qubits=None, num_clbits=None):
    """
    Broadcasts qudit arguments for flexible qudit instructions before instantiating these
    instructions (in general multiple inst. are created) and appending them with their context.
    Broadcasting only works if the arguments are integer multiples of the expected arguments.

    E.g. broadcast_factor = 2, num_qudits = 2, num_single_qubits = 3, num_clbits = 0:
    [qd[0], qd[1], qd[2], qd[3]], [q[0], q[1], q[2], q[3], q[4], q[5]], []
    -> [qd[0], qd[1]], [q[0], q[1], q[2]], []
       [qd[2], qd[3]], [q[3], q[4], q[5]], []

    Args:
        circuit: (QuditCircuit): circuit for the instruction
        instr_class (type(FlexibleQuditInstruction) or type(FlexibleQuditGate)): type of instruction
        qdargs (Object): Representation of d-dimensional quantum bit arguments.
        qargs (Object): Representation of quantum bit arguments.
        cargs (Object): Representation of classical bit arguments.
        num_single_qubits (int): Optional. number of single qubits used by the gate.
        num_clbits (int): Optional. Number of single classical bits used by the gate.

    Raises:
        CircuitError: If the broadcast factor is not an integer greater than one.
        CircuitError: If the broadcast factor is ambiguous.
    """
    qdargs = qdargs if qdargs is not None else []
    qargs = qargs if qargs is not None else []
    cargs = cargs if cargs is not None else []

    if not issubclass(instr_class, (FlexibleQuditInstruction, FlexibleQuditGate)) or \
            not isinstance(instr_class.num_qudits, int) or instr_class.num_qudits < 1:
        raise CircuitError(
            "Only flexible instructions with num_qudits > 0 can be pre-broadcast."
        )

    qdargs = circuit.qdit_argument_conversion(qdargs)
    qargs = circuit.qbit_argument_conversion(qargs)
    cargs = circuit.cbit_argument_conversion(cargs)
    broadcast_factor = len(qdargs) / instr_class.num_qudits

    if int(broadcast_factor) != broadcast_factor or broadcast_factor < 1:
        raise CircuitError("Invalid broadcast factor.")

    if issubclass(instr_class, FlexibleQuditGate):
        num_clbits = 0

    if num_single_qubits is None or num_clbits is None:
        # temporary instruction to look up expected num_single_qubits and num_clbits
        temp_inst = instr_class(
            [qd.dimension for qd in qdargs[:instr_class.num_qudits]]
        )
        num_single_qubits = temp_inst.num_single_qubits
        num_clbits = temp_inst.num_clbits

    if len(qargs) * broadcast_factor != num_single_qubits or \
            len(cargs) * broadcast_factor != num_clbits:
        raise CircuitError("Broadcast factor is ambiguous, check number of arguments.")

    padding = [()] * int(broadcast_factor)
    qdit = zip(*([iter(qdargs)] * instr_class.num_qudits)) if instr_class.num_qudits else padding
    qit = zip(*([iter(qargs)] * num_single_qubits)) if num_single_qubits else padding
    cit = zip(*([iter(cargs)] * num_clbits)) if num_clbits else padding
    for qds, qs, cs in zip(qdit, qit, cit):
        yield list(qds), list(qs), list(cs)
