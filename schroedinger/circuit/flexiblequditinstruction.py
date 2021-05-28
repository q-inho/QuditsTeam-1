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

The method flex_qd_broadcast_arguments should be used like the following example,
with <instruction_name> replaced by the (abbreviated) instruction name and
<instruction_class> replaced by the subclass. See gates/zd.py for an example.
# ----------------------------------------------------------------------------------------
def <instruction_name>(self, qdargs, qargs, cargs):
    for qdargs, qargs, cargs in \
            flex_qd_broadcast_arguments(self, <instruction_class>, qdargs, qargs, cargs):
        qudit_dimensions = [qdarg.dimension for qdarg in qdargs]
        self.append(<instruction_class>(qudit_dimensions), qdargs, qargs, cargs)
# ----------------------------------------------------------------------------------------
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

        if not isinstance(FlexibleQuditInstruction.num_qudits, int) or \
                len(qudit_dimensions) != FlexibleQuditInstruction.num_qudits:
            raise CircuitError(
                f"Number of flexible qudits ({FlexibleQuditInstruction.num_qudits})"
                f" does not match qudit_dimensions {self.qudit_dimensions}"
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

        if not isinstance(FlexibleQuditGate.num_qudits, int) or \
                len(qudit_dimensions) != FlexibleQuditGate.num_qudits:
            raise CircuitError(
                f"Number of flexible qudits ({FlexibleQuditGate.num_qudits})"
                f" does not match qudit_dimensions {self.qudit_dimensions}"
            )
        super().__init__(
            name=name,
            qudit_dimensions=qudit_dimensions,
            num_single_qubits=num_single_qubits,
            params=params,
            label=label
            )


def flex_qd_broadcast_arguments(circuit, instclass, qdargs=None, qargs=None, cargs=None):
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
        instclass (type(FlexibleQuditInstruction) or type(FlexibleQuditGate)): type of instruction
        qdargs (Object): Representation of d-dimensional quantum bit arguments.
        qargs (Object): Representation of quantum bit arguments.
        cargs (Object): Representation of classical bit arguments.

    Raises:
        CircuitError: If the broadcast factor is not an integer greater than one.
        CircuitError: If the broadcast factor is ambiguous.
    """

    qdargs = qdargs if qdargs else []
    qargs = qargs if qargs else []
    cargs = cargs if cargs else []

    if not issubclass(instclass, (FlexibleQuditInstruction, FlexibleQuditGate)) or \
            not isinstance(instclass.num_qudits, int) or instclass.num_qudits < 1:
        raise CircuitError(
            "Only flexible instructions with num_qudits > 0 can be pre-broadcast."
        )

    qdargs = circuit.qdit_argument_conversion(qdargs)
    qargs = circuit.qbit_argument_conversion(qargs)
    cargs = circuit.cbit_argument_conversion(cargs)

    broadcast_factor = len(qdargs) / instclass.num_qudits

    if int(broadcast_factor) != broadcast_factor or broadcast_factor < 1:
        raise CircuitError("Invalid broadcast factor.")

    # temporary instruction to look up expected num_single_qubits and num_clbits
    temp_inst = instclass(
        [qd.dimension for qd in qdargs[:instclass.num_qudits]]
    )
    if len(qargs) * broadcast_factor != temp_inst.num_single_qubits or \
            len(cargs) * broadcast_factor != temp_inst.num_clbits:
        raise CircuitError("Broadcast factor is ambiguous.")

    qdit = zip(*([iter(qdargs)] * instclass.num_qudits))
    qit = zip(*([iter(qargs)] * temp_inst.num_single_qubits))
    cit = zip(*([iter(cargs)] * temp_inst.num_clbits))
    for qds, qs, cs in zip(qdit, qit, cit):
        yield list(qds), list(qs), list(cs)
