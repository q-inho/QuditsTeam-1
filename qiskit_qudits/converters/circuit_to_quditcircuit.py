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


"""Helper function for converting a circuit to a qudit circuit."""

from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumcircuit import QuantumCircuit


def circuit_to_quditcircuit(circuit):
    """Convert a quantum circuit to a qudit quantum circuit.

    Args:
        circuit (QuantumCircuit): quantum circuit to convert

    Returns:
        qd_circuit (QuditCircuit): qudit quantum circuit

    Raises:
        CircuitError: If `circuit` is not a quantum circuit.
    """
    # pylint: disable=cyclic-import
    from qiskit_qudits.circuit.quditcircuit import QuditCircuit

    if isinstance(circuit, QuditCircuit):
        return circuit

    if not isinstance(circuit, QuantumCircuit):
        raise QiskitError("Only a QuantumCircuit can be converted to a QuditCircuit.")

    qd_circuit = QuditCircuit(
        circuit.qubits,
        circuit.clbits,
        *circuit.qregs,
        *circuit.cregs,
        name=circuit.name,
        global_phase=circuit.global_phase,
        metadata=circuit.metadata
    )
    qd_circuit.duration = circuit.duration
    qd_circuit.unit = circuit.unit
    qd_circuit.data = circuit.data
    return qd_circuit
