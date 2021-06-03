# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of circuit_to_gate.py from the original Qiskit-Terra code.
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

"""Helper function for converting a circuit to a qudit gate"""

from qiskit.exceptions import QiskitError
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister

from qiskit_qudits.circuit.quditgate import QuditGate
from qiskit_qudits.circuit.quditregister import QuditRegister


# noinspection DuplicatedCode,DuplicatedCode,DuplicatedCode
def circuit_to_quditgate(circuit, parameter_map=None, equivalence_library=None, label=None):
    """Build a ``QuditGate`` object from a ``QuantumCircuit`` or ``QuditCircuit``.

    The gate is anonymous (not tied to a named qudit- and/or quantumregister),
    and so can be inserted into another quditcircuit. The gate will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the gate.
           If None, existing circuit parameters will also parameterize the
           Gate.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted gate will be registered.
        label (str): Optional gate label.

    Raises:
        QiskitError: if circuit is non-unitary or if
            parameter_map is not compatible with circuit

    Return:
        QuditGate: a qudit Gate equivalent to the action of the
        input circuit.
    """
    # pylint: disable=cyclic-import
    from qiskit_qudits.circuit.quditcircuit import QuditCircuit

    if circuit.clbits:
        raise QiskitError('Circuit with classical bits cannot be converted to gate.')

    for inst, _, _ in circuit.data:
        if not isinstance(inst, Gate):
            raise QiskitError(
                f"One or more instructions cannot be converted  a gate. "
                f"'{inst.name}' is not a gate instruction"
            )

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(('parameter_map should map all circuit parameters. '
                           'Circuit parameters: {}, parameter_map: {}').format(
                               circuit.parameters, parameter_dict))

    if isinstance(circuit, QuditCircuit):
        gate = QuditGate(
            name=circuit.name,
            qudit_dimensions=circuit.qudit_dimensions,
            num_single_qubits=circuit.num_single_qubits,
            params=[*parameter_dict.values()],
            label=label
        )
    else:
        gate = QuditGate(
            name=circuit.name,
            qudit_dimensions=[],
            num_single_qubits=circuit.num_qubits,
            params=[*parameter_dict.values()],
            label=label
        )

    gate.condition = None

    target = circuit.assign_parameters(parameter_dict, inplace=False)

    if equivalence_library is not None:
        equivalence_library.add_equivalence(gate, target)

    rules = target.data

    new_qubits = []
    regs = []
    if gate.qudit_dimensions:
        qd = QuditRegister(gate.qudit_dimensions, 'qd')
        regs.append(qd)
        new_qubits.extend(qd[:])

    if gate.num_single_qubits > 0:
        q = QuantumRegister(gate.num_single_qubits, 'q')
        regs.append(q)
        new_qubits.extend(q[:])

    qubit_map = {qubit: new_qubits[idx] for idx, qubit in enumerate(circuit.qubits)}

    rules = [(inst, [qubit_map[y] for y in qargs], []) for inst, qargs, _ in rules]
    qdc = QuditCircuit(*regs, name=gate.name, global_phase=target.global_phase)
    for instr, qargs, cargs in rules:
        qdc._append(instr, qargs, cargs)
    gate.definition = qdc
    return gate
