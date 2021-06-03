# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of circuit_to_instruction.py from the original Qiskit-Terra code.
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

"""Helper function for converting a circuit to a qudit instruction."""

from qiskit.exceptions import QiskitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister

from qiskit_qudits.circuit.quditinstruction import QuditInstruction
from qiskit_qudits.circuit.quditregister import QuditRegister


def circuit_to_quditinstruction(circuit, parameter_map=None, equivalence_library=None):
    """Build an ``QuditInstruction`` object from a ``QuantumCircuit`` or ``QuditCircuit``.

    The quditinstruction is anonymous (not tied to a named qudit- and/or quantumregister),
    and so can be inserted into another circuit. The instruction will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the instruction.
           If None, existing circuit parameters will also parameterize the
           instruction.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted instruction will be registered.

    Raises:
        QiskitError: if parameter_map is not compatible with circuit

    Return:
        QuditInstruction: a qudit instruction equivalent to the action of the
        input circuit.
    """
    # pylint: disable=cyclic-import
    from qiskit_qudits.circuit.quditcircuit import QuditCircuit

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(('parameter_map should map all circuit parameters. '
                           'Circuit parameters: {}, parameter_map: {}').format(
                               circuit.parameters, parameter_dict))

    if isinstance(circuit, QuditCircuit):
        instruction = QuditInstruction(
            name=circuit.name,
            qudit_dimensions=circuit.qudit_dimensions,
            num_single_qubits=circuit.num_single_qubits,
            num_clbits=circuit.num_clbits,
            params=[*parameter_dict.values()]
        )
    else:
        instruction = QuditInstruction(
            name=circuit.name,
            qudit_dimensions=[],
            num_single_qubits=circuit.num_qubits,
            num_clbits=circuit.num_clbits,
            params=[*parameter_dict.values()]
        )
    instruction.condition = None

    target = circuit.assign_parameters(parameter_dict, inplace=False)

    if equivalence_library is not None:
        equivalence_library.add_equivalence(instruction, target)

    definition = target.data

    regs = []
    new_qubits = []
    if instruction.qudit_dimensions:
        qd = QuditRegister(instruction.qudit_dimensions, 'qd')
        regs.append(qd)
        new_qubits.extend(qd[:])

    if instruction.num_single_qubits > 0:
        q = QuantumRegister(instruction.num_single_qubits, 'q')
        regs.append(q)
        new_qubits.extend(q[:])

    c = []
    if instruction.num_clbits > 0:
        c = ClassicalRegister(instruction.num_clbits, 'c')
        regs.append(c)

    qubit_map = {bit: new_qubits[idx] for idx, bit in enumerate(circuit.qubits)}
    clbit_map = {bit: c[idx] for idx, bit in enumerate(circuit.clbits)}

    definition = [
        (inst, [qubit_map[y] for y in qargs], [clbit_map[y] for y in cargs])
        for inst, qargs, cargs in definition
    ]

    for rule in definition:
        condition = rule[0].condition
        if condition:
            reg, val = condition
            if reg.size == c.size:
                rule[0].condition = (c, val)
            else:
                raise QiskitError('Cannot convert condition in circuit with '
                                  'multiple classical registers to instruction')

    qdc = QuditCircuit(*regs, name=instruction.name)
    for instr, qargs, cargs in definition:
        qdc._append(instr, qargs, cargs)
    if circuit.global_phase:
        qdc.global_phase = circuit.global_phase

    instruction.definition = qdc

    return instruction
