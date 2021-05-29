# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumcircuit.py from the original Qiskit-Terra code.
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

"""Quantum circuit object for qudits."""

import copy
import functools
import warnings
from typing import Union
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.qasm.qasm import Qasm
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_function, deprecate_arguments
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.parametertable import ParameterTable, ParameterView
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.circuit.register import Register
from qiskit.circuit.delay import Delay
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit

from .quditcircuitdata import QuditCircuitData
from .quditregister import QuditRegister, Qudit, AncillaQuditRegister, AncillaQudit
from .quditinstruction import QuditInstruction
from .quditinstructionset import QuditInstructionSet
from .flexiblequditinstruction import flex_qd_broadcast_arguments


class QuditCircuit(QuantumCircuit):
    """Implement a new circuit with qudits. Additionally saves QuditRegisters and Qudits.
    Each QuditRegister is also added as an QuantumRegister and each qudit also adds its qubits
    for transparency (each QuditCircuit behaves like a QuantumCircuit)."""

    prefix = "quditcircuit"

    def __init__(self, *regs, name=None, global_phase=0, metadata=None):
        """Create a new circuit capable of handling qudits.

        Args:
            regs (list(:class:`Register`|list(``int``)|``ìnt``):
                Registers to be included in the circuit.
                Parameters can be Registers or a list of qudit dimensions
                followed by the size of qubit and classical bit registers as integers.
                The for non registers the possibilities are:
                int -> QuantumRegister
                int, int -> QuantumRegister, ClassicalRegister
                list(int) -> QuditRegister
                list(int), int -> QuditRegister, ClassicalRegister
                list(int), int, int -> QuditRegister, QuantumRegister, ClassicalRegister.
            name (str): the name of the quantum circuit. If not set, an
                automatically generated string will be assigned.
            global_phase (float or ParameterExpression): The global phase of the circuit in radians.
            metadata (dict): Arbitrary key value metadata to associate with the
                circuit. This gets stored as free-form data in a dict in the
                :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will
                not be directly used in the circuit.

        Raises:
            CircuitError: If first register of regs cannot be interpreted as a QuditRegister.
        """
        # overwritten _data to change property setter and getter
        self._data = []
        # _qd_data additionally contains qudit context [(inst, qdargs, qargs, cargs), ...]
        self._qd_data = []

        # Map of qudits and qudit registers bound to this circuit, by name.
        # All registers in qdregs will also be in qregs.
        self.qdregs = []
        self._qudits = []
        self._qudit_set = set()
        self._qd_ancillas = []

        super().__init__(*regs, name=name, global_phase=global_phase, metadata=metadata)

    @property
    def data(self):
        """Return the qudit circuit data (instructions applied to qudits and context).

        Returns:
            QuditQuantumCircuitData: A list-like object containing the tuples
                for the circuit's data. Each tuple is in the format
                ``(instruction, qargs, cargs)`` (when accessed with real keys) or
                ``(instruction, qdargs, qargs, cargs)`` (when accessed with imaginary keys),
                where instruction is an Instruction (or subclass)  object, qdrgs is a list of Qudit
                objects, qargs is a list of Qubit objects and cargs is a list of Clbit objects.
        """
        return QuditCircuitData(self)

    @data.setter
    def data(self, data_input):
        """Sets the circuit data from a list of instructions and context.

        Args:
            data_input (list): A list of instructions with context
                in the format (instruction, qargs, cargs) or (instruction, qdargs, qargs, cargs),
                where instruction is an Instruction (or subclass)  object, qdrgs is a list of Qudit
                objects, qargs is a list of Qubit objects and cargs is a list of Clbit objects.
        """

        # If data_input is QuditCircuitData, use qd_data
        if isinstance(data_input, QuditCircuitData):
            data_input = QuditCircuitData[0j:]
        # If data_input is QuantumCircuitData(self), clearing self._data and self._qd_data
        # below will also empty data_input, so make a shallow copy first.
        data_input = data_input.copy()
        self._data = []
        self._qd_data = []
        self._parameter_table = ParameterTable()

        for data_tuple in data_input:
            if len(data_tuple) == 3:
                data_tuple = QuditCircuitData.convert(data_tuple)
            self.qd_append(*data_tuple)

    def __str__(self):
        return str(self.draw(output="text"))

    def __eq__(self, other):
        if not isinstance(other, QuditCircuit):
            return False
        return super().__eq__(other)

    def has_register(self, register):
        """
        Test if this circuit has the register r.

        Args:
            register (Register): a qudit or qubit or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = isinstance(register, QuditRegister) and register in self.qdregs
        return has_reg or super().has_register(register)

    def reverse_ops(self):
        """Reverse the qudit circuit by reversing the order of instructions.

        Returns:
            QuditCircuit: the reversed circuit.
        """
        reverse_circ = QuditCircuit(
            self.qudits,
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + "_reverse"
        )

        for inst, qdargs, qargs, cargs in reversed(self._qd_data):
            reverse_circ._qd_append(inst.reverse_ops(), qdargs, qargs, cargs)

        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self):
        """Return a qudit circuit with the opposite order of wires.

        Returns:
            QuditCircuit: the circuit with reversed bit order.
        """
        circ = QuditCircuit(
            reversed(self.qudits),
            reversed(self.qubits),
            reversed(self.clbits),
            *reversed(self.qregs),
            *reversed(self.cregs),
            name=self.name,
            global_phase=self.global_phase,
        )
        num_qudits = self.num_qudits
        num_qubits = self.num_qubits
        num_clbits = self.num_clbits
        old_qudits = self.qudits
        old_qubits = self.qubits
        old_clbits = self.clbits
        new_qudits = circ.qudits
        new_qubits = circ.qubits
        new_clbits = circ.clbits

        for inst, qdargs, qargs, cargs in self._qd_data:
            new_qdargs = [new_qudits[num_qudits - old_qudits.index(qd) - 1] for qd in qdargs]
            new_qargs = [new_qubits[num_qubits - old_qubits.index(q) - 1] for q in qargs]
            new_cargs = [new_clbits[num_clbits - old_clbits.index(c) - 1] for c in cargs]
            circ._qd_append(inst, new_qdargs, new_qargs, new_cargs)
        return circ

    def inverse(self):
        """Invert (take adjoint of) this qudit circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuditCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.
        """
        inverse_circ = QuditCircuit(
            self.qudits,
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + '_dg',
            global_phase=-self.global_phase
        )

        for inst, qdargs, qargs, cargs in reversed(self._qd_data):
            inverse_circ._qd_append(inst.inverse(), qdargs, qargs, cargs)
        return inverse_circ

    def repeat(self, reps):
        """Repeat this qudit circuit ``reps`` times.

        Args:
            reps (int): How often this circuit should be repeated.

        Returns:
            QuditCircuit: A circuit containing ``reps`` repetitions of this circuit.
        """
        repeated_circ = QuditCircuit(
            self.qudits,
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + "**{}".format(reps)
        )

        # currently composed is used, using to_quditgate and to_quditinstruction would be faster
        for _ in range(reps):
            repeated_circ.compose(self, self.qudits, self.qubits, self.clbits, inplace=True)

        return repeated_circ

    def power(self, power, matrix_power=False):
        """Raise this qudit circuit to the power of ``power``.
        If ``power`` is a positive integer and ``matrix_power`` is ``False``, this implementation
        defaults to calling ``repeat``. Otherwise, if the circuit is unitary, the matrix is
        computed to calculate the matrix power and will only return a QuantumCircuit,
        not a QuditCircuit."""
        return super().power(power, matrix_power=matrix_power)

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Control this circuit on ``num_ctrl_qubits`` qubits.
        This implementation will only return a QuantumCircuit, not a QuditCircuit."""
        return super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state)

    def compose(self, other, qudits=None, qubits=None, clbits=None, front=False, inplace=False):
        """Compose circuit with ``other`` circuit or instruction, optionally permuting wires.

        ``other`` can be narrower or of equal width to ``self``.

        Args:
            other (qiskit.circuit.Instruction or QuditInstruction or
            qiskit.circuit.QuantumCircuit or QuditCircuit
            or BaseOperator): (sub)circuit to compose onto self.
            qudits (list[Qudit|int]): qudits of self to compose onto.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If True, front composition will be performed (not implemented yet).
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Returns:
            QuditCircuit: the composed circuit (returns None if inplace==True).

        Raises:
            CircuitError:
            CircuitError: if composing on the front.
            CircuitError: if ``other`` is wider or there are duplicate edge mappings.
        """
        if inplace:
            dest = self
        else:
            dest = self.copy()

        if not isinstance(other, QuantumCircuit):
            if front:
                dest.data.insert(0, (other, qudits, qubits, clbits))
            else:
                dest.qd_append(other, qdargs=qudits, qargs=qubits, cargs=clbits)

            if inplace:
                return None
            return dest

        other = to_quditcircuit(other)

        if other.num_qudits > self.num_qudits or \
                other.num_qubits > self.num_qubits or other.num_clbits > self.num_clbits:
            raise CircuitError(
                "Trying to compose with another QuantumCircuit which has more 'in' edges."
            )

        bit_list = [
            ("qudits", qudits, other.qudits, self.qudits),
            ("qubits", qubits, other.qubits, self.qubits),
            ("clbits", clbits, other.clbits, self.clbits)
        ]
        edge_map = {}

        for bit_name, arg_bits, other_bits, self_bits in bit_list:
            # number of qudits, qubits and clbits must match number in circuit or None
            identity_bit_map = dict(zip(other_bits, self_bits))

            if arg_bits is None:
                bit_map = identity_bit_map
            elif len(arg_bits) != len(other_bits):
                raise CircuitError(
                    f"Number of items in {bit_name} parameter does not "
                    f"match number of {bit_name} in the circuit."
                )
            else:
                bit_map = {
                    other_bits[i]: (self_bits[q] if isinstance(q, int) else q)
                    for i, q in enumerate(arg_bits)
                } or identity_bit_map

            edge_map.update(**bit_map)

        if isinstance(other, QuditCircuit):
            qd_data = other.data[0j:]
        else:
            qd_data = [QuditCircuitData.convert(data_tuple) for data_tuple in other.data]

        mapped_qd_instrs = []

        for instr, qdargs, qargs, cargs in qd_data:
            n_qdargs = [edge_map[qdarg] for qdarg in qdargs]
            n_qargs = [edge_map[qarg] for qarg in qargs]
            n_cargs = [edge_map[carg] for carg in cargs]
            n_instr = instr.copy()

            if instr.condition is not None:
                from qiskit.dagcircuit import DAGCircuit  # pylint: disable=cyclic-import

                n_instr.condition = DAGCircuit._map_condition(edge_map, instr.condition, self.cregs)

            mapped_qd_instrs.append((n_instr, n_qdargs, n_qargs, n_cargs))

        mapped_instrs = [QuditCircuitData.convert(data_tuple) for data_tuple in mapped_qd_instrs]

        if front:
            dest._data = mapped_instrs + dest._data
            dest._qd_data = mapped_qd_instrs + dest._qd_data
        else:
            dest._data += mapped_instrs
            dest._qd_data += mapped_qd_instrs

        if front:
            dest._parameter_table.clear()
            for instr, _, _, _ in dest._qd_data:
                dest._update_parameter_table(instr)
        else:
            # just append new parameters
            for instr, _, _, _ in mapped_qd_instrs:
                dest._update_parameter_table(instr)

        for gate, cals in other.calibrations.items():
            dest._calibrations[gate].update(cals)

        dest.global_phase += other.global_phase

        if inplace:
            return None

        return dest

    def tensor(self, other, inplace=False):
        """Tensor ``self`` with ``other``.

        Returns:
            QuditCircuit: The tensored circuit (returns None if inplace==True).
        """
        other = to_quditcircuit(other)

        qudit_dimensions = self.qudit_dimensions + other.qudit_dimensions
        num_qudits = self.num_qudits + other.num_qudits
        num_single_qubits = self.num_single_qubits + other.num_single_qubits
        num_clbits = self.num_clbits + other.num_clbits

        # Check name collisions of circuits.
        # To still allow tensoring we define new registers of the correct sizes.
        if (
            (len(self.qdregs) == len(other.qdregs) == 1 and
             self.qdregs[0].name == other.qdregs[0].name == "qd") or
            (len(self.qregs) == len(other.qregs) == 1 and
             self.qregs[0].name == other.qregs[0].name == "q")
        ):
            # register sizes must be positive
            regs = []
            if qudit_dimensions:
                regs.append(qudit_dimensions)
            if num_single_qubits > 0:
                regs.append(num_single_qubits)
            if num_clbits > 0:
                regs.append(num_clbits)
            dest = QuditCircuit(*regs)

        # handle case if ``measure_all`` was called on both circuits, in which case the
        # registers are both named "meas"
        elif (
            len(self.cregs) == len(other.cregs) == 1
            and self.cregs[0].name == other.cregs[0].name == "meas"
        ):
            dest = QuditCircuit(
                other.qudits,
                self.qudits,
                other.qubits,
                self.qubits,
                other.clbits,
                self.clbits,
                *other.qregs,
                *self.qregs,
                ClassicalRegister(self.num_clbits + other.num_clbits, "meas")
            )

        # Now we don't have to handle any more cases arising from special implicit naming
        else:
            dest = QuditCircuit(
                other.qudits,
                self.qudits,
                other.qubits,
                self.qubits,
                other.clbits,
                self.clbits,
                *other.qregs,
                *self.qregs,
                *other.cregs,
                *self.cregs,
            )

        # compose self onto the output, and then other
        dest.compose(
            other,
            list(range(other.num_qudits)),
            list(range(other.num_single_qubits)),
            list(range(other.num_clbits)),
            inplace=True
        )
        dest.compose(
            self,
            list((other.num_qudits, num_qudits)),
            list((other.num_single_qubits, num_single_qubits)),
            list((other.num_clbits, num_clbits)),
            inplace=True,
        )

        # Replace information from tensored circuit into self when inplace = True
        if inplace:
            self.__dict__.update(dest.__dict__)
            return None
        return dest

    @property
    def qudits(self):
        """
        Returns a list of d-dimensional quantum bits in the order that the registers were added.
        """
        return self._qudits

    @property
    def qd_ancillas(self):
        """
        Returns a list of ancilla bits in the order that the registers were added.
        """
        return self._qd_ancillas

    # TODO: check if len(data) == len(qd_data)
    def __len__(self):
        """Return number of operations in circuit."""
        return len(self._data)

    def __getitem__(self, key):
        """Return indexed operation."""
        return self.data[key]

    @staticmethod
    def _bit_argument_conversion(bit_representation, in_array):
        """Bit argument conversion including QuditRegister -> [Qudit]"""
        if isinstance(bit_representation, QuditRegister):
            return bit_representation[0j:]
        return super()._bit_argument_conversion(bit_representation, in_array)

    def qdit_argument_conversion(self, qudit_representation):
        """
        Converts several qudit representations (such as indexes, range, etc.)
        into a list of qudits.

        Args:
            qudit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a qudit.
        """
        return QuditCircuit._bit_argument_conversion(qudit_representation, self.qudits)

    def append(self, instruction, qargs=None, cargs=None):
        """Calls qd_append with no qdargs."""
        return self.qd_append(instruction, qargs=qargs, cargs=cargs)

    def qd_append(self, instruction, qdargs=None, qargs=None, cargs=None):
        """Append one or more instructions to the end of the circuit, modifying
        the circuit in place. Expands qdargs, qargs and cargs.

        Args:
            instruction (qiskit.circuit.Instruction): Instruction instance to append
            qdargs (list(argument)): qudits to attach instruction to
            qargs (list(argument)): qubits to attach instruction to
            cargs (list(argument)): clbits to attach instruction to

        Returns:
            QuditInstruction: a handle to the instruction that was just added

        Raises:
            CircuitError: if object passed is a subclass of Instruction
            CircuitError: if object passed is neither subclass nor an instance of Instruction
        """
        # Convert input to instruction
        if not isinstance(instruction, Instruction) and not hasattr(instruction, "to_instruction"):
            if issubclass(instruction, Instruction):
                raise CircuitError(
                    "Object is a subclass of Instruction, please add () to "
                    "pass an instance of this object."
                )

            raise CircuitError(
                "Object to append must be an Instruction or " "have a to_instruction() method."
            )
        if not isinstance(instruction, Instruction) and hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

        # Make copy of parameterized gate instances
        if hasattr(instruction, "params"):
            is_parameter = any(isinstance(param, Parameter) for param in instruction.params)
            if is_parameter:
                instruction = copy.deepcopy(instruction)

        expanded_qdargs = [self.qdit_argument_conversion(qdarg) for qdarg in qdargs or []]
        expanded_qargs = [self.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]

        instructions = QuditInstructionSet()
        if isinstance(instruction, QuditInstruction):
            for qdarg, qarg, carg in instruction.qd_broadcast_arguments(
                    expanded_qdargs, expanded_qargs, expanded_cargs):
                instructions.qd_add(
                    self._qd_append(instruction, qdarg, qarg, carg), qdarg, qarg, carg
                )
        else:
            for qarg, carg in instruction.broadcast_arguments(expanded_qargs, expanded_cargs):
                instructions.add(self._append(instruction, qarg, carg), qarg, carg)
        return instructions

    def _append(self, instruction, qargs=None, cargs=None):
        """Calls _qd_append with no qdargs."""
        return self._qd_append(instruction, [], qargs, cargs)

    def _qd_append(self, instruction, qdargs, qargs, cargs):
        """Append an instruction to the end of the circuit, modifying
        the circuit in place.

        Args:
            instruction (Instruction or Operator): Instruction instance to append
            qdargs (list(tuple)): qudits to attach instruction to
            qargs (list(tuple)): qubits to attach instruction to
            cargs (list(tuple)): clbits to attach instruction to

        Returns:
            Instruction: a handle to the instruction that was just added

        Raises:
            CircuitError: If the gate is of a different shape than the wires
                it is being attached to.
            CircuitError: If qudits are part of a non-QuditInstruction context.
        """
        if not isinstance(instruction, Instruction):
            raise CircuitError("object is not an Instruction.")
        if isinstance(instruction, QuditInstruction) and qdargs:
            raise CircuitError("Qudits can only be used for a QuditInstruction.")

        # do some compatibility checks
        self._check_dups(qdargs)
        self._check_dups(qargs)
        self._check_qdargs(qdargs)
        self._check_qargs(qargs)
        self._check_cargs(cargs)

        # add the instruction onto the given wires
        instruction_context = instruction, qdargs, qargs, cargs
        self._qd_data.append(instruction_context)
        self._data.append(QuditCircuitData.convert(instruction_context))

        self._update_parameter_table(instruction)

        # mark as normal circuit if a new instruction is added
        self.duration = None
        self.unit = "dt"

        return instruction

    def add_register(self, *regs):
        """Add registers."""
        if not regs:
            return

        if any(isinstance(reg, int) for reg in regs) or \
                any(isinstance(reg, list) and all(isinstance(d, int) for d in reg) for reg in regs):
            # QuantumCircuit defined without registers / with anonymous wires
            if len(regs) == 1:
                if isinstance(regs[0], int):
                    regs = (QuantumRegister(regs[0], "q"),)
                else:
                    regs = (QuditRegister(regs[0], "qd"),)
            elif len(regs) == 2:
                if all(isinstance(reg, int) for reg in regs):
                    regs = (QuantumRegister(regs[0], "q"), ClassicalRegister(regs[1], "c"))
                else:
                    regs = (QuditRegister(regs[0], "qd"), ClassicalRegister(regs[1], "c"))
            elif len(regs) == 3:
                regs = (
                    QuditRegister(regs[0], "qd"),
                    QuantumRegister(regs[1], "q"),
                    ClassicalRegister(regs[2], "c")
                )
            else:
                raise CircuitError(
                    "QuditCircuit parameters can be Registers or a list of qudit dimensions"
                    "followed by the size of qubit and classical bit registers as integers. "
                    "The for non registers the possibilities are: "
                    "int -> QuantumRegister | "
                    "int, int -> QuantumRegister, ClassicalRegister | "
                    "list(int) -> QuditRegister | "
                    "list(int), int -> QuditRegister, ClassicalRegister | "
                    "list(int), int, int -> QuditRegister, QuantumRegister, ClassicalRegister. "
                    "QuditCircuit was called with %s." % (regs,)
                )

        for register in regs:
            if isinstance(register, Register) and any(
                register.name == reg.name for reg in self.qregs + self.cregs
            ):
                raise CircuitError('register name "%s" already exists' % register.name)

            if isinstance(register, AncillaQuditRegister):
                self._qd_ancillas.extend(register[0j:])
            if isinstance(register, QuditRegister):
                self.qdregs.append(register)
                # qubits in QuditRegister get added later in QuantumRegister check
                new_bits = [bit for bit in register[0j:] if bit not in self._qudit_set]
                self._qudits.extend(new_bits)
                self._qudit_set.update(new_bits)
            if isinstance(register, AncillaRegister):
                self._ancillas.extend(register)
            if isinstance(register, QuantumRegister):
                # QuditRegister is also a QuantumRegister
                self.qregs.append(register)
                new_bits = [bit for bit in register if bit not in self._qubit_set]
                self._qubits.extend(new_bits)
                self._qubit_set.update(new_bits)
            elif isinstance(register, ClassicalRegister):
                self.cregs.append(register)
                new_bits = [bit for bit in register if bit not in self._clbit_set]
                self._clbits.extend(new_bits)
                self._clbit_set.update(new_bits)
            elif isinstance(register, list):
                self.add_bits(register)
            else:
                raise CircuitError("expected a register")

    def add_bits(self, bits):
        """Add Bits to the circuit."""
        duplicate_bits = set(self.qudits + self.qubits + self.clbits).intersection(bits)
        if duplicate_bits:
            raise CircuitError(
                "Attempted to add bits found already in circuit: " "{}".format(duplicate_bits)
            )

        for bit in bits:
            if isinstance(bit, AncillaQudit):
                self._qd_ancillas.append(bit)
            if isinstance(bit, Qudit):
                self._qudits.append(bit)
                self._qudit_set.add(bit)
            if isinstance(bit, AncillaQubit):
                self._ancillas.append(bit)
            if isinstance(bit, Qubit):
                self._qubits.append(bit)
                self._qubit_set.add(bit)
            elif isinstance(bit, Clbit):
                self._clbits.append(bit)
                self._clbit_set.add(bit)
            else:
                raise CircuitError(
                    "Expected an instance of Qudit, Qubit, Clbit, or "
                    "AncillaQubit / AncillaQudit, but was passed {}".format(bit)
                )

    def _check_dups(self, bits):
        """Raise exception if list of bits contains duplicates."""
        sbits = set(bits)
        if len(sbits) != len(bits):
            raise CircuitError("duplicate bit arguments")

    def _check_qdargs(self, qdargs):
        """Raise exception if a qdarg is not in this circuit or bad format."""
        if not all(isinstance(i, Qudit) for i in qdargs):
            raise CircuitError("qdarg is not a Qudit")
        if not set(qdargs).issubset(self._qudit_set):
            raise CircuitError("qdargs not in this circuit")

    # TODO to_quditinstruction, to_quditgate, decompose (quditdag_to_quditcircuit)

    def draw(self, show_qudits=True, **kwargs):
        """Draw the quantum circuit.

        Args:
            show_qudits: if the drawing should be in terms of qudits or underlying qubits
            kwargs: draw options, see `quantumcircuit.draw()`

        Returns:
            :class:`TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
            :class:`str`:

            * `TextDrawing` (output='text')
                A drawing that can be printed as ascii art.
            * `matplotlib.figure.Figure` (output='mpl')
                A matplotlib figure object for the circuit diagram.
            * `PIL.Image` (output='latex')
                An in-memory representation of the image of the circuit diagram.
            * `str` (output='latex_source')
                The LaTeX source code for visualizing the circuit diagram.

        Raises:
            VisualizationError: when an invalid output method is selected
            ImportError: when the output methods requires non-installed libraries.
        """
        if show_qudits:
            # pylint: disable=cyclic-import
            # TODO qudit circuit drawer
            warnings.warn("Qudit circuit drawer not implemented yet")
            pass

        super().draw(**kwargs)

    def qd_width(self):
        """Return number of qudits plus single_qubits plus clbits in circuit.

        Returns:
            int: Qudit width of circuit.

        """
        return self.num_qudits + self.num_single_qubits + self.num_clbits

    @property
    def qudit_dimensions(self):
        """Dimensions of contained qudits as a list"""
        return [qudit.dimension for qudit in self.qudits]

    @property
    def num_qudits(self):
        """Return number of qudits."""
        return len(self.qudits)

    @property
    def num_single_qubits(self):
        """Return number of qubits not associated with qudits"""
        return self.num_qubits - sum(qudit.size for qudit in self.qudits)

    @property
    def num_qd_ancillas(self):
        """Return the number of ancilla qudits."""
        return len(self.qd_ancillas)

    def copy(self, name=None):
        """Copy the qudit circuit.

        Args:
          name (str): Name to be given to the copied circuit. If None, then the name stays the same.

        Returns:
          QuditCircuit: a deepcopy of the current circuit, with the specified name
        """
        cpy = copy.copy(self)
        # copy registers correctly, in copy.copy they are only copied via reference
        cpy.qdregs = self.qdregs.copy()
        cpy.qregs = self.qregs.copy()
        cpy.cregs = self.cregs.copy()
        cpy._qudits = self._qudits.copy()
        cpy._qubits = self._qubits.copy()
        cpy._clbits = self._clbits.copy()
        cpy._qudit_set = self._qudit_set.copy()
        cpy._qubit_set = self._qubit_set.copy()
        cpy._clbit_set = self._clbit_set.copy()

        instr_instances = {id(instr): instr for instr, _, __ in self._data}

        instr_copies = {id_: instr.copy() for id_, instr in instr_instances.items()}

        cpy._parameter_table = ParameterTable(
            {
                param: [
                    (instr_copies[id(instr)], param_index)
                    for instr, param_index in self._parameter_table[param]
                ]
                for param in self._parameter_table
            }
        )

        cpy._qd_data = [
            (instr_copies[id(inst)], qdargs.copy(), qargs.copy(), cargs.copy())
            for inst, qdargs, qargs, cargs in self._qd_data
        ]
        cpy._data = [QuditCircuitData.convert(data_tuple) for data_tuple in cpy._qd_data]

        cpy._calibrations = copy.deepcopy(self._calibrations)
        cpy._metadata = copy.deepcopy(self._metadata)

        if name:
            cpy.name = name
        return cpy

    def _create_qdreg(self, qudit_dimensions, name):
        """Creates a qdreg, checking if QuditRegister with same name exists."""
        if name in [qdreg.name for qdreg in self.qdregs]:
            save_prefix = QuditRegister.prefix
            QuditRegister.prefix = name
            new_qdreg = QuditRegister(qudit_dimensions)
            QuditRegister.prefix = save_prefix
        else:
            new_qdreg = QuantumRegister(qudit_dimensions, name)
        return new_qdreg

    def measure_active(self, inplace=True):
        """Adds measurement to all non-idle qudits and qubits. Creates a new ClassicalRegister
        with a size equal to the number of non-idle qudits and qubits being measured.

        Returns a new qudit circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new qudit circuit.

        Returns:
            QuditCircuit: Returns qudit circuit with measurements when `inplace = False`.
        """
        from qiskit.converters.circuit_to_dag import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()

        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        qudits_to_measure = [qudit for qudit in circ.qudits
                             if any(qubit in qubits_to_measure for qubit in qudit.qubits)]
        num_qubits_of_qudits_to_measure = sum(qudit.size for qudit in qudits_to_measure)

        new_creg = circ._create_creg(len(qubits_to_measure), "measure")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(circ.qudits, new_creg[:num_qubits_of_qudits_to_measure])
        circ.measure(circ.qubits, new_creg[num_qubits_of_qudits_to_measure:])
        if not inplace:
            return circ
        else:
            return None

    def measure_all(self, inplace=True):
        """Adds measurement to all qubits. Creates a new ClassicalRegister with a
        size equal to the number of qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        if inplace:
            circ = self
        else:
            circ = self.copy()

        new_creg = circ._create_creg(len(circ.qubits), "meas")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(circ.qudits, new_creg[:new_creg.size-circ.num_single_qubits])
        circ.measure(circ.qubits, new_creg[new_creg.sizecirc.num_single_qubits:])

        if not inplace:
            return circ
        else:
            return None

    # TODO
    def remove_final_measurements(self, inplace=True):
        """Removes final measurement on all qubits if they are present.
        Deletes the ClassicalRegister that was used to store the values from these measurements
        if it is idle.

        Returns a new circuit without measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements removed inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements removed when `inplace = False`.
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes import RemoveFinalMeasurements
        from qiskit.converters import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()

        dag = circuit_to_dag(circ)
        remove_final_meas = RemoveFinalMeasurements()
        new_dag = remove_final_meas.run(dag)

        # Set circ cregs and instructions to match the new DAGCircuit's
        circ.data.clear()
        circ._parameter_table.clear()
        circ.cregs = list(new_dag.cregs.values())

        for node in new_dag.topological_op_nodes():
            # Get arguments for classical condition (if any)
            inst = node.op.copy()
            circ.append(inst, node.qargs, node.cargs)

        circ.clbits.clear()

        if not inplace:
            return circ
        else:
            return None

    @staticmethod
    def from_qasm_file(path):
        """Take in a QASM file and generate a QuantumCircuit object.

        Args:
          path (str): Path to the file for a QASM program
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = Qasm(filename=path)
        return _circuit_from_qasm(qasm)

    @staticmethod
    def from_qasm_str(qasm_str):
        """Take in a QASM string and generate a QuantumCircuit object.

        Args:
          qasm_str (str): A QASM program string
        Return:
          QuantumCircuit: The QuantumCircuit object for the input QASM
        """
        qasm = Qasm(data=qasm_str)
        return _circuit_from_qasm(qasm)

    @property
    def global_phase(self):
        """Return the global phase of the circuit in radians."""
        return self._global_phase

    @global_phase.setter
    def global_phase(self, angle):
        """Set the phase of the circuit.

        Args:
            angle (float, ParameterExpression): radians
        """
        if isinstance(angle, ParameterExpression) and angle.parameters:
            self._global_phase = angle
        else:
            # Set the phase to the [0, 2π) interval
            angle = float(angle)
            if not angle:
                self._global_phase = 0
            else:
                self._global_phase = angle % (2 * np.pi)

    @property
    def parameters(self):
        """Convenience function to get the parameters defined in the parameter table."""
        # parameters from gates
        if self._parameters is None:
            unsorted = self._unsorted_parameters()
            self._parameters = sorted(unsorted, key=functools.cmp_to_key(_compare_parameters))

        # return as parameter view, which implements the set and list interface
        return ParameterView(self._parameters)

    @property
    def num_parameters(self):
        """Convenience function to get the number of parameter objects in the circuit."""
        return len(self._unsorted_parameters())

    def _unsorted_parameters(self):
        """Efficiently get all parameters in the circuit, without any sorting overhead."""
        parameters = set(self._parameter_table)
        if isinstance(self.global_phase, ParameterExpression):
            parameters.update(self.global_phase.parameters)

        return parameters

    @deprecate_arguments({"param_dict": "parameters"})
    def assign_parameters(
        self, parameters, inplace=False, param_dict=None
    ):  # pylint: disable=unused-argument
        """Assign parameters to new parameters or values.

        The keys of the parameter dictionary must be Parameter instances in the current circuit. The
        values of the dictionary can either be numeric values or new parameter objects.
        The values can be assigned to the current circuit object or to a copy of it.

        Args:
            parameters (dict or iterable): Either a dictionary or iterable specifying the new
                parameter values. If a dict, it specifies the mapping from ``current_parameter`` to
                ``new_parameter``, where ``new_parameter`` can be a new parameter object or a
                numeric value. If an iterable, the elements are assigned to the existing parameters
                in the order they were inserted. You can call ``QuantumCircuit.parameters`` to check
                this order.
            inplace (bool): If False, a copy of the circuit with the bound parameters is
                returned. If True the circuit instance itself is modified.
            param_dict (dict): Deprecated, use ``parameters`` instead.

        Raises:
            CircuitError: If parameters is a dict and contains parameters not present in the
                circuit.
            ValueError: If parameters is a list/array and the length mismatches the number of free
                parameters in the circuit.

        Returns:
            Optional(QuantumCircuit): A copy of the circuit with bound parameters, if
            ``inplace`` is True, otherwise None.

        Examples:

            Create a parameterized circuit and assign the parameters in-place.

            .. jupyter-execute::

                from qiskit.circuit import QuantumCircuit, Parameter

                circuit = QuantumCircuit(2)
                params = [Parameter('A'), Parameter('B'), Parameter('C')]
                circuit.ry(params[0], 0)
                circuit.crx(params[1], 0, 1)

                print('Original circuit:')
                print(circuit.draw())

                circuit.assign_parameters({params[0]: params[2]}, inplace=True)

                print('Assigned in-place:')
                print(circuit.draw())

            Bind the values out-of-place and get a copy of the original circuit.

            .. jupyter-execute::

                from qiskit.circuit import QuantumCircuit, ParameterVector

                circuit = QuantumCircuit(2)
                params = ParameterVector('P', 2)
                circuit.ry(params[0], 0)
                circuit.crx(params[1], 0, 1)

                bound_circuit = circuit.assign_parameters({params[0]: 1, params[1]: 2})
                print('Bound circuit:')
                print(bound_circuit.draw())

                print('The original circuit is unchanged:')
                print(circuit.draw())

        """
        # replace in self or in a copy depending on the value of in_place
        if inplace:
            bound_circuit = self
        else:
            bound_circuit = self.copy()
            self._increment_instances()
            bound_circuit._name_update()

        if isinstance(parameters, dict):
            # unroll the parameter dictionary (needed if e.g. it contains a ParameterVector)
            unrolled_param_dict = self._unroll_param_dict(parameters)
            unsorted_parameters = self._unsorted_parameters()

            # check that all param_dict items are in the _parameter_table for this circuit
            params_not_in_circuit = [
                param_key
                for param_key in unrolled_param_dict
                if param_key not in unsorted_parameters
            ]
            if len(params_not_in_circuit) > 0:
                raise CircuitError(
                    "Cannot bind parameters ({}) not present in the circuit.".format(
                        ", ".join(map(str, params_not_in_circuit))
                    )
                )

            # replace the parameters with a new Parameter ("substitute") or numeric value ("bind")
            for parameter, value in unrolled_param_dict.items():
                bound_circuit._assign_parameter(parameter, value)
        else:
            if len(parameters) != self.num_parameters:
                raise ValueError(
                    "Mismatching number of values and parameters. For partial binding "
                    "please pass a dictionary of {parameter: value} pairs."
                )
            for i, value in enumerate(parameters):
                bound_circuit._assign_parameter(self.parameters[i], value)
        return None if inplace else bound_circuit

    @deprecate_arguments({"value_dict": "values"})
    def bind_parameters(self, values, value_dict=None):  # pylint: disable=unused-argument
        """Assign numeric parameters to values yielding a new circuit.

        To assign new Parameter objects or bind the values in-place, without yielding a new
        circuit, use the :meth:`assign_parameters` method.

        Args:
            values (dict or iterable): {parameter: value, ...} or [value1, value2, ...]
            value_dict (dict): Deprecated, use ``values`` instead.

        Raises:
            CircuitError: If values is a dict and contains parameters not present in the circuit.
            TypeError: If values contains a ParameterExpression.

        Returns:
            QuantumCircuit: copy of self with assignment substitution.
        """
        if isinstance(values, dict):
            if any(isinstance(value, ParameterExpression) for value in values.values()):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)
        else:
            if any(isinstance(value, ParameterExpression) for value in values):
                raise TypeError(
                    "Found ParameterExpression in values; use assign_parameters() instead."
                )
            return self.assign_parameters(values)

    def _unroll_param_dict(self, value_dict):
        unrolled_value_dict = {}
        for (param, value) in value_dict.items():
            if isinstance(param, ParameterVector):
                if not len(param) == len(value):
                    raise CircuitError(
                        "ParameterVector {} has length {}, which "
                        "differs from value list {} of "
                        "len {}".format(param, len(param), value, len(value))
                    )
                unrolled_value_dict.update(zip(param, value))
            # pass anything else except number through. error checking is done in assign_parameter
            elif isinstance(param, (ParameterExpression, str)) or param is None:
                unrolled_value_dict[param] = value
        return unrolled_value_dict

    def _assign_parameter(self, parameter, value):
        """Update this circuit where instances of ``parameter`` are replaced by ``value``, which
        can be either a numeric value or a new parameter expression.

        Args:
            parameter (ParameterExpression): Parameter to be bound
            value (Union(ParameterExpression, float, int)): A numeric or parametric expression to
                replace instances of ``parameter``.
        """
        # parameter might be in global phase only
        if parameter in self._parameter_table.keys():
            for instr, param_index in self._parameter_table[parameter]:
                new_param = instr.params[param_index].assign(parameter, value)
                # if fully bound, validate
                if len(new_param.parameters) == 0:
                    instr.params[param_index] = instr.validate_parameter(new_param)
                else:
                    instr.params[param_index] = new_param

                self._rebind_definition(instr, parameter, value)

            if isinstance(value, ParameterExpression):
                entry = self._parameter_table.pop(parameter)
                for new_parameter in value.parameters:
                    if new_parameter in self._parameter_table:
                        self._parameter_table[new_parameter].extend(entry)
                    else:
                        self._parameter_table[new_parameter] = entry
            else:
                del self._parameter_table[parameter]  # clear evaluated expressions

        if (
            isinstance(self.global_phase, ParameterExpression)
            and parameter in self.global_phase.parameters
        ):
            self.global_phase = self.global_phase.assign(parameter, value)

        # clear parameter cache
        self._parameters = None
        self._assign_calibration_parameters(parameter, value)

    def _assign_calibration_parameters(self, parameter, value):
        """Update parameterized pulse gate calibrations, if there are any which contain
        ``parameter``. This updates the calibration mapping as well as the gate definition
        ``Schedule``s, which also may contain ``parameter``.
        """
        for cals in self.calibrations.values():
            for (qubit, cal_params), schedule in copy.copy(cals).items():
                if any(
                    isinstance(p, ParameterExpression) and parameter in p.parameters
                    for p in cal_params
                ):
                    del cals[(qubit, cal_params)]
                    new_cal_params = []
                    for p in cal_params:
                        if isinstance(p, ParameterExpression) and parameter in p.parameters:
                            new_param = p.assign(parameter, value)
                            if not new_param.parameters:
                                new_param = float(new_param)
                            new_cal_params.append(new_param)
                        else:
                            new_cal_params.append(p)
                    schedule.assign_parameters({parameter: value})
                    cals[(qubit, tuple(new_cal_params))] = schedule

    def _rebind_definition(self, instruction, parameter, value):
        if instruction._definition:
            for op, _, _ in instruction._definition:
                for idx, param in enumerate(op.params):
                    if isinstance(param, ParameterExpression) and parameter in param.parameters:
                        if isinstance(value, ParameterExpression):
                            op.params[idx] = param.subs({parameter: value})
                        else:
                            op.params[idx] = param.bind({parameter: value})
                        self._rebind_definition(op, parameter, value)

    def barrier(self, *qargs):
        """Apply :class:`~qiskit.circuit.Barrier`. If qargs is None, applies to all."""
        from qiskit.circuit.barrier import Barrier

        qubits = []

        if not qargs:  # None
            qubits.extend(self.qubits)

        for qarg in qargs:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, range):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        return self.append(Barrier(len(qubits)), qubits, [])

    def delay(self, duration, qarg=None, unit="dt"):
        """Apply :class:`~qiskit.circuit.Delay`. If qarg is None, applies to all qubits.
        When applying to multiple qubits, delays with the same duration will be created.

        Args:
            duration (int or float or ParameterExpression): duration of the delay.
            qarg (Object): qubit argument to apply this delay.
            unit (str): unit of the duration. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
                Default is ``dt``, i.e. integer time unit depending on the target backend.

        Returns:
            qiskit.Instruction: the attached delay instruction.

        Raises:
            CircuitError: if arguments have bad format.
        """
        qubits = []
        if qarg is None:  # -> apply delays to all qubits
            for q in self.qubits:
                qubits.append(q)
        else:
            if isinstance(qarg, QuantumRegister):
                qubits.extend([qarg[j] for j in range(qarg.size)])
            elif isinstance(qarg, list):
                qubits.extend(qarg)
            elif isinstance(qarg, (range, tuple)):
                qubits.extend(list(qarg))
            elif isinstance(qarg, slice):
                qubits.extend(self.qubits[qarg])
            else:
                qubits.append(qarg)

        instructions = InstructionSet()
        for q in qubits:
            inst = (Delay(duration, unit), [q], [])
            self.append(*inst)
            instructions.add(*inst)
        return instructions

    def zd(self, qdargs):
        """Apply :class:`~schroedinger.circuit.gates.ZDGate`."""
        from .gates.zd import ZDGate
        ret_data = []
        for qdargs, qargs, cargs in flex_qd_broadcast_arguments(self, ZDGate, qdargs=qdargs):
            qudit_dimensions = [qdarg.dimension for qdarg in qdargs]
            data_tuple = (ZDGate(qudit_dimensions), qdargs, qargs, cargs)
            self._qd_append(*data_tuple)
            ret_data.append(data_tuple)

        return ret_data

    def add_calibration(self, gate, qubits, schedule, params=None):
        """Register a low-level, custom pulse definition for the given gate.

        Args:
            gate (Union[Gate, str]): Gate information.
            qubits (Union[int, Tuple[int]]): List of qubits to be measured.
            schedule (Schedule): Schedule information.
            params (Optional[List[Union[float, Parameter]]]): A list of parameters.

        Raises:
            Exception: if the gate is of type string and params is None.
        """
        if isinstance(gate, Gate):
            self._calibrations[gate.name][(tuple(qubits), tuple(gate.params))] = schedule
        else:
            self._calibrations[gate][(tuple(qubits), tuple(params or []))] = schedule

    # Functions only for scheduled circuits
    def qubit_duration(self, *qubits: Union[Qubit, int]) -> Union[int, float]:
        """Return the duration between the start and stop time of the first and last instructions,
        excluding delays, over the supplied qubits. Its time unit is ``self.unit``.

        Args:
            *qubits: Qubits within ``self`` to include.

        Returns:
            Return the duration between the first start and last stop time of non-delay instructions
        """
        return self.qubit_stop_time(*qubits) - self.qubit_start_time(*qubits)

    def qubit_start_time(self, *qubits: Union[Qubit, int]) -> Union[int, float]:
        """Return the start time of the first instruction, excluding delays,
        over the supplied qubits. Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the start time of the first instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for inst, _, _ in self.data:
                if not isinstance(inst, Delay):
                    raise CircuitError(
                        "qubit_start_time undefined. " "Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        starts = {q: 0 for q in qubits}
        dones = {q: False for q in qubits}
        for inst, qargs, _ in self.data:
            for q in qubits:
                if q in qargs:
                    if isinstance(inst, Delay):
                        if not dones[q]:
                            starts[q] += inst.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return min(start for start in starts.values())

        return 0  # If there are no instructions over bits

    def qubit_stop_time(self, *qubits: Union[Qubit, int]) -> Union[int, float]:
        """Return the stop time of the last instruction, excluding delays, over the supplied qubits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self.duration is None:
            # circuit has only delays, this is kind of scheduled
            for inst, _, _ in self.data:
                if not isinstance(inst, Delay):
                    raise CircuitError(
                        "qubit_stop_time undefined. " "Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        stops = {q: self.duration for q in qubits}
        dones = {q: False for q in qubits}
        for inst, qargs, _ in reversed(self.data):
            for q in qubits:
                if q in qargs:
                    if isinstance(inst, Delay):
                        if not dones[q]:
                            stops[q] -= inst.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return max(stop for stop in stops.values())

        return 0  # If there are no instructions over bits


def _circuit_from_qasm(qasm):
    # pylint: disable=cyclic-import
    from qiskit.converters import ast_to_dag
    from qiskit.converters import dag_to_circuit

    ast = qasm.parse()
    dag = ast_to_dag(ast)
    return dag_to_circuit(dag)


def _standard_compare(value1, value2):
    if value1 < value2:
        return -1
    if value1 > value2:
        return 1
    return 0


def _compare_parameters(param1, param2):
    if isinstance(param1, ParameterVectorElement) and isinstance(param2, ParameterVectorElement):
        # if they belong to a vector with the same name, sort by index
        if param1.vector.name == param2.vector.name:
            return _standard_compare(param1.index, param2.index)

    # else sort by name
    return _standard_compare(param1.name, param2.name)


def to_quditcircuit(circuit):
    """Convert a quantum circuit to a qudit quantum circuit.

    Args:
        circuit (QuantumCircuit): quantum circuit to convert

    Returns:
        qd_circuit (QuditCircuit): qudit quantum circuit

    Raises:
        CircuitError: If `circuit` is not a quantum circuit.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise CircuitError("Only a QuantumCircuit can be converted to a QuditCircuit.")

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
