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
import warnings
from typing import Union
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.circuit.bit import Bit
from qiskit.circuit.parametertable import ParameterTable
from qiskit.circuit.register import Register
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
    for transparency (each QuditCircuit behaves like a QuantumCircuit).
    To access qudits for gates, use imaginary indices (e.g. 2j)."""

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
        # Overwritten data for new getter and setter methods
        self._data = []

        # Map of qudits and qudit registers bound to this circuit, by name.
        # All registers in qdregs will also be in qregs.
        self.qdregs = []
        self._qudits = []
        self._qudit_set = set()
        self._qd_ancillas = []

        # If qudit_dimensions is in register argument,
        # superclass will raise error "Circuit args must be Registers or integers".
        super().__init__(name=name, global_phase=global_phase, metadata=metadata)
        self.add_register(*regs)

    @staticmethod
    def to_quditcircuit(circuit):
        """Convert a quantum circuit to a qudit quantum circuit.

        Args:
            circuit (QuantumCircuit): quantum circuit to convert

        Returns:
            qd_circuit (QuditCircuit): qudit quantum circuit

        Raises:
            CircuitError: If `circuit` is not a quantum circuit.
        """
        if isinstance(circuit, QuditCircuit):
            return circuit
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

    @property
    def data(self):
        """Return the qudit circuit data (instructions and context).

        Returns:
            QuditQuantumCircuitData: A list-like object containing the tuples
                for the circuit's data. Each tuple is in the format
                ``(instruction, qargs, cargs)`` (when accessed with real keys) or
                ``(instruction, qdargs, qargs, cargs)`` (when accessed with imaginary keys),
                where instruction is an Instruction (or subclass)  object, qdargs is a list of Qudit
                objects, qargs is a list of Qubit objects and cargs is a list of Clbit objects.
        """
        return QuditCircuitData(self)

    @data.setter
    def data(self, data_input):
        """Sets the qudit circuit data from a list of instructions and context.

        Args:
            data_input (list): A list of instructions with context
                in the format ``(instruction, qargs, cargs)`` or
                 ``(instruction, qdargs, qargs, cargs)``,
                where instruction is an Instruction (or subclass)  object,
                qdargs is a list of Qudit objects, qargs is a list of
                Qubit objects and cargs is a list of Clbit objects.
        """
        # If data_input is QuantumCircuitData(self), clearing self._data
        # below will also empty data_input, so make a shallow copy first.
        qudit_circuit_data_inst = QuditCircuitData(self)
        data_input = data_input.copy()
        self._data = []
        self._parameter_table = ParameterTable()

        for data_tuple in data_input:
            if len(data_tuple) == 3:
                data_tuple = qudit_circuit_data_inst.to_qd_data(data_tuple)
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

        for inst, qargs, cargs in reversed(self.data):
            reverse_circ._append(inst.reverse_ops(), qargs, cargs)

        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self):
        """Return a qudit circuit with the opposite order of wires.

        Returns:
            QuditCircuit: the circuit with reversed bit order.
        """
        circ = QuditCircuit(
            list(qudit for qudit in reversed(self.qudits)),
            list(qubit for qubit in reversed(self.qubits)),
            list(clbit for clbit in reversed(self.clbits)),
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

        for inst, qdargs, qargs, cargs in self.data[0j:]:
            new_qdargs = [new_qudits[num_qudits - old_qudits.index(qd) - 1] for qd in qdargs]
            new_cargs = [new_clbits[num_clbits - old_clbits.index(c) - 1] for c in cargs]

            new_qargs = []
            offset = self._qubit_offset()
            for q in qargs:
                new_index = offset - old_qubits.index(q) - 1
                if new_index < 0:
                    new_index += num_qubits
                new_qargs.append(new_qubits[new_index])

            circ._append(inst, *QuditCircuitData.to_data((new_qdargs, new_qargs, new_cargs)))
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

        for inst, qargs, cargs in reversed(self.data):
            inverse_circ._append(inst.inverse(), qargs, cargs)
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
                dest.data.insert(0, *QuditCircuitData.to_data((other, qudits, qubits, clbits)))
            else:
                dest.qd_append(other, qdargs=qudits, qargs=qubits, cargs=clbits)

            if inplace:
                return None
            return dest

        other = QuditCircuit.to_quditcircuit(other)

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

            edge_map.update(bit_map)

        if isinstance(other, QuditCircuit):
            qd_data = other.data[0j:]
        else:
            qd_data = [self.data.to_qd_data(data_tuple) for data_tuple in other.data]

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

        mapped_instrs = [self.data.to_data(data_tuple) for data_tuple in mapped_qd_instrs]

        if front:
            dest._data = mapped_instrs + dest._data
        else:
            dest._data += mapped_instrs

        if front:
            dest._parameter_table.clear()
            for instr, _, _ in dest._data:
                dest._update_parameter_table(instr)
        else:
            # just append new parameters
            for instr, _, _ in mapped_instrs:
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
        other = QuditCircuit.to_quditcircuit(other)

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

    def __len__(self):
        """Return number of operations in circuit."""
        return len(self._data)

    def __getitem__(self, key):
        """Return indexed operation."""
        return self.data[key]

    def qdit_argument_conversion(self, qudit_representation):
        """
        Converts several qudit representations (such as indexes, range, etc.)
        into a list of qudits.

        Args:
            qudit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a qudit.
        """
        if isinstance(qudit_representation, QuditRegister):
            return qudit_representation[0j:]
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
            qargs (list(argument)): single qubits to attach instruction to
            cargs (list(argument)): clbits to attach instruction to

        Returns:
            QuditInstructionSet: Set of appended instructions
                (multiple due to argument broadcasting).

        Raises:
            CircuitError: if object passed is a subclass of Instruction
            CircuitError: if object passed is neither subclass nor an instance of Instruction
        """
        # Convert input to instruction
        if not isinstance(instruction, QuditInstruction) and \
                hasattr(instruction, "to_quditinstruction"):
            instruction = instruction.to_quditinstruction()
        if not isinstance(instruction, Instruction) and \
                hasattr(instruction, "to_instruction"):
            instruction = instruction.to_instruction()

        if not isinstance(instruction, Instruction):
            if issubclass(instruction, Instruction):
                raise CircuitError(
                    "Object is a subclass of Instruction, please add () to "
                    "pass an instance of this object."
                )

            raise CircuitError(
                "Object to append must be an Instruction or have a to_instruction() "
                "or to_quditinstruction() method."
            )
        if qdargs and not isinstance(instruction, QuditInstruction):
            raise CircuitError("Qudits can only be used for a QuditInstruction.")

        # Make copy of parameterized gate instances
        if hasattr(instruction, "params"):
            is_parameter = any(isinstance(param, Parameter) for param in instruction.params)
            if is_parameter:
                instruction = copy.deepcopy(instruction)

        expanded_qdargs = [self.qdit_argument_conversion(qdarg) for qdarg in qdargs or []]
        expanded_qargs = [self.qbit_argument_conversion(self._offset_qubit_representation(qarg))
                          for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]

        # All qargs must be single qubits not associated with qudits
        if any(self._split_qargs(qargs)[0] for qargs in expanded_qargs):
            raise CircuitError(
                "Individual underlying qubits of qudits cannot be part of an Instruction, "
                "only as a whole qudit. ``qargs`` contains qubits from qudits."
            )

        instructions = QuditInstructionSet()
        if isinstance(instruction, QuditInstruction):
            for qdarg, qarg, carg in instruction.qd_broadcast_arguments(
                    expanded_qdargs, expanded_qargs, expanded_cargs):
                self._qd_append(instruction, qdarg, qarg, carg)
                instructions.qd_add(instruction, qdarg, qarg, carg)
        else:
            for qarg, carg in instruction.broadcast_arguments(expanded_qargs, expanded_cargs):
                instructions.add(self._append(instruction, qarg, carg), qarg, carg)
        return instructions

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
            CircuitError: If individual qubits of qudits are part of a context.
        """
        if not isinstance(instruction, QuditInstruction):
            raise CircuitError("Qudits can only be used for a QuditInstruction.")

        # do some compatibility checks
        self._check_dups(qdargs)
        self._check_qdargs(qdargs)

        return self._append(*QuditCircuitData.to_data((instruction, qdargs, qargs, cargs)))

    def _qubit_offset(self):
        return sum(qudit.size for qudit in self.qudits)

    def _offset_qubit_representation(self, qubit_representation):
        """Offset by qubits of qudits if qubit_representation contains numbers."""
        ret = qubit_representation
        offset = self._qubit_offset()
        size = self.num_qubits

        def _offset(n):
            if n < 0:
                n += size
                if n < offset:
                    raise IndexError("Index out of range.")
            else:
                n += offset
            if n > size:
                raise IndexError("Index out of range.")
            return n

        try:
            if isinstance(QuantumCircuit.cast(qubit_representation, int), int):
                ret = _offset(qubit_representation)
            elif isinstance(qubit_representation, slice):
                ret = slice(
                    _offset(qubit_representation.start),
                    _offset(qubit_representation.stop),
                    qubit_representation.step
                )
            elif isinstance(QuantumCircuit.cast(qubit_representation, list), (range, list)):
                ret = [index if isinstance(index, Bit) else
                       _offset(index) for index in qubit_representation]
        except (IndexError, TypeError):
            # handled in QuantumCircuit._bit_argument_conversion
            pass
        return ret

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
            if isinstance(register, AncillaRegister):
                self._ancillas.extend(register)
            if isinstance(register, QuantumRegister):
                self.qregs.append(register)
                new_bits = [bit for bit in register if bit not in self._qubit_set]
                self._qubit_set.update(new_bits)
                if isinstance(register, QuditRegister):
                    self.qdregs.append(register)
                    # keep qudit qubits in front
                    qubit_offset = self._qubit_offset()
                    old_qubit_qudits = self._qubits[:qubit_offset]
                    self._qubits = old_qubit_qudits + new_bits + self._qubits[qubit_offset:]
                    # added afterwards to not affect _qubit_offset()
                    new_qudits = [bit for bit in register[0j:] if bit not in self._qudit_set]
                    self._qudits.extend(new_qudits)
                    self._qudit_set.update(new_qudits)
                else:
                    self._qubits.extend(new_bits)
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
        # filter out new qubits coming from already added qudits
        _, bits = self._split_qargs(bits)

        duplicate_bits = set(self.qudits + self.qubits + self.clbits).intersection(bits)
        if duplicate_bits:
            raise CircuitError(
                "Attempted to add bits found already in circuit: " "{}".format(duplicate_bits)
            )

        # do not add already added qubits coming from new qudits
        new_qudits = [bit for bit in bits if isinstance(bit, Qudit)]
        if len(new_qudits) > 0:
            self._qudits.extend(new_qudits)
            added_qudits, _ = self._split_qargs(self.qubits)

            for qudit in new_qudits:
                self._qudit_set.add(qudit)
                if isinstance(qudit, AncillaQudit):
                    self._qd_ancillas.append(qudit)

                if qudit not in added_qudits:
                    # keep qudit qubits in front
                    qubit_offset = self._qubit_offset()
                    old_qubit_qudits = self._qubits[:qubit_offset]
                    self._qubits = old_qubit_qudits + qudit.qubits + self._qubits[qubit_offset:]
                    self._qubit_set.update(qudit.qubits)

        for bit in bits:
            if isinstance(bit, Qudit):
                continue
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
        bit_set = set(bits)
        if len(bit_set) != len(bits):
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

        return super().draw(**kwargs)

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

        cpy._data = [(instr_copies[id(inst)], qargs.copy(), cargs.copy())
                     for inst, qargs, cargs in self._data]
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

    def _split_qargs(self, qubits):
        """Splits a list of qubits into associated qudits and single qubits.

        Args:
            qubits (list(:class:`Qubit`)): list of qubits

        Returns:
            tuple(list(:class:`Qudit`), list(:class:`Qubit`): list of qudits and single qubits

        Raise:
            CircuitError: If only part of a qudits underlying qubits are in given qubits.
        """
        qudits = [qudit for qudit in self.qudits if any(qubit in qubits for qubit in qudit.qubits)]
        qudit_qubits = [qubit for qudit in qudits for qubit in qudit.qubits]

        if any(qubit not in qubits for qubit in qudit_qubits):
            raise CircuitError("Given qubits contain incomplete part of a qudit.")

        single_qubits = [qubit for qubit in qubits if qubit not in qudit_qubits]
        return qudits, single_qubits

    def measure(self, qdargs=None, qargs=None, cargs=None):
        """Apply :class:`~.QuditMeasure` for each qudit representations in qdargs,
        apply :class:`~qiskit.circuit.Measure` for each qubit representations in qargs.

        Args:
            qdargs (object): Qudit representations to measure.
            qargs (object): Qubit representations to measure.
            cargs (Object): ClassicalBit representations to store measurement result.

        Returns:
            QuditInstructionSet: created instructions

        Raises:
            CircuitError: If the number of classical bits does not match the amount needed.
        """
        if cargs is None:
            cargs = []
        cargs = self.cbit_argument_conversion(cargs)

        if qdargs is not None:
            num_qudit_qubits = sum(qudit.size for qudit in self.qdit_argument_conversion(qdargs))
        else:
            num_qudit_qubits = 0
        if qargs is not None:
            num_single_qudits = len(self.qbit_argument_conversion(qargs))
        else:
            num_single_qudits = 0

        instructions = QuditInstructionSet()

        if len(cargs) == num_qudit_qubits + num_single_qudits:
            if len(cargs) == 0:
                return instructions
        else:
            raise CircuitError(
                f"Number of given classical bits ({len(cargs)}) does not match "
                f"amount needed ({num_qudit_qubits + num_single_qudits})."
            )

        if qdargs is not None:
            from .quditmeasure import QuditMeasure

            for qudits, qubits, clbits in flex_qd_broadcast_arguments(
                    self, QuditMeasure, qdargs=qdargs, cargs=cargs[:num_qudit_qubits]):
                qudit_dimensions = [qudit.dimension for qudit in qudits]
                inst = (QuditMeasure(qudit_dimensions), qudits, qubits, clbits)
                instructions.qd_extend(self.qd_append(*inst))

        if qargs is not None:
            from qiskit.circuit.measure import measure as _measure

            instructions.extend(_measure(self, qargs, cargs[num_qudit_qubits:]))

        return instructions

    def measure_active(self, inplace=True):
        """Adds measurement to all non-idle qudits and qubits. Creates a new ClassicalRegister
        with a size equal to the number of non-idle qudits and underlying qudits of non-idle
        qubits being measured.

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
        qudits_to_measure, single_qudits_to_measure = self._split_qargs(qubits_to_measure)

        new_creg = circ._create_creg(len(qubits_to_measure), "measure")
        circ.add_register(new_creg)
        circ.barrier(qudits_to_measure, single_qudits_to_measure)
        circ.measure(qudits_to_measure, single_qudits_to_measure, new_creg)
        if not inplace:
            return circ
        else:
            return None

    def measure_all(self, inplace=True):
        """Adds measurement to all qubits and qudits. Creates a new ClassicalRegister with
        a size equal to the number of qubits and underlying qubits of qudits being measured.

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

        new_creg = circ._create_creg(circ.num_qubits, "meas")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(circ.qudits, circ.qubits, new_creg)

        if not inplace:
            return circ
        else:
            return None

    def reset(self, qdargs=None, qargs=None):
        """Apply :class:`~.QuditReset` for each qudit representations in qdargs,
        apply :class:`~qiskit.circuit.Reset` for each qubit representations in qargs.
        If qdargs and qargs is None, applies to all qudits and qubits.

        Args:
            qdargs (object): qudit representations
            qargs (object): qubit representations

        Returns:
            QuditInstructionSet: created instructions
        """
        if qdargs is None and qargs is None:
            qdargs = self.qudits
            qargs = self.qubits[self._qubit_offset():]

        instructions = QuditInstructionSet()

        if qdargs is not None:
            from .quditreset import QuditReset

            for qudits, qubits, clbits in \
                    flex_qd_broadcast_arguments(self, QuditReset, qdargs=qdargs):

                qudit_dimensions = [qudit.dimension for qudit in qudits]
                inst = (QuditReset(qudit_dimensions), qudits, qubits, clbits)
                instructions.qd_extend(self.qd_append(*inst))

        if qargs is not None:
            from qiskit.circuit.reset import reset as _reset

            instructions.extend(_reset(self, qargs))

        return instructions

    def barrier(self, qdargs=None, qargs=None):
        """Apply :class:`~.QuditBarrier` for each qudit representations in qdargs,
        apply :class:`~qiskit.circuit.Barrier` for each qubit representations in qargs.
        If qdargs and qargs is None, applies to all qudits and qubits.

        Args:
            qdargs (object): qudit representations
            qargs (object): qubit representations

        Returns:
            QuditInstructionSet: created instructions
        """
        if qdargs is None and qargs is None:
            qdargs = self.qudits
            qargs = self.qubits[self._qubit_offset():]

        instructions = QuditInstructionSet()

        if qdargs is not None:
            from .quditbarrier import QuditBarrier

            for qudits, qubits, clbits in \
                    flex_qd_broadcast_arguments(self, QuditBarrier, qdargs=qdargs):

                qudit_dimensions = [qudit.dimension for qudit in qudits]
                inst = (QuditBarrier(qudit_dimensions), qudits, qubits, clbits)
                instructions.qd_extend(self.qd_append(*inst))

        if qargs is not None:
            print(qargs)
            try:
                instructions.extend(super().barrier(*iter(qargs)))
            except TypeError:
                instructions.extend(super().barrier(qargs))

        return instructions

    def delay(self, duration, qdargs=None, qargs=None, unit="dt"):
        """Apply :class:`~.QuditDelay` for each qudit representations in qdargs,
        apply :class:`~qiskit.circuit.Delay` for each qubit representations in qargs.
        If qdargs and qargs is None, applies to all qudits and qubits.

        Args:
            duration (int or float or ParameterExpression): Duration of the delay.
            qdargs (object): qudit representations
            qargs (object): qubit representations
            unit (str): Unit of the duration. Supported units: 's', 'ms', 'us', 'ns', 'ps', 'dt'.
                Default is ``dt``, i.e. integer time unit depending on the target backend.

        Returns:
            QuditInstructionSet: created instructions
        """
        if qdargs is None and qargs is None:
            qdargs = self.qudits
            qargs = self.qubits[self._qubit_offset():]

        instructions = QuditInstructionSet()

        if qdargs is not None:
            from .quditdelay import QuditDelay

            for qudits, qubits, clbits in \
                    flex_qd_broadcast_arguments(self, QuditDelay, qdargs=qdargs):

                qudit_dimensions = [qudit.dimension for qudit in qudits]
                inst = (
                    QuditDelay(qudit_dimensions, duration=duration, unit=unit),
                    qudits, qubits, clbits
                )
                instructions.qd_extend(self.qd_append(*inst))

        if qargs is not None:
            instructions.extend(super().delay(duration, qarg=qargs, unit=unit))

        return instructions

    def zd(self, qdargs):
        """Apply :class:`~.gates.ZDGate`."""
        from .gates.zd import ZDGate

        instructions = QuditInstructionSet()

        for qdargs, qargs, cargs in flex_qd_broadcast_arguments(self, ZDGate, qdargs=qdargs):
            qudit_dimensions = [qdarg.dimension for qdarg in qdargs]
            inst = (ZDGate(qudit_dimensions), qdargs, qargs, cargs)
            instructions.qd_extend(self.qd_append(*inst))

        return instructions

    # Functions only for scheduled circuits
    def qudit_duration(self, *qudits: Union[Qudit, int]) -> Union[int, float]:
        """Return the duration between the start and stop time of the first and last instructions,
        excluding delays, over the supplied qudits. Its time unit is ``self.unit``.

        Args:
            *qudits: Qudits within ``self`` to include.

        Returns:
            Return the duration between the first start and last stop time of non-delay instructions
        """
        return self.qudit_stop_time(*qudits) - self.qudit_start_time(*qudits)

    def qudit_start_time(self, *qudits: Union[Qubit, int]) -> Union[int, float]:
        """Return the start time of the first instruction, excluding delays,
        over the supplied qudits. Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qudits

        Args:
            *qudits: Qudits within ``self`` to include. Integers are allowed for qudits, indicating
            indices of ``self.qudits``.

        Returns:
            Return the start time of the first instruction, excluding delays, over the qudits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if not qudits:
            return 0
        min_start_time = -1
        for qudit in self.qdit_argument_conversion(qudits):
            if min_start_time < 0:
                min_start_time = self.qubit_start_time(*qudit.qubits)
            min_start_time = min(min_start_time, self.qubit_start_time(*qudit.qubits))
        return min_start_time

    def qudit_stop_time(self, *qudits: Union[Qubit, int]) -> Union[int, float]:
        """Return the stop time of the last instruction, excluding delays, over the supplied qudits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qudits

        Args:
            *qudits: Qudits within ``self`` to include. Integers are allowed for qudits, indicating
            indices of ``self.qudits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qudits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if not qudits:
            return 0
        max_stop_time = -1
        for qudit in self.qdit_argument_conversion(qudits):
            max_stop_time = max(max_stop_time, self.qubit_start_time(*qudit.qubits))
        return max_stop_time
