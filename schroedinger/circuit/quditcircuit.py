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
import itertools
import functools
import warnings
import numbers
import multiprocessing as mp
from collections import OrderedDict, defaultdict
from typing import Union
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import is_main_process
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.qasm.qasm import Qasm
from qiskit.qasm.exceptions import QasmError
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_function, deprecate_arguments
from .parameterexpression import ParameterExpression
from .quantumregister import QuantumRegister, Qubit, AncillaRegister, AncillaQubit
from .classicalregister import ClassicalRegister, Clbit
from .parametertable import ParameterTable, ParameterView
from .parametervector import ParameterVector, ParameterVectorElement
from .instructionset import InstructionSet
from .register import Register
from .bit import Bit
from .delay import Delay
from qiskit.circuit.quantumcircuit import QuantumCircuit

from .quditcircuitdata import QuditCircuitData
from .quditregister import QuditRegister, Qudit, AncillaQuditRegister, AncillaQudit


class QuditCircuit(QuantumCircuit):
    """Implement a new circuit with qudits. Additionally saves QuditRegisters and Qudits.
    Each QuditRegister is also added as an QuantumRegister and each qudit also adds its qubits
    for transparency (each QuditCircuit behaves like a QuantumCircuit)."""

    prefix = "quditcircuit"

    def __init__(self, *regs, name=None, global_phase=0, metadata=None):
        """Create a new circuit capable of handling qudits.

        Args:
            regs (list(:class:`Register`|``ìnt``): Registers to be included in the circuit.
                If the registers are described with integers, the first integer will be
                interpreted as an QuditRegister. If only two integers are handed, the
                second one will be interpreted as an ClassicalRegister. For three integers
                the order is qudit register, qubit register, classical register.
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
            self.append(*data_tuple)

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
            *self.qdregs,
            *self.qregs,
            *self.cregs,
            name=self.name + "_reverse"
        )

        for inst, qdargs, qargs, cargs in reversed(self._qd_data):
            reverse_circ._append(inst.reverse_ops(), qdargs, qargs, cargs)

        reverse_circ.duration = self.duration
        reverse_circ.unit = self.unit
        return reverse_circ

    def reverse_bits(self):
        """Return a qudit circuit with the opposite order of wires.

        Returns:
            QuditCircuit: the circuit with reversed bit order.
        """
        circ = QuditCircuit(
            *reversed(self.qdregs),
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
            circ._append(inst, new_qdargs, new_qargs, new_cargs)
        return circ

    def inverse(self):
        """Invert (take adjoint of) this qudit circuit.

        This is done by recursively inverting all gates.

        Returns:
            QuditCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.
        """
        inverse_circ = QuantumCircuit(
            *self.qdregs,
            *self.qregs,
            *self.cregs,
            name=self.name + '_dg',
            global_phase=-self.global_phase
        )

        for inst, qdargs, qargs, cargs in reversed(self._qd_data):
            inverse_circ._append(inst.inverse(), qdargs, qargs, cargs)
        return inverse_circ

    def repeat(self, reps):
        """Repeat this qudit circuit ``reps`` times.

        Args:
            reps (int): How often this circuit should be repeated.

        Returns:
            QuditCircuit: A circuit containing ``reps`` repetitions of this circuit.
        """
        repeated_circ = QuditCircuit(
            *self.qdregs,
            *self.qregs,
            *self.cregs,
            name=self.name + "**{}".format(reps)
        )

        # benefit of appending instructions: decomposing shows the subparts, i.e. the power
        # is actually `reps` times this circuit, and it is currently much faster than `compose`.
        if reps > 0:
            try:  # try to append as gate if possible to not disallow to_gate
                inst = self.to_quditgate()
            except QiskitError:
                inst = self.to_quditinstruction()
            for _ in range(reps):
                repeated_circ._append(inst, self.qudits, self.qubits, self.clbits)

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
            other (Instruction or QuditInstruction or QuantumCircuit or QuditCircuit
            or BaseOperator): (sub)circuit to compose onto self.
            qudits (list[Qudit]|int]): qudits of self to compose onto.
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
                dest.append(other, qdargs=qudits, qargs=qubits, cargs=clbits)

            if inplace:
                return None
            return dest

        if other.num_qubits > self.num_qubits or other.num_clbits > self.num_clbits:
            raise CircuitError(
                "Trying to compose with another QuantumCircuit which has more 'in' edges."
            )

        bit_list = [("qubits", qubits, other.qubits, self.qubits),
                    ("clbits", clbits, other.clbits, self.clbits)]

        if isinstance(other, QuditCircuit):
            bit_list.append(("qudits", qudits, other.qudits, self.qudits))
        else:
            if qudits is not None:
                raise CircuitError(
                    f"Other circuit is not a Quditcircuit and can not be mapped to {qudits}."
                )

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

    #TODO
    def tensor(self, other, inplace=False):
        """Tensor ``self`` with ``other``.

        Remember that in the little-endian convention the leftmost operation will be at the bottom
        of the circuit. See also
        [the docs](qiskit.org/documentation/tutorials/circuits/3_summary_of_quantum_operations.html)
        for more information.

        .. parsed-literal::

                 ┌────────┐        ┌─────┐          ┌─────┐
            q_0: ┤ bottom ├ ⊗ q_0: ┤ top ├  = q_0: ─┤ top ├──
                 └────────┘        └─────┘         ┌┴─────┴─┐
                                              q_1: ┤ bottom ├
                                                   └────────┘

        Args:
            other (QuantumCircuit): The other circuit to tensor this circuit with.
            inplace (bool): If True, modify the object. Otherwise return composed circuit.

        Examples:

            .. jupyter-execute::

                from qiskit import QuantumCircuit
                top = QuantumCircuit(1)
                top.x(0);
                bottom = QuantumCircuit(2)
                bottom.cry(0.2, 0, 1);
                tensored = bottom.tensor(top)
                print(tensored.draw())

        Returns:
            QuantumCircuit: The tensored circuit (returns None if inplace==True).
        """
        num_qubits = self.num_qubits + other.num_qubits
        num_clbits = self.num_clbits + other.num_clbits

        # If a user defined both circuits with via register sizes and not with named registers
        # (e.g. QuantumCircuit(2, 2)) then we have a naming collision, as the registers are by
        # default called "q" resp. "c". To still allow tensoring we define new registers of the
        # correct sizes.
        if (
            len(self.qregs) == len(other.qregs) == 1
            and self.qregs[0].name == other.qregs[0].name == "q"
        ):
            # check if classical registers are in the circuit
            if num_clbits > 0:
                dest = QuantumCircuit(num_qubits, num_clbits)
            else:
                dest = QuantumCircuit(num_qubits)

        # handle case if ``measure_all`` was called on both circuits, in which case the
        # registers are both named "meas"
        elif (
            len(self.cregs) == len(other.cregs) == 1
            and self.cregs[0].name == other.cregs[0].name == "meas"
        ):
            cr = ClassicalRegister(self.num_clbits + other.num_clbits, "meas")
            dest = QuantumCircuit(*other.qregs, *self.qregs, cr)

        # Now we don't have to handle any more cases arising from special implicit naming
        else:
            dest = QuantumCircuit(
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
        dest.compose(other, range(other.num_qubits), range(other.num_clbits), inplace=True)
        dest.compose(
            self,
            range(other.num_qubits, num_qubits),
            range(other.num_clbits, num_clbits),
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

    @deprecate_function(
        "The QuantumCircuit.__add__() method is being deprecated."
        "Use the compose() method which is more flexible w.r.t "
        "circuit register compatibility."
    )
    def __add__(self, rhs):
        """Overload + to implement self.combine."""
        return self.combine(rhs)

    @deprecate_function(
        "The QuantumCircuit.__iadd__() method is being deprecated. Use the "
        "compose() (potentially with the inplace=True argument) and tensor() "
        "methods which are more flexible w.r.t circuit register compatibility."
    )
    def __iadd__(self, rhs):
        """Overload += to implement self.extend."""
        return self.extend(rhs)

    #TODO: check if len(data) == len(qd_data)
    def __len__(self):
        """Return number of operations in circuit."""
        return len(self._data)

    def __getitem__(self, key):
        """Return indexed operation."""
        return self.data[key]

    @staticmethod
    def cast(value, _type):
        """Best effort to cast value to type. Otherwise, returns the value."""
        try:
            return _type(value)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _bit_argument_conversion(bit_representation, in_array):
        ret = None
        try:
            if isinstance(bit_representation, Bit):
                # circuit.h(qr[0]) -> circuit.h([qr[0]])
                ret = [bit_representation]
            elif isinstance(bit_representation, Register):
                # circuit.h(qr) -> circuit.h([qr[0], qr[1]])
                ret = bit_representation[:]
            elif isinstance(QuantumCircuit.cast(bit_representation, int), int):
                # circuit.h(0) -> circuit.h([qr[0]])
                ret = [in_array[bit_representation]]
            elif isinstance(bit_representation, slice):
                # circuit.h(slice(0,2)) -> circuit.h([qr[0], qr[1]])
                ret = in_array[bit_representation]
            elif isinstance(bit_representation, list) and all(
                isinstance(bit, Bit) for bit in bit_representation
            ):
                # circuit.h([qr[0], qr[1]]) -> circuit.h([qr[0], qr[1]])
                ret = bit_representation
            elif isinstance(QuantumCircuit.cast(bit_representation, list), (range, list)):
                # circuit.h([0, 1])     -> circuit.h([qr[0], qr[1]])
                # circuit.h(range(0,2)) -> circuit.h([qr[0], qr[1]])
                # circuit.h([qr[0],1])  -> circuit.h([qr[0], qr[1]])
                ret = [
                    index if isinstance(index, Bit) else in_array[index]
                    for index in bit_representation
                ]
            else:
                raise CircuitError(
                    "Not able to expand a %s (%s)" % (bit_representation, type(bit_representation))
                )
        except IndexError as ex:
            raise CircuitError("Index out of range.") from ex
        except TypeError as ex:
            raise CircuitError(
                f"Type error handling {bit_representation} ({type(bit_representation)})"
            ) from ex
        return ret

    def qbit_argument_conversion(self, qubit_representation):
        """
        Converts several qubit representations (such as indexes, range, etc.)
        into a list of qubits.

        Args:
            qubit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a qubit.
        """
        return QuantumCircuit._bit_argument_conversion(qubit_representation, self.qubits)

    def cbit_argument_conversion(self, clbit_representation):
        """
        Converts several classical bit representations (such as indexes, range, etc.)
        into a list of classical bits.

        Args:
            clbit_representation (Object): representation to expand

        Returns:
            List(tuple): Where each tuple is a classical bit.
        """
        return QuantumCircuit._bit_argument_conversion(clbit_representation, self.clbits)

    def append(self, instruction, qargs=None, cargs=None):
        """Append one or more instructions to the end of the circuit, modifying
        the circuit in place. Expands qargs and cargs.

        Args:
            instruction (qiskit.circuit.Instruction): Instruction instance to append
            qargs (list(argument)): qubits to attach instruction to
            cargs (list(argument)): clbits to attach instruction to

        Returns:
            qiskit.circuit.Instruction: a handle to the instruction that was just added

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

        expanded_qargs = [self.qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self.cbit_argument_conversion(carg) for carg in cargs or []]

        instructions = InstructionSet()
        for (qarg, carg) in instruction.broadcast_arguments(expanded_qargs, expanded_cargs):
            instructions.add(self._append(instruction, qarg, carg), qarg, carg)
        return instructions

    def _append(self, instruction, qargs, cargs):
        """Append an instruction to the end of the circuit, modifying
        the circuit in place.

        Args:
            instruction (Instruction or Operator): Instruction instance to append
            qargs (list(tuple)): qubits to attach instruction to
            cargs (list(tuple)): clbits to attach instruction to

        Returns:
            Instruction: a handle to the instruction that was just added

        Raises:
            CircuitError: if the gate is of a different shape than the wires
                it is being attached to.
        """
        if not isinstance(instruction, Instruction):
            raise CircuitError("object is not an Instruction.")

        # do some compatibility checks
        self._check_dups(qargs)
        self._check_qargs(qargs)
        self._check_cargs(cargs)

        # add the instruction onto the given wires
        instruction_context = instruction, qargs, cargs
        self._data.append(instruction_context)

        self._update_parameter_table(instruction)

        # mark as normal circuit if a new instruction is added
        self.duration = None
        self.unit = "dt"

        return instruction

    def _update_parameter_table(self, instruction):

        for param_index, param in enumerate(instruction.params):
            if isinstance(param, ParameterExpression):
                current_parameters = self._parameter_table

                for parameter in param.parameters:
                    if parameter in current_parameters:
                        if not self._check_dup_param_spec(
                            self._parameter_table[parameter], instruction, param_index
                        ):
                            self._parameter_table[parameter].append((instruction, param_index))
                    else:
                        if parameter.name in self._parameter_table.get_names():
                            raise CircuitError(
                                "Name conflict on adding parameter: {}".format(parameter.name)
                            )
                        self._parameter_table[parameter] = [(instruction, param_index)]

                        # clear cache if new parameter is added
                        self._parameters = None

        return instruction

    def _check_dup_param_spec(self, parameter_spec_list, instruction, param_index):
        for spec in parameter_spec_list:
            if spec[0] is instruction and spec[1] == param_index:
                return True
        return False

    def add_register(self, *regs):
        # check if first register is a valid qudit register
        # check and ignore multiple qubits due to qudit overlap
        qdreg = regs[0]
        if not isinstance(qdreg, (list, dict, QuditRegister)):
            try:
                valid_reg_size = qdreg == int(qdreg)
            except (ValueError, TypeError):
                valid_reg_size = False

            if not valid_reg_size:
                raise CircuitError(
                    "Circuit args must be Registers or valid arguments to pass on to registers. "
                    "(%s '%s' was provided)" % ([type(reg).__name__ for reg in regs], regs)
                )

            qdreg = int(qdreg)

        # reorder classical register
        if len(regs) == 2:
            regs = [0, regs[1]]
        """Add registers."""
        if not regs:
            return

        if any(isinstance(reg, int) for reg in regs):
            # QuantumCircuit defined without registers
            if len(regs) == 1 and isinstance(regs[0], int):
                # QuantumCircuit with anonymous quantum wires e.g. QuantumCircuit(2)
                regs = (QuantumRegister(regs[0], "q"),)
            elif len(regs) == 2 and all(isinstance(reg, int) for reg in regs):
                # QuantumCircuit with anonymous wires e.g. QuantumCircuit(2, 3)
                regs = (QuantumRegister(regs[0], "q"), ClassicalRegister(regs[1], "c"))
            else:
                raise CircuitError(
                    "QuantumCircuit parameters can be Registers or Integers."
                    " If Integers, up to 2 arguments. QuantumCircuit was called"
                    " with %s." % (regs,)
                )

        for register in regs:
            if isinstance(register, Register) and any(
                register.name == reg.name for reg in self.qregs + self.cregs
            ):
                raise CircuitError('register name "%s" already exists' % register.name)

            if isinstance(register, AncillaRegister):
                self._ancillas.extend(register)

            if isinstance(register, QuantumRegister):
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
        duplicate_bits = set(self.qubits + self.clbits).intersection(bits)
        if duplicate_bits:
            raise CircuitError(
                "Attempted to add bits found already in circuit: " "{}".format(duplicate_bits)
            )

        for bit in bits:
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
                    "Expected an instance of Qubit, Clbit, or "
                    "AncillaQubit, but was passed {}".format(bit)
                )

    def _check_dups(self, bits):
        """Raise exception if list of bits contains duplicates.
            Overwrites superclass method to support qudits."""
        sbits = set(bits)
        if len(sbits) != len(bits):
            raise CircuitError("duplicate qubit or qudit arguments")

    def _check_qdargs(self, qdargs):
        """Raise exception if a qdarg is not in this circuit or bad format."""
        if not all(isinstance(i, Qudit) for i in qdargs):
            raise CircuitError("qdarg is not a Qudit")
        if not set(qdargs).issubset(self._qudit_set):
            raise CircuitError("qdargs not in this circuit")

    def to_instruction(self, parameter_map=None):
        """Create an Instruction out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.

        Returns:
            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_instruction import circuit_to_instruction

        return circuit_to_instruction(self, parameter_map)

    def to_gate(self, parameter_map=None, label=None):
        """Create a Gate out of this circuit.

        Args:
            parameter_map(dict): For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               gate. If None, existing circuit parameters will also
               parameterize the gate.
            label (str): Optional gate label.

        Returns:
            Gate: a composite gate encapsulating this circuit
            (can be decomposed back)
        """
        from qiskit.converters.circuit_to_gate import circuit_to_gate

        return circuit_to_gate(self, parameter_map, label=label)

    def decompose(self):
        """Call a decomposition pass on this circuit,
        to decompose one level (shallow decompose).

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """
        # pylint: disable=cyclic-import
        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from qiskit.converters.dag_to_circuit import dag_to_circuit

        pass_ = Decompose()
        decomposed_dag = pass_.run(circuit_to_dag(self))
        return dag_to_circuit(decomposed_dag)

    def _check_compatible_regs(self, rhs):
        """Raise exception if the circuits are defined on incompatible registers"""
        list1 = self.qregs + self.cregs
        list2 = rhs.qregs + rhs.cregs
        for element1 in list1:
            for element2 in list2:
                if element2.name == element1.name:
                    if element1 != element2:
                        raise CircuitError(
                            "circuits are not compatible:"
                            f" registers {element1} and {element2} not compatible"
                        )

    @staticmethod
    def _get_composite_circuit_qasm_from_instruction(instruction):
        """Returns OpenQASM string composite circuit given an instruction.
        The given instruction should be the result of composite_circuit.to_instruction()."""

        gate_parameters = ",".join(["param%i" % num for num in range(len(instruction.params))])
        qubit_parameters = ",".join(["q%i" % num for num in range(instruction.num_qubits)])
        composite_circuit_gates = ""

        definition = instruction.definition
        definition_bit_labels = {
            bit: idx
            for bits in (definition.qubits, definition.clbits)
            for idx, bit in enumerate(bits)
        }
        for data, qargs, _ in definition:
            gate_qargs = ",".join(
                ["q%i" % index for index in [definition_bit_labels[qubit] for qubit in qargs]]
            )
            composite_circuit_gates += "%s %s; " % (data.qasm(), gate_qargs)

        if composite_circuit_gates:
            composite_circuit_gates = composite_circuit_gates.rstrip(" ")

        if gate_parameters:
            qasm_string = "gate %s(%s) %s { %s }" % (
                instruction.name,
                gate_parameters,
                qubit_parameters,
                composite_circuit_gates,
            )
        else:
            qasm_string = "gate %s %s { %s }" % (
                instruction.name,
                qubit_parameters,
                composite_circuit_gates,
            )

        return qasm_string

    def qasm(self, formatted=False, filename=None):
        """Return OpenQASM string.

        Args:
            formatted (bool): Return formatted Qasm string.
            filename (str): Save Qasm to file with name 'filename'.

        Returns:
            str: If formatted=False.

        Raises:
            ImportError: If pygments is not installed and ``formatted`` is
                ``True``.
            QasmError: If circuit has free parameters.
        """
        from qiskit.circuit.controlledgate import ControlledGate

        if self.num_parameters > 0:
            raise QasmError("Cannot represent circuits with unbound parameters in OpenQASM 2.")

        existing_gate_names = [
            "ch",
            "cp",
            "cx",
            "cy",
            "cz",
            "crx",
            "cry",
            "crz",
            "ccx",
            "cswap",
            "csx",
            "cu",
            "cu1",
            "cu3",
            "dcx",
            "h",
            "i",
            "id",
            "iden",
            "iswap",
            "ms",
            "p",
            "r",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzx",
            "rzz",
            "s",
            "sdg",
            "swap",
            "sx",
            "x",
            "y",
            "z",
            "t",
            "tdg",
            "u",
            "u1",
            "u2",
            "u3",
        ]

        existing_composite_circuits = []

        string_temp = self.header + "\n"
        string_temp += self.extension_lib + "\n"
        for register in self.qregs:
            string_temp += register.qasm() + "\n"
        for register in self.cregs:
            string_temp += register.qasm() + "\n"

        qreg_bits = set(bit for reg in self.qregs for bit in reg)
        creg_bits = set(bit for reg in self.cregs for bit in reg)
        regless_qubits = []
        regless_clbits = []

        if set(self.qubits) != qreg_bits:
            regless_qubits = [bit for bit in self.qubits if bit not in qreg_bits]
            string_temp += "qreg %s[%d];\n" % ("regless", len(regless_qubits))

        if set(self.clbits) != creg_bits:
            regless_clbits = [bit for bit in self.clbits if bit not in creg_bits]
            string_temp += "creg %s[%d];\n" % ("regless", len(regless_clbits))

        unitary_gates = []

        bit_labels = {
            bit: "%s[%d]" % (reg.name, idx)
            for reg in self.qregs + self.cregs
            for (idx, bit) in enumerate(reg)
        }

        bit_labels.update(
            {
                bit: "regless[%d]" % idx
                for reg in (regless_qubits, regless_clbits)
                for idx, bit in enumerate(reg)
            }
        )

        for instruction, qargs, cargs in self._data:
            if instruction.name == "measure":
                qubit = qargs[0]
                clbit = cargs[0]
                string_temp += "%s %s -> %s;\n" % (
                    instruction.qasm(),
                    bit_labels[qubit],
                    bit_labels[clbit],
                )

            # If instruction is a root gate or a root instruction (in that case, compositive)

            elif (
                type(instruction)
                in [  # pylint: disable=unidiomatic-typecheck
                    Gate,
                    Instruction,
                ]
                or (isinstance(instruction, ControlledGate) and instruction._open_ctrl)
            ):
                if instruction not in existing_composite_circuits:
                    if instruction.name in existing_gate_names:
                        old_name = instruction.name
                        instruction.name += "_" + str(id(instruction))

                        warnings.warn(
                            "A gate named {} already exists. "
                            "We have renamed "
                            "your gate to {}".format(old_name, instruction.name)
                        )

                    # Get qasm of composite circuit
                    qasm_string = self._get_composite_circuit_qasm_from_instruction(instruction)

                    # Insert composite circuit qasm definition right after header and extension lib
                    string_temp = string_temp.replace(
                        self.extension_lib, "%s\n%s" % (self.extension_lib, qasm_string)
                    )

                    existing_composite_circuits.append(instruction)
                    existing_gate_names.append(instruction.name)

                # Insert qasm representation of the original instruction
                string_temp += "%s %s;\n" % (
                    instruction.qasm(),
                    ",".join([bit_labels[j] for j in qargs + cargs]),
                )
            else:
                string_temp += "%s %s;\n" % (
                    instruction.qasm(),
                    ",".join([bit_labels[j] for j in qargs + cargs]),
                )
            if instruction.name == "unitary":
                unitary_gates.append(instruction)

        # this resets them, so if another call to qasm() is made the gate def is added again
        for gate in unitary_gates:
            gate._qasm_def_written = False

        if filename:
            with open(filename, "w+") as file:
                file.write(string_temp)
            file.close()

        if formatted:
            if not HAS_PYGMENTS:
                raise ImportError(
                    "To use the formatted output pygments>2.4 "
                    "must be installed. To install pygments run "
                    '"pip install pygments".'
                )
            code = pygments.highlight(
                string_temp, OpenQASMLexer(), Terminal256Formatter(style=QasmTerminalStyle)
            )
            print(code)
            return None
        else:
            return string_temp

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
            #from .visualization import circuit_drawer
            pass

        super().draw(kwargs)

    def size(self):
        """Returns total number of gate operations in circuit.

        Returns:
            int: Total number of gate operations.
        """
        gate_ops = 0
        for instr, _, _ in self._data:
            if not instr._directive:
                gate_ops += 1
        return gate_ops

    def depth(self):
        """Return circuit depth (i.e., length of critical path).
        This does not include compiler or simulator directives
        such as 'barrier' or 'snapshot'.

        Returns:
            int: Depth of circuit.

        Notes:
            The circuit depth and the DAG depth need not be the
            same.
        """
        # Assign each bit in the circuit a unique integer
        # to index into op_stack.
        bit_indices = {bit: idx for idx, bit in enumerate(self.qubits + self.clbits)}

        # If no bits, return 0
        if not bit_indices:
            return 0

        # A list that holds the height of each qubit
        # and classical bit.
        op_stack = [0] * len(bit_indices)

        # Here we are playing a modified version of
        # Tetris where we stack gates, but multi-qubit
        # gates, or measurements have a block for each
        # qubit or cbit that are connected by a virtual
        # line so that they all stacked at the same depth.
        # Conditional gates act on all cbits in the register
        # they are conditioned on.
        # We treat barriers or snapshots different as
        # They are transpiler and simulator directives.
        # The max stack height is the circuit depth.
        for instr, qargs, cargs in self._data:
            levels = []
            reg_ints = []
            # If count then add one to stack heights
            count = True
            if instr._directive:
                count = False
            for ind, reg in enumerate(qargs + cargs):
                # Add to the stacks of the qubits and
                # cbits used in the gate.
                reg_ints.append(bit_indices[reg])
                if count:
                    levels.append(op_stack[reg_ints[ind]] + 1)
                else:
                    levels.append(op_stack[reg_ints[ind]])
            # Assuming here that there is no conditional
            # snapshots or barriers ever.
            if instr.condition:
                # Controls operate over all bits in the
                # classical register they use.
                for cbit in instr.condition[0]:
                    idx = bit_indices[cbit]
                    if idx not in reg_ints:
                        reg_ints.append(idx)
                        levels.append(op_stack[idx] + 1)

            max_level = max(levels)
            for ind in reg_ints:
                op_stack[ind] = max_level

        return max(op_stack)

    def width(self):
        """Return number of qubits plus clbits in circuit.

        Returns:
            int: Width of circuit.

        """
        return len(self.qubits) + len(self.clbits)

    @property
    def num_qudits(self):
        """Return number of qudits."""
        return len(self.qudits)

    @property
    def num_qd_ancillas(self):
        """Return the number of ancilla qudits."""
        return len(self.qd_ancillas)

    def count_ops(self):
        """Count each operation kind in the circuit.

        Returns:
            OrderedDict: a breakdown of how many operations of each kind, sorted by amount.
        """
        count_ops = {}
        for instr, _, _ in self._data:
            count_ops[instr.name] = count_ops.get(instr.name, 0) + 1
        return OrderedDict(sorted(count_ops.items(), key=lambda kv: kv[1], reverse=True))

    def num_nonlocal_gates(self):
        """Return number of non-local gates (i.e. involving 2+ qubits).

        Conditional nonlocal gates are also included.
        """
        multi_qubit_gates = 0
        for instr, _, _ in self._data:
            if instr.num_qubits > 1 and not instr._directive:
                multi_qubit_gates += 1
        return multi_qubit_gates

    def num_connected_components(self, unitary_only=False):
        """How many non-entangled subcircuits can the circuit be factored to.

        Args:
            unitary_only (bool): Compute only unitary part of graph.

        Returns:
            int: Number of connected components in circuit.
        """
        # Convert registers to ints (as done in depth).
        bits = self.qubits if unitary_only else (self.qubits + self.clbits)
        bit_indices = {bit: idx for idx, bit in enumerate(bits)}

        # Start with each qubit or cbit being its own subgraph.
        sub_graphs = [[bit] for bit in range(len(bit_indices))]

        num_sub_graphs = len(sub_graphs)

        # Here we are traversing the gates and looking to see
        # which of the sub_graphs the gate joins together.
        for instr, qargs, cargs in self._data:
            if unitary_only:
                args = qargs
                num_qargs = len(args)
            else:
                args = qargs + cargs
                num_qargs = len(args) + (1 if instr.condition else 0)

            if num_qargs >= 2 and not instr._directive:
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                if instr.condition and not unitary_only:
                    creg = instr.condition[0]
                    for bit in creg:
                        idx = bit_indices[bit]
                        for k in range(num_sub_graphs):
                            if idx in sub_graphs[k]:
                                graphs_touched.append(k)
                                num_touched += 1
                                break

                for item in args:
                    reg_int = bit_indices[item]
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                num_touched += 1
                                break

                # If the gate touches more than one subgraph
                # join those graphs together and return
                # reduced number of subgraphs
                if num_touched > 1:
                    connections = []
                    for idx in graphs_touched:
                        connections.extend(sub_graphs[idx])
                    _sub_graphs = []
                    for idx in range(num_sub_graphs):
                        if idx not in graphs_touched:
                            _sub_graphs.append(sub_graphs[idx])
                    _sub_graphs.append(connections)
                    sub_graphs = _sub_graphs
                    num_sub_graphs -= num_touched - 1
            # Cannot go lower than one so break
            if num_sub_graphs == 1:
                break
        return num_sub_graphs

    def num_unitary_factors(self):
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.
        """
        return self.num_connected_components(unitary_only=True)

    def num_tensor_factors(self):
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.

        Notes:
            This is here for backwards compatibility, and will be
            removed in a future release of Qiskit. You should call
            `num_unitary_factors` instead.
        """
        return self.num_unitary_factors()

    def copy(self, name=None):
        """Copy the circuit.

        Args:
          name (str): name to be given to the copied circuit. If None, then the name stays the same

        Returns:
          QuantumCircuit: a deepcopy of the current circuit, with the specified name
        """
        cpy = copy.copy(self)
        # copy registers correctly, in copy.copy they are only copied via reference
        cpy.qregs = self.qregs.copy()
        cpy.cregs = self.cregs.copy()
        cpy._qubits = self._qubits.copy()
        cpy._clbits = self._clbits.copy()
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

        cpy._data = [
            (instr_copies[id(inst)], qargs.copy(), cargs.copy())
            for inst, qargs, cargs in self._data
        ]

        cpy._calibrations = copy.deepcopy(self._calibrations)
        cpy._metadata = copy.deepcopy(self._metadata)

        if name:
            cpy.name = name
        return cpy

    def _create_creg(self, length, name):
        """Creates a creg, checking if ClassicalRegister with same name exists"""
        if name in [creg.name for creg in self.cregs]:
            save_prefix = ClassicalRegister.prefix
            ClassicalRegister.prefix = name
            new_creg = ClassicalRegister(length)
            ClassicalRegister.prefix = save_prefix
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length, name):
        """Creates a qreg, checking if QuantumRegister with same name exists"""
        if name in [qreg.name for qreg in self.qregs]:
            save_prefix = QuantumRegister.prefix
            QuantumRegister.prefix = name
            new_qreg = QuantumRegister(length)
            QuantumRegister.prefix = save_prefix
        else:
            new_qreg = QuantumRegister(length, name)
        return new_qreg

    def measure_active(self, inplace=True):
        """Adds measurement to all non-idle qubits. Creates a new ClassicalRegister with
        a size equal to the number of non-idle qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when `inplace = False`.
        """
        from qiskit.converters.circuit_to_dag import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        new_creg = circ._create_creg(len(qubits_to_measure), "measure")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(qubits_to_measure, new_creg)

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
        circ.measure(circ.qubits, new_creg)

        if not inplace:
            return circ
        else:
            return None

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
        from .barrier import Barrier

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

    def h(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.HGate`."""
        from .library.standard_gates.h import HGate

        return self.append(HGate(), [qubit], [])

    def ch(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CHGate`."""
        from .library.standard_gates.h import CHGate

        return self.append(
            CHGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def i(self, qubit):
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        from .library.standard_gates.i import IGate

        return self.append(IGate(), [qubit], [])

    def id(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.IGate`."""
        return self.i(qubit)

    def ms(self, theta, qubits):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.MSGate`."""
        # pylint: disable=cyclic-import
        from .library.generalized_gates.gms import MSGate

        return self.append(MSGate(len(qubits), theta), qubits)

    def p(self, theta, qubit):
        """Apply :class:`~qiskit.circuit.library.PhaseGate`."""
        from .library.standard_gates.p import PhaseGate

        return self.append(PhaseGate(theta), [qubit], [])

    def cp(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CPhaseGate`."""
        from .library.standard_gates.p import CPhaseGate

        return self.append(
            CPhaseGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def mcp(self, lam, control_qubits, target_qubit):
        """Apply :class:`~qiskit.circuit.library.MCPhaseGate`."""
        from .library.standard_gates.p import MCPhaseGate

        num_ctrl_qubits = len(control_qubits)
        return self.append(
            MCPhaseGate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], []
        )

    def r(self, theta, phi, qubit):
        """Apply :class:`~qiskit.circuit.library.RGate`."""
        from .library.standard_gates.r import RGate

        return self.append(RGate(theta, phi), [qubit], [])

    def rv(self, vx, vy, vz, qubit):
        """Apply :class:`~qiskit.circuit.library.RVGate`."""
        from .library.generalized_gates.rv import RVGate

        return self.append(RVGate(vx, vy, vz), [qubit], [])

    def rccx(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RCCXGate`."""
        from .library.standard_gates.x import RCCXGate

        return self.append(RCCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def rcccx(self, control_qubit1, control_qubit2, control_qubit3, target_qubit):
        """Apply :class:`~qiskit.circuit.library.RC3XGate`."""
        from .library.standard_gates.x import RC3XGate

        return self.append(
            RC3XGate(), [control_qubit1, control_qubit2, control_qubit3, target_qubit], []
        )

    def rx(self, theta, qubit, label=None):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RXGate`."""
        from .library.standard_gates.rx import RXGate

        return self.append(RXGate(theta, label=label), [qubit], [])

    def crx(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRXGate`."""
        from .library.standard_gates.rx import CRXGate

        return self.append(
            CRXGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rxx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RXXGate`."""
        from .library.standard_gates.rxx import RXXGate

        return self.append(RXXGate(theta), [qubit1, qubit2], [])

    def ry(self, theta, qubit, label=None):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RYGate`."""
        from .library.standard_gates.ry import RYGate

        return self.append(RYGate(theta, label=label), [qubit], [])

    def cry(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRYGate`."""
        from .library.standard_gates.ry import CRYGate

        return self.append(
            CRYGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def ryy(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RYYGate`."""
        from .library.standard_gates.ryy import RYYGate

        return self.append(RYYGate(theta), [qubit1, qubit2], [])

    def rz(self, phi, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.RZGate`."""
        from .library.standard_gates.rz import RZGate

        return self.append(RZGate(phi), [qubit], [])

    def crz(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CRZGate`."""
        from .library.standard_gates.rz import CRZGate

        return self.append(
            CRZGate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def rzx(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZXGate`."""
        from .library.standard_gates.rzx import RZXGate

        return self.append(RZXGate(theta), [qubit1, qubit2], [])

    def rzz(self, theta, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.RZZGate`."""
        from .library.standard_gates.rzz import RZZGate

        return self.append(RZZGate(theta), [qubit1, qubit2], [])

    def ecr(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.ECRGate`."""
        from .library.standard_gates.ecr import ECRGate

        return self.append(ECRGate(), [qubit1, qubit2], [])

    def s(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.SGate`."""
        from .library.standard_gates.s import SGate

        return self.append(SGate(), [qubit], [])

    def sdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.SdgGate`."""
        from .library.standard_gates.s import SdgGate

        return self.append(SdgGate(), [qubit], [])

    def swap(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.SwapGate`."""
        from .library.standard_gates.swap import SwapGate

        return self.append(SwapGate(), [qubit1, qubit2], [])

    def iswap(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.iSwapGate`."""
        from .library.standard_gates.iswap import iSwapGate

        return self.append(iSwapGate(), [qubit1, qubit2], [])

    def cswap(self, control_qubit, target_qubit1, target_qubit2, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        from .library.standard_gates.swap import CSwapGate

        return self.append(
            CSwapGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit1, target_qubit2],
            [],
        )

    def fredkin(self, control_qubit, target_qubit1, target_qubit2):
        """Apply :class:`~qiskit.circuit.library.CSwapGate`."""
        return self.cswap(control_qubit, target_qubit1, target_qubit2)

    def sx(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.SXGate`."""
        from .library.standard_gates.sx import SXGate

        return self.append(SXGate(), [qubit], [])

    def sxdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.SXdgGate`."""
        from .library.standard_gates.sx import SXdgGate

        return self.append(SXdgGate(), [qubit], [])

    def csx(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CSXGate`."""
        from .library.standard_gates.sx import CSXGate

        return self.append(
            CSXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def t(self, qubit):  # pylint: disable=invalid-name
        """Apply :class:`~qiskit.circuit.library.TGate`."""
        from .library.standard_gates.t import TGate

        return self.append(TGate(), [qubit], [])

    def tdg(self, qubit):
        """Apply :class:`~qiskit.circuit.library.TdgGate`."""
        from .library.standard_gates.t import TdgGate

        return self.append(TdgGate(), [qubit], [])

    def u(self, theta, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.UGate`."""
        from .library.standard_gates.u import UGate

        return self.append(UGate(theta, phi, lam), [qubit], [])

    def cu(self, theta, phi, lam, gamma, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CUGate`."""
        from .library.standard_gates.u import CUGate

        return self.append(
            CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    @deprecate_function(
        "The QuantumCircuit.u1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.p method instead, which acts "
        "identically."
    )
    def u1(self, theta, qubit):
        """Apply :class:`~qiskit.circuit.library.U1Gate`."""
        from .library.standard_gates.u1 import U1Gate

        return self.append(U1Gate(theta), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.cu1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.cp method instead, which acts "
        "identically."
    )
    def cu1(self, theta, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CU1Gate`."""
        from .library.standard_gates.u1 import CU1Gate

        return self.append(
            CU1Gate(theta, label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    @deprecate_function(
        "The QuantumCircuit.mcu1 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You should use the "
        "QuantumCircuit.mcp method instead, which acts "
        "identically."
    )
    def mcu1(self, lam, control_qubits, target_qubit):
        """Apply :class:`~qiskit.circuit.library.MCU1Gate`."""
        from .library.standard_gates.u1 import MCU1Gate

        num_ctrl_qubits = len(control_qubits)
        return self.append(MCU1Gate(lam, num_ctrl_qubits), control_qubits[:] + [target_qubit], [])

    @deprecate_function(
        "The QuantumCircuit.u2 method is deprecated as of "
        "0.16.0. It will be removed no earlier than 3 months "
        "after the release date. You can use the general 1-"
        "qubit gate QuantumCircuit.u instead: u2(φ,λ) = "
        "u(π/2, φ, λ). Alternatively, you can decompose it in"
        "terms of QuantumCircuit.p and QuantumCircuit.sx: "
        "u2(φ,λ) = p(π/2+φ) sx p(λ-π/2) (1 pulse on hardware)."
    )
    def u2(self, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.U2Gate`."""
        from .library.standard_gates.u2 import U2Gate

        return self.append(U2Gate(phi, lam), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.u3 method is deprecated as of 0.16.0. It will be "
        "removed no earlier than 3 months after the release date. You should use "
        "QuantumCircuit.u instead, which acts identically. Alternatively, you can "
        "decompose u3 in terms of QuantumCircuit.p and QuantumCircuit.sx: "
        "u3(ϴ,φ,λ) = p(φ+π) sx p(ϴ+π) sx p(λ) (2 pulses on hardware)."
    )
    def u3(self, theta, phi, lam, qubit):
        """Apply :class:`~qiskit.circuit.library.U3Gate`."""
        from .library.standard_gates.u3 import U3Gate

        return self.append(U3Gate(theta, phi, lam), [qubit], [])

    @deprecate_function(
        "The QuantumCircuit.cu3 method is deprecated as of 0.16.0. It will be "
        "removed no earlier than 3 months after the release date. You should "
        "use the QuantumCircuit.cu method instead, where "
        "cu3(ϴ,φ,λ) = cu(ϴ,φ,λ,0)."
    )
    def cu3(self, theta, phi, lam, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CU3Gate`."""
        from .library.standard_gates.u3 import CU3Gate

        return self.append(
            CU3Gate(theta, phi, lam, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
        )

    def x(self, qubit, label=None):
        """Apply :class:`~qiskit.circuit.library.XGate`."""
        from .library.standard_gates.x import XGate

        return self.append(XGate(label=label), [qubit], [])

    def cx(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        from .library.standard_gates.x import CXGate

        return self.append(
            CXGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def cnot(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CXGate`."""
        self.cx(control_qubit, target_qubit, label, ctrl_state)

    def dcx(self, qubit1, qubit2):
        """Apply :class:`~qiskit.circuit.library.DCXGate`."""
        from .library.standard_gates.dcx import DCXGate

        return self.append(DCXGate(), [qubit1, qubit2], [])

    def ccx(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        from .library.standard_gates.x import CCXGate

        return self.append(CCXGate(), [control_qubit1, control_qubit2, target_qubit], [])

    def toffoli(self, control_qubit1, control_qubit2, target_qubit):
        """Apply :class:`~qiskit.circuit.library.CCXGate`."""
        self.ccx(control_qubit1, control_qubit2, target_qubit)

    def mcx(self, control_qubits, target_qubit, ancilla_qubits=None, mode="noancilla"):
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:
        - 'noancilla': Requires 0 ancilla qubits.
        - 'recursion': Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - 'v-chain': Requires 2 less ancillas than the number of control qubits.
        - 'v-chain-dirty': Same as for the clean ancillas (but the circuit will be longer).
        """
        from .library.standard_gates.x import MCXGrayCode, MCXRecursive, MCXVChain

        num_ctrl_qubits = len(control_qubits)

        available_implementations = {
            "noancilla": MCXGrayCode(num_ctrl_qubits),
            "recursion": MCXRecursive(num_ctrl_qubits),
            "v-chain": MCXVChain(num_ctrl_qubits, False),
            "v-chain-dirty": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
            # outdated, previous names
            "advanced": MCXRecursive(num_ctrl_qubits),
            "basic": MCXVChain(num_ctrl_qubits, dirty_ancillas=False),
            "basic-dirty-ancilla": MCXVChain(num_ctrl_qubits, dirty_ancillas=True),
        }

        # check ancilla input
        if ancilla_qubits:
            _ = self.qbit_argument_conversion(ancilla_qubits)

        try:
            gate = available_implementations[mode]
        except KeyError as ex:
            all_modes = list(available_implementations.keys())
            raise ValueError(
                f"Unsupported mode ({mode}) selected, choose one of {all_modes}"
            ) from ex

        if hasattr(gate, "num_ancilla_qubits") and gate.num_ancilla_qubits > 0:
            required = gate.num_ancilla_qubits
            if ancilla_qubits is None:
                raise AttributeError("No ancillas provided, but {} are needed!".format(required))

            # convert ancilla qubits to a list if they were passed as int or qubit
            if not hasattr(ancilla_qubits, "__len__"):
                ancilla_qubits = [ancilla_qubits]

            if len(ancilla_qubits) < required:
                actually = len(ancilla_qubits)
                raise ValueError(
                    "At least {} ancillas required, but {} given.".format(required, actually)
                )
            # size down if too many ancillas were provided
            ancilla_qubits = ancilla_qubits[:required]
        else:
            ancilla_qubits = []

        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    def mct(self, control_qubits, target_qubit, ancilla_qubits=None, mode="noancilla"):
        """Apply :class:`~qiskit.circuit.library.MCXGate`."""
        return self.mcx(control_qubits, target_qubit, ancilla_qubits, mode)

    def y(self, qubit):
        """Apply :class:`~qiskit.circuit.library.YGate`."""
        from .library.standard_gates.y import YGate

        return self.append(YGate(), [qubit], [])

    def cy(
        self,
        control_qubit,
        target_qubit,  # pylint: disable=invalid-name
        label=None,
        ctrl_state=None,
    ):
        """Apply :class:`~qiskit.circuit.library.CYGate`."""
        from .library.standard_gates.y import CYGate

        return self.append(
            CYGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def z(self, qubit):
        """Apply :class:`~qiskit.circuit.library.ZGate`."""
        from .library.standard_gates.z import ZGate

        return self.append(ZGate(), [qubit], [])

    def cz(self, control_qubit, target_qubit, label=None, ctrl_state=None):
        """Apply :class:`~qiskit.circuit.library.CZGate`."""
        from .library.standard_gates.z import CZGate

        return self.append(
            CZGate(label=label, ctrl_state=ctrl_state), [control_qubit, target_qubit], []
        )

    def pauli(self, pauli_string, qubits):
        """Apply :class:`~qiskit.circuit.library.PauliGate`."""
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate

        return self.append(PauliGate(pauli_string), qubits, [])

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
