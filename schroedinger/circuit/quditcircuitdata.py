# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumcircuitdata.py from the original Qiskit-Terra code.
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

"""A wrapper class for the purposes of validating modifications to both QuditCircuit.data
and QuditCircuit.qd_data while maintaining the interface of a python list.
Keeps the data and qd_data synchronized,
as each item of data is a subset of the item of qd_data at the same index."""

from qiskit.circuit.quantumcircuitdata import QuantumCircuitData
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.exceptions import CircuitError

from .quditcircuit import QuditCircuit
from .quditinstruction import QuditInstruction

# TODO: check if sync may mean qd_len1 <= len2
class QuditCircuitData(QuantumCircuitData):
    """A wrapper class for the purposes of validating modifications to QuditCircuit.data
    and QuditCircuit.qd_data while maintaining the interface of a python list."""

    def __init__(self, circuit):
        """
        Attempts to synchronize data with qd_data in case one of them was directly assigned.

        Args:
            circuit: Associated QuditCircuit instance.
        Raises:
            CircuitError: If circuit is not a QuditCircuit.
            CircuitError: If data and qd_data were both previously assigned but mismatch.
        """
        if not isinstance(circuit, QuditCircuit):
            raise CircuitError("Circuit of QuditCircuitData must be a QuditCircuit")
        super().__init__(circuit)

        if self._circuit._data != [self.convert(data_tuple)
                                   for data_tuple in self._circuit._qd_data]:

            if self._circuit._data and self._circuit._qd_data:
                raise CircuitError("data and qd_data mismatch due to previous assignments")

            if self._circuit._data:
                self._circuit._qd_data = [self.convert(data_tuple)
                                          for data_tuple in self._circuit._data]

            if self._circuit._qd_data:
                self._circuit._data = [self.convert(data_tuple)
                                       for data_tuple in self._circuit._qd_data]

    @staticmethod
    def convert(data_tuple):
        """
        Converts a qd_data tuple to a data tuple and vice versa.
        Conversion from data to qd_data is lossless.
        Conversion from qd_data to data is lossless if qargs is empty.

        Args:
            data_tuple (tuple): Tuple (instruction, (qdargs,) qargs, cargs),
                qdargs / qargs / cargs is a list of qudits / qubits / classical bits.

        Returns:
            data_tuple (tuple): Converted data tuple.

        """
        if len(data_tuple) == 3:
            inst, qargs, cargs = data_tuple
            return inst, [], qargs, cargs

        inst, qdargs, qargs, cargs = data_tuple
        return inst, [qubit for qudit in qdargs for qubit in qudit] + qargs, cargs

    def qd_get(self, key):
        """Returns data tuple including qudit context at index key or
         a list of tuples with slice as key."""
        return self._circuit._qd_data[key]

    def __getitem__(self, key):
        """
        Supports imaginary numbers (e.g. 0j) for indexing into qd_data tuples.

        Arg:
            key (int or complex or slice or list): Index of the data tuple to be retrieved.
        Returns:
            Data tuple / list of data tuples with qudit context if key is a purely imaginary
            integer / slice with purely imaginary integers.
            Data tuple / list of data tuples without qudit context if key is a real
            integer / slice with real integers.

        Raises:
            TypeError: If the `key` is complex but not a purely imaginary integer.
            TypeError: If the `key` is a slice with different index types (not regarding None).
        """
        if isinstance(key, complex):
            if key.real != 0 or int(key.imag) != key.imag:
                raise TypeError("Complex keys must be purely imaginary integers.")

            return self.qd_get(int(key.imag))

        if isinstance(key, slice):
            slice_types = set(type(i) for i in (key.start, key.stop, key.step) if i is not None)
            if len(slice_types) > 1:
                raise TypeError("All slice indices must either have the same type or be None.")

            if any(type(idx) is complex for idx in (key.start, key.stop, key.step)):

                if any(idx.real != 0 or int(idx.imag) != idx.imag
                       for idx in (key.start, key.stop, key.step) if idx is not None):
                    raise TypeError("Complex slice indices must be purely imaginary integers.")

                return self.qd_get(slice(
                    *(int(idx.imag) if idx is not None else None
                      for idx in (key.start, key.stop, key.step))
                ))

        return super().__getitem__(key)

    def __setitem__(self, key, value):
        """
        Supports imaginary numbers (e.g. 0j) to get qd_data tuples.

        Arg:
            key int: Index of the data tuple to set.
            value tuple: Data tuple containing instruction, qdargs (optional), qargs, cargs.

        Raises:
            CircuitError: If the instruction is not a valid Instruction.
            CircuitError: If the instruction is not a QuditInstruction but `value` includes qdargs.
            CircuitError: If `value` cannot be interpreted without broadcasting.
        """
        inst, qargs, cargs = value[0], value[-2], value[-1]

        if len(value) == 4:
            qdargs = value[2]
        else:
            qdargs = []

        expanded_qdargs = [self._circuit.qdit_argument_conversion(qdarg)
                           for qdarg in qdargs or []]
        expanded_qargs = [self._circuit.qbit_argument_conversion(qarg)
                          for qarg in qargs or []]
        expanded_cargs = [self._circuit.cbit_argument_conversion(carg)
                          for carg in cargs or []]

        if qdargs:
            if not isinstance(inst, QuditInstruction) and \
                    hasattr(inst, "to_qd_instruction"):
                inst = inst.to_qd_instruction()

            if not isinstance(inst, QuditInstruction):
                raise CircuitError(
                    "Instruction argument is not an QuditInstruction, "
                    "but value includes qdargs."
                )

            broadcast_args = list(inst.qd_broadcast_arguments(
                expanded_qdargs, expanded_qargs, expanded_cargs)
            )
            qdargs, qargs, cargs = broadcast_args[0]
        else:
            if not isinstance(inst, Instruction) \
                    and hasattr(inst, 'to_instruction'):
                inst = inst.to_instruction()

            if not isinstance(inst, Instruction):
                raise CircuitError('Instruction argument is not an Instruction.')

            broadcast_args = list(inst.broadcast_arguments(
                expanded_qargs, expanded_cargs)
            )
            qargs, cargs = broadcast_args[0]

        if len(broadcast_args) > 1:
            raise CircuitError(
                "Circuit data modification does not support argument broadcasting."
            )

        self._circuit._check_dups(qdargs)
        self._circuit._check_dups(qargs)
        self._circuit._check_qdargs(qdargs)
        self._circuit._check_qargs(qargs)
        self._circuit._check_cargs(cargs)

        # add underlying qubits (of qdargs) and single qubits (qargs) to data
        # here qdargs is a list of qudits
        self._circuit._data[key] = (self.convert((inst, qdargs, qargs, cargs)))
        self._circuit._qd_data[key] = (inst, qdargs, qargs, cargs)

        self._circuit._update_parameter_table(inst)

    def insert(self, index, value):
        """inserts value for data and qd_data at index"""
        self._circuit._data.insert(index, None)
        self._circuit._qd_data.insert(index, None)
        self[index] = value

    def __delitem__(self, i):
        """deletes data and qd_data at index"""
        del self._circuit._data[i]
        del self._circuit._qd_data[i]

    def __cast(self, other):
        """casts to QuditCircuitData if possible"""
        if isinstance(other, QuditCircuitData):
            return other._circuit._qd_data
        if isinstance(other, QuantumCircuitData):
            return other._circuit._data
        return other

    def __repr__(self):
        """returns qd_data representation"""
        return repr(self._circuit._qd_data)

    def __add__(self, other):
        """Adds qd_data to other qd_data if possible.
        Otherwise defaults to adding data to other data."""
        if isinstance(other, QuditCircuitData):
            return self._circuit._qd_data + self.__cast(other)
        return self._circuit._data + self.__cast(other)

    def __radd__(self, other):
        """Adds other qd_data to qd_data if possible.
        Otherwise defaults to adding other data to data."""
        if isinstance(other, QuditCircuitData):
            return self.__cast(other) + self._circuit._qd_data
        return self.__cast(other) + self._circuit._data

    def __mul__(self, n):
        """
        Multiplies and returns data*n if `n` is real integer,
        qd_data*n if `n` is a purely imaginary integer.

        Raises:
            TypeError: If the `n` is complex but not a purely imaginary integer.
        """
        if isinstance(n, complex):
            if n.real != 0 or int(n.imag) != n.imag:
                raise TypeError("Complex factors must be purely imaginary integers.")

            return self._circuit._qd_data * int(n.imag)
        return self._circuit._data * n

    def __rmul__(self, n):
        """
        Multiplies and returns n*data if `n` is real integer,
        n*qd_data if `n` is a purely imaginary integer.

        Raises:
            TypeError: If the `n` is complex but not a purely imaginary integer.
        """
        if isinstance(n, complex):
            if n.real != 0 or int(n.imag) != n.imag:
                raise TypeError("Complex factors must be purely imaginary integers.")

            return int(n.imag) * self._circuit._qd_data
        return n * self._circuit._data

    def __iadd__(self, other):
        """In-place adding of data and qd_data with other data and qd_data."""
        if isinstance(other, QuditCircuitData):
            self._circuit._qd_data += other._circuit._qd_data
            self._circuit._data += other._circuit._data
        else:
            self._circuit._qd_data += [
                self.convert(data_tuple) for data_tuple in self.__cast(other)
            ]
            self._circuit._data += self.__cast(other)

    def __imul__(self, n):
        """In-place multiplication of data and qd_data with `n`."""
        self._circuit._qd_data *= n
        self._circuit._data *= n


    def sort(self, *args, **kwargs):
        """In-place stable sort of both data and qd_data. Accepts arguments of list.sort."""
        self._circuit._data.sort(*args, **kwargs)
        self._circuit._qd_data.sort(*args, **kwargs)

    def qd_copy(self):
        """Returns a shallow copy of qudit instruction list."""
        return self._circuit._qd_data.copy()
