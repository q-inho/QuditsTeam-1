# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumregister.py from the original Qiskit-Terra code.
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

"""
Qudit and qudit register reference object.
"""

import numpy as np
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.exceptions import CircuitError


class Qudit(Qubit):
    """Implement a higher dimension qudit via multiple qubits.

    A Qudit of dimension d is represented by ceil(log2(d)) qubits
    which are referenced by an internal list.
    """

    __slots__ = ["_dimension", "_size", "_qubits"]

    def __init__(self, dimension, qubits, register=None, index=None):
        """Creates a qudit.

        Args:
            dimension (int): Dimension of the qudit
            qubits (list): List of Qubits representing the qudit.
            register (QuditRegister): Optional.
                A qudit register containing the bit.
            index (int): Optional. The index of the first qudit in its register.
        Raises:
            TypeError: If the ``dimension`` type is incorrect.
            TypeError: If ``qubits`` contained qubits of an incorrect type.
            ValueError: If the ``dimension`` value is smaller than two
            CircuitError: If the provided ``register`` is not a
                valid :class:`QuditQuantumRegister`.
            CircuitError: If index is not within bounds of ``register``.
        """
        try:
            dimension = int(dimension)
        except Exception as ex:
            raise TypeError(
                "dimension needs to be castable to an int: "
                f"type {type(dimension)} was provided"
            ) from ex
        if dimension < 2:
            raise ValueError(
                "dimension must be an integer greater or equal to 2"
            )
        self._dimension = dimension

        if not isinstance(qubits, list) or \
                any(not isinstance(qubit, Qubit) for qubit in qubits):
            raise TypeError(
                "only a list of Qubit instances is accepted for qubits argument"
            )
        self._size = len(qubits)
        self._qubits = qubits

        if register is None or isinstance(register, QuditRegister):
            # Super class will only check if index is within size of QuditRegister
            # which is not the number of qudits in the register, but the number of qubits
            # representing them.
            try:
                index = int(index)
                if index < 0:
                    index += register.qd_size
                if index > register.qd_size:
                    raise CircuitError(
                        "index must be under the size of the register: "
                        f"{index} was provided"
                    )
            except (ValueError, TypeError):
                # Will be properly addressed in super class.
                pass

            super().__init__(register, index)
        else:
            raise CircuitError(
                f"Qudit needs a QuditRegister and {type(register)} was provided"
            )

    @property
    def dimension(self):
        """Get the qubit dimension."""
        return self._dimension

    @property
    def size(self):
        """Return number of qubits representing this qudit."""
        return self._size

    def __len__(self):
        """Return number of qubits representing this qudit."""
        return self._size

    def __getitem__(self, key):
        """
        Arg:
            key (int or slice or list): Index of the qubit to be retrieved.
        Returns:
            A Qubit instance if key is int, a list of Qubits if key is slice.

        Raises:
            CircuitError: If the `key` is not an integer.
            QiskitIndexError: If the `key` is not in the range `(0, self.qubit_count)`.
        """
        if not isinstance(key, (int, np.integer, slice, list)):
            raise CircuitError("expected integer or slice index into qubits")
        if isinstance(key, slice):
            return self._qubits[key]
        elif isinstance(key, list):
            if max(key) < len(self):
                return [self._qubits[idx] for idx in key]
            else:
                raise CircuitError("qubit index out of range")
        else:
            return self._qubits[key]

    def __iter__(self):
        """Iterates through Qubits of the Qudit."""
        for idx in range(self._size):
            yield self._qubits[idx]


class QuditRegister(QuantumRegister):
    """Implement a qudit register.

    A QuditRegister acts like a QuantumRegister except when addressed with
    any non-inherited subclass method. As a QuantumRegister it contains Qubits used
    for the quantum circuit. Additionally Qudits are registered and can be accessed
    via the ``qd_get`` or ``qd_iter`` method. Qudits are layered on top of their
    associated Qubits and hold references to them.
    """

    __slots__ = ["_qd_size", "_qudits"]

    # Prefix to use for auto naming.
    prefix = "qd"
    # bit_type is not set and inherited as qubit to
    # use bit_type constructor of super class

    def __init__(self, qudit_dimensions, name=None):
        """Create a new register for qudits.

        Args:
            qudit_dimensions (list[int]): A list of int with dimensions of
                multiple qudits in order.
            name (str): Optional. The name of the register. If not provided, a
               unique name will be auto-generated from the register type.
        Raises:
            TypeError: If ``qudit_dimensions`` has an incorrect type.
            CircuitError: If a dimension in ``qudit_dimensions`` is smaller than two.
        """
        if not isinstance(qudit_dimensions, list) or \
                any(not isinstance(dimension, int) for dimension in qudit_dimensions):
            raise TypeError("qudit_dimensions must be a list of integers")
        if any(d < 2 for d in qudit_dimensions):
            raise CircuitError("qudit dimension must be 2 or higher")

        qubit_counts = [int(np.ceil(np.log2(dimension))) for dimension in qudit_dimensions]
        super().__init__(sum(qubit_counts), name)

        self._qd_size = len(qudit_dimensions)
        self._qudits = []
        for idx in range(self._qd_size):

            self._qudits.append(
                # Each qudit gets a partial and disjoint list of
                # all qubits with the length according to the dimension
                Qudit(qudit_dimensions[idx],
                      self[sum(qubit_counts[:idx]): sum(qubit_counts[:idx + 1])],
                      self,
                      idx
                      )
            )

    @property
    def qd_size(self):
        """get the register size in terms of qudits"""
        return self._qd_size

    def qd_len(self):
        """get the register size in terms of qudits"""
        return self._qd_size

    def qd_get(self, key):
        """
        Get single qudit at index key or list of qudits with slice as key.

        Arg:
            key (int or slice or list): Index of the qudit to be retrieved.
        Returns:
            A Qudit instance if key is int, A list of Qudits if key is slice.

        Raises:
            CircuitError: If the `key` is not an integer.
            QiskitIndexError: If the `key` is not in the range `(0, self.qd_size)`.
        """
        if not isinstance(key, (int, np.integer, slice, list)):
            raise CircuitError(f"Expected integer or slice index into register, got {type(key)}.")
        if isinstance(key, slice):
            return self._qudits[key]
        elif isinstance(key, list):
            if max(key) < len(self):
                return [self._qudits[idx] for idx in key]
            else:
                raise CircuitError("register index out of range")
        else:
            return self._qudits[key]

    def __getitem__(self, key):
        """
        Supports imaginary numbers (e.g. 0j) for indexing into qudits.

        Arg:
            key (int or complex or slice or list): Index of the qubit or qudit to be retrieved.
        Returns:
            A Qudit instance / list of Qudits if key is a purely imaginary integer / slice
            with purely imaginary integers.
            A Qubit instance / list of Qubits if key is a real integer / slice with real integers.

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

    def qd_iter(self):
        """iterator for the register"""
        for idx in range(self._qd_size):
            yield self._qudits[idx]


class AncillaQudit(Qudit):
    """A qudit used as ancillary qudit."""

    pass


class AncillaQuditRegister(QuditRegister):
    """Implement an ancilla register for qudits."""

    # Prefix to use for auto naming.
    prefix = "ad"
    bit_type = AncillaQudit
