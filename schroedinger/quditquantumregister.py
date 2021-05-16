# This code is from Qiskit Hackathon 2021 by the Team
# Qiskit for high dimensional multipartite quantum states.
# It is a modified version of quantumregister.py from the original Qiskit-Terra code.
#
# (C) Copyright Qiskit Hackathon 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qudit and qudit register reference object.
"""

import itertools
from numpy import ceil, log2, integer

from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.exceptions import CircuitError


class Qudit(Qubit):
    """Implement a higher dimension qudit via multiple qubits.

    A Qudit of dimension d is represented by ceil(log2(d)) qubits
    which are referenced by an internal list.
    """

    __slots__ = ["_dimension", "_qubit_count", "_qubits"]

    def __init__(self, dimension, qubits, register=None, index=None):
        """Creates a qudit.
        Args:
            dimension (int): Dimension of the qudit
            qubits (list): List of Qubits representing the qudit.
            register (QuditQuantumRegister): Optional.
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
        self._qubit_count = len(qubits)
        self._qubits = qubits

        if register is None or isinstance(register, QuditQuantumRegister):
            # Super class will only check if index is within size of QuditQuantumRegister
            # which is not the number of qudits in the register, but the number of qubits
            # representing them.
            try:
                index = int(index)
                if index < 0:
                    index += register.qdsize
                if index > register.qdsize:
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
                f"Qudit needs a QuditQuantumRegister and {type(register)} was provided"
            )

    @property
    def dimension(self):
        """Get the qubit dimension."""
        return self._dimension

    @property
    def qubit_count(self):
        """Return number of qubits representing this qudit."""
        return self._qubit_count

    def __len__(self):
        """Return number of qubits representing this qudit."""
        return self._qubit_count

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
        if not isinstance(key, (int, integer, slice, list)):
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
        for idx in range(self._qubit_count):
            yield self._qubits[idx]


class QuditQuantumRegister(QuantumRegister):
    """Implement a qudit register.

    A QuditQuantumRegister acts like a QuantumRegister except when addressed with
    any non-inherited subclass method. As a QuantumRegister it contains Qubits used
    for the quantum circuit. Additionally Qudits are registered and can be accessed
    via the ``qdget`` or ``qditer`` method. Qudits are layered on top of their
    associated Qubits and hold references to them.
    """

    __slots__ = ["_qdsize", "_qudits"]

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "qd"
    # bit_type is not set and inherited as qubit to
    # use bit_type constructor of super class

    def __init__(self, dimensions, name=None):
        """Create a new register for qudits.

        bla

        Args:
            dimensions (int, list[int], dict[int: int]): Either an int describing
                the dimension of a single qudit or a list of int with dimensions of
                multiple qudits in order or a dictionary containing the
                dimensions as keys and corresponding qudit counts as values.
            name (str): Optional. The name of the register. If not provided, a
               unique name will be auto-generated from the register type.
        Raises:
            TypeError: If ``dimensions`` has an incorrect type.
            CircuitError: If a dimension in ``dimensions`` is smaller than two.
        """

        if isinstance(dimensions, int):
            dimensions = [dimensions]
        if isinstance(dimensions, dict):
            dimensions = [dimension for dimension in dimensions
                          for _ in range(dimensions[dimension])]
        if not isinstance(dimensions, list) or \
                any(not isinstance(dimension, int) for dimension in dimensions):
            raise TypeError(
                "Dimensions must either be the dimension for a single qudit (as an int) "
                "or a list of dimensions of qudits or a dictionary containing the"
                "the dimensions as keys and corresponding qudit counts as values"
            )
        if any(d < 2 for d in dimensions):
            raise CircuitError(
                "Qudit dimension must be 2 or higher"
            )

        qubit_counts = [int(ceil(log2(dimension))) for dimension in dimensions]
        super().__init__(sum(qubit_counts), name)

        self._qdsize = len(dimensions)
        self._qudits = []
        for idx in range(self._qdsize):

            self._qudits.append(
                # Each qudit gets a partial and disjoint list of
                # all qubits with the length according to the dimension
                Qudit(dimensions[idx],
                      self[sum(qubit_counts[:idx]): sum(qubit_counts[:idx + 1])],
                      self,
                      idx)
            )

    @property
    def qdsize(self):
        """Get the register size."""
        return self._qdsize

    def qdlen(self):
        """Get the register size."""
        return self._qdsize

    def qdget(self, key):
        """
        Arg:
            key (int or slice or list): Index of the qudit to be retrieved.
        Returns:
            A Qudit instance if key is int, A list of Qudits if key is slice.

        Raises:
            CircuitError: If the `key` is not an integer.
            QiskitIndexError: If the `key` is not in the range `(0, self.qdsize)`.
        """
        if not isinstance(key, (int, integer, slice, list)):
            raise CircuitError("expected integer or slice index into register")
        if isinstance(key, slice):
            return self._qudits[key]
        elif isinstance(key, list):
            if max(key) < len(self):
                return [self._qudits[idx] for idx in key]
            else:
                raise CircuitError("register index out of range")
        else:
            return self._qudits[key]

    def qditer(self):
        """Iterator for the register."""
        for idx in range(self._qdsize):
            yield self._qudits[idx]


class AncillaQudit(Qudit):
    """A qudit used as ancillary qudit."""

    pass


class AncillaQuditRegister(QuditQuantumRegister):
    """Implement an ancilla register for qudits."""

    # Counter for the number of instances in this class.
    instances_counter = itertools.count()
    # Prefix to use for auto naming.
    prefix = "ad"
    bit_type = AncillaQudit
