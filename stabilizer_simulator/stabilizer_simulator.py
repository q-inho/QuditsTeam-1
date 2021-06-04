#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is from Qiskit Hackathon 2021 by the team
Qiskit for high dimensional multipartite quantum states.

Author: Hoang Van Do

(C) Copyright 2021 Hoang Van Do, Tim Alexis Körner, Inho Choi, Timothé Presles
and Élie Gouzien.

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
"""
from math import prod


class PauliND:
    """Class for Pauli operators in Weyl-Heisenberg group for one qudit."""

    def __init__(self, d, pauli_list):
        """Initialize a new Pauli operator.

        Arguments:
            d   : dimensionality of the qudit (prime number)
            pauli_list : (w_N,  P_N, Q_N) : orders of the generators.
            Q_N
        """
        # if max(pauli_list) >= d or min(pauli_list) < 0:
        #     warn("Generalized Pauli matrices have orders as their dimension."
        #          "Orders will be modulo by dimension.")
        self.d = d
        self.pauli_list = pauli_list
        self._reduce_orders()
        if len(self.pauli_list) != 3:
            raise ValueError("Invalid phase and generator orders format. "
                             "Please enter a list of length 3.")

    def _reduce_orders(self):
        """Bring orders back in Z_p."""
        self.pauli_list = [i % self.d for i in self.pauli_list]

    def __repr__(self):
        """Representation."""
        return f"PauliND({self.d}, {self.pauli_list})"

    def __str__(self):
        """Zoli string."""
        w, p, q = self.pauli_list
        return f"ω**{w} p**{p} q**{q}"

    def __mul__(self, other):
        """Multiplication betwen two Pauli operators (not commutative)."""
        if not self.d == other.d:
            raise ValueError("Only qudits with same d can be multiplied.")
        w1, p1, q1 = self.pauli_list
        w2, p2, q2 = other.pauli_list
        return __class__(self.d, [(w1+w2+q1*p2), p1+p2, q1+q2])

    def __pow__(self, power):
        """Raise to a power."""
        # Stupid algorithm (use instead fast exponentiation if large d).
        if not isinstance(power, int):
            return NotImplemented
        return prod((self for _ in range(power)),
                    start=__class__(self.d, (0, 0, 0)))


class CliffordND:
    """Represent Clifford operators."""

    def __init__(self, new_p: PauliND, new_q: PauliND):
        """Initialize the Clifford operator from it's tableau."""
        if not new_p.d == new_q.d:
            raise ValueError("Pauli operator should work on same qudits.")
        self.d = new_p.d
        self.new_p = new_p
        self.new_q = new_q

    def __repr__(self):
        """Representation."""
        return f"CliffordND({self.new_p!r}, {self.new_q!r})"

    def __str__(self):
        """Zoli string."""
        return f"[p -> {self.new_p}, q -> {self.new_q}]"

    def __call__(self, pauli: PauliND):
        """Apply the Clifford gate to a Pauli operator."""
        if not self.d == pauli.d:
            raise ValueError("Dimension of qudits doesn't match.")
        w, p, q = pauli.pauli_list
        return PauliND(self.d, (w, 0, 0)) * self.new_p**p * self.new_q**q

    def __mul__(self, other):
        """Do the multiplication (rhs applied before lhs)."""
        if not self.d == other.d:
            raise ValueError("Only Clifford of qudits with same d can be *.")
        return __class__(self(other.new_p), self(other.new_q))

    @classmethod
    def P(cls, d):
        """P gate (as a Clifford gate."""
        p = PauliND(d, [0, 1, 0])
        p_inv = PauliND(d, [0, d-1, 0])
        q = PauliND(d, [0, 0, 1])
        return cls(p, p*q*p_inv)

    @classmethod
    def Q(cls, d):
        """Q gate (as a Clifford gate."""
        p = PauliND(d, [0, 1, 0])
        q = PauliND(d, [0, 0, 1])
        q_inv = PauliND(d, [0, 0, q])
        return cls(q*p*q_inv, q)

    @classmethod
    def S(cls, d):
        """S gate."""
        return cls(PauliND(d, [0, 0, d-1]), PauliND(d, [0, 1, 0]))

    @classmethod
    def D(cls, d):
        """D¨ gate."""
        return cls(PauliND(d, [0, 1, 0]), PauliND(d, [0, 1, 1]))
