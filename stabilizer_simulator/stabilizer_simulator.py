#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is from Qiskit Hackathon 2021 by the team
Qiskit for high dimensional multipartite quantum states.

Author: Hoang Van Do, Élie Gouzien

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
            pauli_list : (w, q, p) : orders of the generators.
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
        w, q, p = self.pauli_list
        return f"ω**{w} q**{q} p**{p}"

    def __mul__(self, other):
        """Multiplication betwen two Pauli operators (not commutative)."""
        if not self.d == other.d:
            raise ValueError("Only qudits with same d can be multiplied.")
        w1, q1, p1 = self.pauli_list
        w2, q2, p2 = other.pauli_list
        return __class__(self.d, [(w1+w2+p1*q2), q1+q2, p1+p2])

    def __pow__(self, power):
        """Raise to a power."""
        # Stupid algorithm (use instead fast exponentiation if large d).
        if not isinstance(power, int):
            return NotImplemented
        return prod((self for _ in range(power)),
                    start=__class__(self.d, (0, 0, 0)))

    @classmethod
    def q(cls, d):
        """Q operator, as Pauli operator."""
        return cls(d, (0, 1, 0))

    @classmethod
    def p(cls, d):
        """P operator, as Pauli operator."""
        return cls(d, (0, 0, 1))


class CliffordND:
    """Represent Clifford operators."""

    def __init__(self, new_q: PauliND, new_p: PauliND):
        """Initialize the Clifford operator from it's tableau."""
        if not new_q.d == new_p.d:
            raise ValueError("Pauli operator should work on same qudits.")
        self.d = new_q.d
        self.new_q = new_q
        self.new_p = new_p

    def __repr__(self):
        """Representation."""
        return f"CliffordND({self.new_q!r}, {self.new_p!r})"

    def __str__(self):
        """Zoli string."""
        return f"[q -> {self.new_q}, p -> {self.new_p}]"

    def __call__(self, pauli: PauliND):
        """Apply the Clifford gate to a Pauli operator."""
        if not self.d == pauli.d:
            raise ValueError("Dimension of qudits doesn't match.")
        w, q, p = pauli.pauli_list
        return PauliND(self.d, (w, 0, 0)) * self.new_q**q * self.new_p**p

    def __mul__(self, other):
        """Do the multiplication (rhs applied before lhs)."""
        if not self.d == other.d:
            raise ValueError("Only Clifford of qudits with same d can be *.")
        return __class__(self(other.new_q), self(other.new_p))

    @classmethod
    def Q(cls, d):
        """Q gate (as a Clifford gate."""
        q = PauliND.q(d)
        q_inv = q**(d-1)
        p = PauliND.p(d)
        return cls(q, q*p*q_inv)

    @classmethod
    def P(cls, d):
        """P gate (as a Clifford gate."""
        q = PauliND.q(d)
        p = PauliND.p(d)
        p_inv = p**(d-1)
        return cls(p*q*p_inv, p)

    @classmethod
    def S(cls, d):
        """S gate."""
        return cls(PauliND(d, [0, 0, d-1]), PauliND(d, [0, 1, 0]))

    @classmethod
    def D(cls, d):
        """D¨ gate."""
        return cls(PauliND(d, [0, 1, 0]), PauliND(d, [0, 1, 1]))
