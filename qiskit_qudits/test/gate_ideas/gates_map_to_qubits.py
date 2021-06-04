# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
#
# Author: Hoang Van Do
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

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError


#Dephasing to m level. The inverse of the dephasing function would be giving the opposite phase Dephasing(m,-phase,dimension)
def Dephasing(m,phase,dimension):
    if dimension < m :
        raise QiskitError('The level is higher than the dimension')
    n=int(np.ceil(np.log2(dimension)))
    qubits=QuantumRegister(n+1)
    circuit=QuantumCircuit(qubits)
    control_qubits = qubits[:n]
    target_qubit = qubits[n]
    marray=[]
    for i in range(0,n): #bit decomposition
        if (( m >>  i) & 1) != 1 :
            marray.append(i)
    control_qubits=[]
    for i in range(0,n):
        control_qubits.append([i])
    target_qubit=[n]
    #check if m
    if len(marray)>0:
        circuit.x(marray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(marray)
    for i in range(0,n):
        circuit.cp(phase,n,i)
    if len(marray)>0:
        #check if m, put back auxiliary qubit
        circuit.x(marray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(marray)
    return circuit


#Pi coupling between m and l level. The inverse of the LevelsSwitch fucntion is itself
def level_switch(m, l, dimension):
    if m > dimension or l > dimension:
        raise QiskitError('The level is higher than the dimension')
    n = int(np.ceil(np.log2(dimension)))
    qubits = QuantumRegister(n + 1)
    circuit = QuantumCircuit(qubits)
    control_qubits = qubits[:n]
    target_qubit = qubits[n]

    # save indices of qubits which are 1 for states m, l
    marray = []
    larray = []
    for i in range(n):
        if (m >> i) & 1 != 1:
            marray.append(i)
    for i in range(n):
        if (l >> i) & 1 != 1:
            larray.append(i)

    # control on m, l
    if len(marray) > 0:
        circuit.x(marray)
    circuit.mcx(control_qubits, target_qubit)
    if len(marray) > 0:
        circuit.x(marray)
    if len(larray) > 0:
        circuit.x(larray)
    circuit.mcx(control_qubits, target_qubit)
    if len(larray) > 0:
        circuit.x(larray)

    # swap
    for i in range(n):
        if ((m >> i) & 1) != ((l >> i) & 1):
            circuit.cx(n, i)

    # control on m, l to reset auxiliary qubit
    if len(marray) > 0:
        circuit.x(marray)
    circuit.mcx(control_qubits, target_qubit)
    if len(marray) > 0:
        circuit.x(marray)
    if len(larray) > 0:
        circuit.x(larray)
    circuit.mcx(control_qubits, target_qubit)
    if len(larray) > 0:
        circuit.x(larray)

    return circuit
