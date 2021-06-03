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

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import XGate
from qiskit.exceptions import QiskitError


#Dephasing to m level    
def Dephasing(circuit,m,phase):
    n=circuit.width()-1
    if 2**n <m :
        raise QiskitError('Circuit does not allow to map this level. Try a lower level or a bigger circuit.')
    qubits=QuantumRegister(n+1)
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
    


#Pi coupling between m and l level
def LevelsSwitch(circuit,m,l):
    n=circuit.width()-1 
    if 2**n <m or 2**n <l:
        raise QiskitError('Circuit does not allow to map this level. Try a lower level or a bigger circuit.')
    qubits= QuantumRegister(n+1)
    marray=[]
    larray=[]
    for i in range(0,n): #bit decomposition
        if (( m >>  i) & 1) != 1 :
            marray.append(i)
    for i in range(0,n):
        if (( l >>  i) & 1) != 1 :
            larray.append(i)
    control_qubits=[]
    for i in range(0,n):
        control_qubits.append([i])
    target_qubit=[n]
    #check if m
    if len(marray)>0:
        circuit.x(marray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(marray)
    #check if l
    if len(larray)>0:
        circuit.x(larray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(larray)
    #swap
    for i in range(0,n):
        if ((( m >>  i) & 1) != (( l >>  i) & 1)):
            circuit.cx(n,i)
    #check if m, put back auxiliary qubit    
    if len(marray)>0:
        circuit.x(marray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(marray)
    #check if l, put back auxiliary qubit
    if len(larray)>0:
        circuit.x(larray)
        circuit.mcx(control_qubits,target_qubit)
        circuit.x(larray)
    return circuit
