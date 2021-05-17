# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:28:23 2021

@author: Hoang Van Do
"""

#import kinda everything
import numpy as np
from qiskit import QuantumCircuit, transpile, assemble, Aer, IBMQ
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_bloch_multivector

# Use Aer's qasm_simulator
simulator = QasmSimulator()



"""Van -  Single qubit gate"""

def dLvlSwap(circuit,x,y,d):
    n=int(np.ceil(np.log2(d)))
    for i in range(0,n):
        if ((( x >>  i) & 1) != (( y >>  i) & 1)):
            circuit.x(n-i)
            
def dFourier(circuit,d):
    n=int(np.ceil(np.log2(d)))-1  #start from 0
    for i in range(0,n):
        circuit.cp(np.pi/2**(n-i), i, n)
        
def dHadamard(circuit, d):
    n=int(np.ceil(np.log2(d)))
    for i in range(0,n):
        circuit.h(i)
    

#try  
d=6          
n=int(np.ceil(np.log2(d)))
qc = QuantumCircuit(n)
dLvlSwap(qc,3,4,d)
qc.draw()

qc = QuantumCircuit(n)
dFourier(qc,d)
qc.draw()

qc = QuantumCircuit(n)
dHadamard(qc,d)
qc.draw()
