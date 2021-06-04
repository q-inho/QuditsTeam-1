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

from qiskit.exceptions import QiskitError




#PauliND() is the class for Pauli operators in Weyl-Heisenberg group
#taking arguments the list of 3 orders of the generators elements w_N (phase), P_N and Q_N (symplectic) and the dimension. 
#
#PauliND().PauliList check the phase and symplectic orders for an operator. 
#
#PauliND().PauliProduct calculate the product of two Pauli operators.

class PauliND(): 
    def PauliOrdersList(self,SymplecticList,dimension):
        if not isinstance(dimension, int):
            raise QiskitError('Dimension is not an integer')
        if not isinstance(SymplecticList, list):
            raise QiskitError('Invalid phase and symplectic orders format. Please enter a list of length 3.')
        elif len(SymplecticList)!=3:
            raise QiskitError('Invalid phase and symplectic orders format. Please enter a list of length 3.')
        elif max(SymplecticList)> dimension or min(SymplecticList)<0:
            print('Generalized Pauli matrices have orders as their dimension. Orders will be modulo by dimension.')
        SymplecticList=[i % dimension for i in SymplecticList]
        return SymplecticList
    def PauliProduct(self,SymplecticList1,SymplecticList2,dimension):
        SymplecticList1=PauliND().PauliOrdersList(SymplecticList1,dimension)
        SymplecticList1=PauliND().PauliOrdersList(SymplecticList2,dimension)
        SymplecticProduct=[SymplecticList1[0]+SymplecticList2[0]+SymplecticList1[2]+SymplecticList2[1],
                           SymplecticList1[1]+SymplecticList2[1],
                           SymplecticList1[2]+SymplecticList2[2]]
        return PauliND().PauliOrdersList(SymplecticProduct,dimension)




#ClifforND() is the class for Clifford operators.
#
#ClifforND().CliffordTable() take argument OutputQ as the Pauli string of the output gate after applying Clifford action on Pauli [0,1,0]
#argument OutputP as the Pauli string of the output gate after applying Clifford action on Pauli [0,0,1] and the dimension.
#
#ClifforND().W() gives the first generator of single qubit Clifford group
#
#CliffordND().D() gives the other generator of single qubit Clifford group

class CliffordND():
    def CliffordTable(self,Output,dimension):
        if not isinstance(dimension, int):
            raise QiskitError('Dimension is not an integer')
        if not isinstance(Output, list):
            raise QiskitError('Invalid Clifford definition.'
                              ' Please enter the tuple containing two output lists for the Clifford transformation on Pauli [0,1,0] and [0,0,1]')
        elif len(Output)!=2:
            raise QiskitError('Invalid Clifford definition.'
                              ' Please enter the tuple containing two output lists for the Clifford transformation on Pauli [0,1,0] and [0,0,1]')
        OutputQ=PauliND().PauliOrdersList(Output[0],dimension)
        OutputP=PauliND().PauliOrdersList(Output[1],dimension)
        return [OutputQ,OutputP]
    def W(self,dimension):
        return [[0,0,dimension-1],[0,1,0]]
    def D(self,dimension):
        return [[0,1,0],[0,1,1]]




#HeisenbergTransformND() is the class that apply a Clifford transformation to a Pauli operator

class HeisenbergTransformND():
    def __new__(self,SymplecticList,Output,dimension):
        PauliList=PauliND().PauliOrdersList(SymplecticList,dimension)
        CliffordGate=CliffordND().CliffordTable(Output,dimension)
        QOrderList=[0,0,0]
        POrderList=[0,0,0]
        for i in range(PauliList[1]):
            QOrderList=PauliND().PauliProduct(QOrderList,CliffordGate[0],dimension)
        for i in range(PauliList[2]):
            POrderList=PauliND().PauliProduct(POrderList,CliffordGate[1],dimension)
        PostCliffordList= PauliND().PauliProduct(QOrderList,POrderList,dimension)
        PostCliffordList[0]=(PostCliffordList[0]+PauliList[0]) % dimension
        return PostCliffordList
        
