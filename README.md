# QISKIT FOR HIGH DIMENSIONAL MULTIPARTITE QUANTUM STATES
Repository for Qudits team for IBM Hackathon

We aim to extend QiSkit's versatility to higher dimensional quantum states, ie, qudits, allowing access to the complex entanglement structure of various quantum systems. We test our idea for various gates with applications in quantum cryptography, quantum communication, quantum simulation and quantum error correction.

We propose to add to QiSkit the capacity to handle qudits following two approaches: 

1. In the Schrödinger picture; remapping d-dimension qudits to ⌈log₂(d)⌉ qubits, and find the decomposition of qudits gates into qubit gates. The result is the python package qiskit_qudits.
2. In the Heisenberg picture; using generalized Pauli matrices to write a simulator handling directly gates acting on qudits. The results is the stabilizer_simulator.

## Team members

Hoang Van Do, Tim Alexis Körner, Inho Choi, Timothé Presles and Élie Gouzien

## License

[Apache License 2.0](LICENSE.txt)
