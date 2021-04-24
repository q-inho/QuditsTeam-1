# QISKIT FOR HIGH DIMENSIONAL MULTIPARTITE QUANTUM STATES
Repository for Qudits team for IBM Hackathon

We aim to extend QiSkit's versatility to higher dimensional multipartite quantum states, ie, qudits, allowing access to the complex entanglement structure of various quantum systems. We test our idea for a family of graph states with applications in quantum cryptography, quantum communication, quantum simulation and quantum error correction.

We write the graph states in term of quantum circuits and identify useful d-dimension gates. We propose 2 approaches: 
1. We remap d dimension qu-dits to ceil(Log2(d)) qu-bits (practical for Schrodinger picture), and find the corresponding gate decomposition on QiSkit. 
2. We rewrite generalized Pauli matrices for our interested gates in d-dimension (in Heisenberg picture) to add to QiSkit basis.
