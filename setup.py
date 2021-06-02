# This code is from Qiskit Hackathon 2021 by the team
# Qiskit for high dimensional multipartite quantum states.
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

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIREMENTS = f.read().splitlines()

setup(
    name='qiskit-qudits',
    version='0.1',
    keywords="qiskit sdk quantum",
    packages=find_packages(exclude=['test*', 'heisenberg']),
    url='https://github.com/Vanimiaou/QuditsTeam',
    license='Apache 2.0',
    author='Tim Alexis Körner',
    author_email='tim.alexis.koerner@gmail.com',
    description='IBM Qiskit-Terra extension for qudits, simulated via multiple qubits',
    install_requires=REQUIREMENTS
)
