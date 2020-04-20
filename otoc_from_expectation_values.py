import qiskit
import numpy
import matplotlib.pyplot as plt
import ray
from functools import partial
import pandas

from functions import *


def x_informative_measure(target, ancilla, phi):

    qasm = ['']
    qasm.append('ry({}) q[{}];\n'.format(phi, ancilla))
    qasm.append('cx q[{}], q[{}];\n'.format(ancilla, target))
    qasm.append('h q[{}];\n'.format(ancilla))

    return ''.join(qasm)


def z_informative_measure(target, ancilla, phi):

    qasm = ['']
    qasm.append('ry({}) q[{}];\n'.format(phi, ancilla))
    qasm.append('cz q[{}], q[{}];\n'.format(ancilla, target))
    qasm.append('h q[{}];\n'.format(ancilla))

    return ''.join(qasm)


def ancilla_measuremnts(ancillas):
    qasm = ['']
    for ancilla in ancillas:
        qasm.append('measure q[{0}] -> c[{0}];'.format(ancilla))

    return ''.join(qasm)


def otoc_with_info_measures(qubit_a, qubit_b, n_qubits, ancillas, phi, time):
    # A = X, B = Z
    qasm = ['']
    U, U_conj = entangling_operation(time, list(numpy.arange(n_qubits)))
    # A
    qasm.append(x_informative_measure(qubit_a, ancillas[0], phi))
    # U(t)
    qasm.append(U)
    # B
    qasm.append(z_informative_measure(qubit_b, ancillas[1], phi))
    # U(-t)
    qasm.append(U_conj)
    # A'
    qasm.append(x_informative_measure(qubit_a, ancillas[2], phi))

    return ''.join(qasm)


if __name__ == "__main__":

    noise = False
    shots_per_experiment = 1000

    qubit_A = 1
    qubit_B = 2

    # at least 2
    n_qubits = 6

    # three ancilla qubits to perform the informative measure A,B,A'
    ancilla_qubit_a = n_qubits
    ancilla_qubit_b = n_qubits + 1
    ancilla_qubit_a_2 = n_qubits + 2
    ancillas = [ancilla_qubit_a, ancilla_qubit_b, ancilla_qubit_a_2]

    # variables
    time = 0
    phi = numpy.pi/10

    header_qasm = qasm_header(n_qubits + 3)
    otoc_circuit_qasm = otoc_with_info_measures(qubit_A, qubit_B, n_qubits, ancillas, phi, time)
    measurements_qasm = ancilla_measuremnts(ancillas)

    qasm = header_qasm + otoc_circuit_qasm + measurements_qasm

    measurement_probs = get_measurement_probs(qasm, noise=noise, shots=shots_per_experiment)

    print(measurement_probs)

    print('Ciao')
