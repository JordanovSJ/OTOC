import qiskit
import numpy
import matplotlib.pyplot as plt
import ray
from functools import partial
import pandas
import itertools

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


def otoc_informative_measures_circuit(qubit_a, qubit_b, n_qubits, ancillas, phis, time):
    # A = X, B = Z
    qasm = ['']
    U, U_conj = max_entangling_gate(time, list(numpy.arange(n_qubits)))
    # A
    qasm.append(x_informative_measure(qubit_a, ancillas[0], phis['phi_a']))
    # U(t)
    qasm.append(U)
    # B
    qasm.append(z_informative_measure(qubit_b, ancillas[1], phis['phi_b']))
    # U(-t)
    qasm.append(U_conj)
    # A'
    qasm.append(x_informative_measure(qubit_a, ancillas[2], phis['phi_a2']))

    return ''.join(qasm)


def otoc_informative_measures_value(time, qubit_a=0, qubit_b=1, ancillas=None, n_qubits=2, noise=False, phis=None,
                                    shots_per_experiment=2000, noise_model=None):

    if ancillas is None:
        ancillas = [n_qubits, n_qubits + 1, n_qubits + 2]

    otoc_value = 0

    if phis is None:
        phis = {'phi_a': numpy.pi / 2, 'phi_b': numpy.pi / 2, 'phi_a2': numpy.pi / 2}

    header_qasm = qasm_header(n_qubits + 3)
    initial_state_qasm = 'h q[{}];\n'.format(qubit_a)
    otoc_circuit_qasm = otoc_informative_measures_circuit(qubit_a, qubit_b, n_qubits, ancillas, phis, time)
    # otoc_circuit_qasm = ''
    measurements_qasm = qasm_measurements_all_qubits(n_qubits + 3)

    qasm = header_qasm + initial_state_qasm + otoc_circuit_qasm + measurements_qasm

    state_counts = get_measurement_counts(qasm, noise=noise, shots=shots_per_experiment, noise_model=noise_model)

    info_meas_1 = [0, 0]
    info_meas_2 = [0, 0]
    info_meas_3 = [0, 0]

    for state in state_counts.keys():
        count = state_counts[state]

        if int(state[n_qubits]):
            info_meas_1[1] += count
        else:
            info_meas_1[0] += count

        if int(state[n_qubits + 1]):
            info_meas_2[1] += count
        else:
            info_meas_2[0] += count

        if int(state[n_qubits + 2]):
            info_meas_3[1] += count
        else:
            info_meas_3[0] += count

    p_test = 0
    for ancilla_outcomes in itertools.product([0, 1], repeat=3):
        p = info_meas_1[ancilla_outcomes[0]] * info_meas_2[ancilla_outcomes[1]] * info_meas_3[ancilla_outcomes[2]] \
            / (sum(info_meas_1) * sum(info_meas_2) * sum(info_meas_3))
        p_test += p
        term_value = \
            ((-1) ** (ancilla_outcomes[0] + ancilla_outcomes[2]) / (
                        numpy.sin(phis['phi_a']) * numpy.sin(phis['phi_a2'])) -
             numpy.cos(phis['phi_b'] / 2) ** 2) * p  # / numpy.sin(phis['phi_b'] / 2) ** 2  # move at the end
        otoc_value += term_value

    # print('p_test = ', p_test)
    return otoc_value / numpy.sin(phis['phi_b'] / 2) ** 2


if __name__ == "__main__":

    noise = True
    shots_per_experiment = 50000
    time_points = 25
    n_max_qubits = 7

    phis = {'phi_a': numpy.pi/4, 'phi_b': numpy.pi/4, 'phi_a2': numpy.pi/4}
    noise_model = device_noise_model()

    qubit_A = 0
    qubit_B = 1

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    times = numpy.arange(time_points)*numpy.pi/time_points

    plot_title = True
    # ray.init(num_cpus=3)
    for n_qubits in numpy.arange(n_max_qubits-2)+2:
        print('n_qunits=', n_qubits)
        # three ancilla qubits to perform the informative measure A,B,A'
        ancilla_qubit_a = n_qubits
        ancilla_qubit_b = n_qubits + 1
        ancilla_qubit_a_2 = n_qubits + 2
        ancillas = [ancilla_qubit_a, ancilla_qubit_b, ancilla_qubit_a_2]

        for noise in [True, False]:
            otoc_values = []
            for time in times:
                # print('time: ', time)
                otoc_value = otoc_informative_measures_value(time, qubit_a=qubit_A, qubit_b=qubit_B, n_qubits=n_qubits,
                                                             noise=noise, ancillas=ancillas, phis=phis,
                                                             shots_per_experiment=shots_per_experiment,
                                                             noise_model=noise_model)
                otoc_values.append(otoc_value)

            # @ ray.remote()
            # def get_otoc_values_in_parallel(t):
            #     otoc = otoc_informative_measures_value(t,  n_qubits=n_qubits, noise=noise, phis=phis)
            #     return otoc
            #
            # rai_ids = [[time, get_otoc_values_in_parallel.remote(t=time)] for time in times]
            #
            # rai_ids.sort()
            # otoc_values = [[ray.get(rai_id[1])] for rai_id in rai_ids]

            plt.subplot(n_max_qubits - 1, 1, n_qubits - 1)
            plt.plot(times/numpy.pi, otoc_values, label='Noise={}'.format(noise))
            plt.ylabel('OTOC \n n={}'.format(n_qubits))
            if plot_title:
                plt.title(r'OTOC value: $\phi_a$={}, $\phi_b$={}, $\phi_a^\prime$={}'
                          .format((phis['phi_a'] / numpy.pi).__round__(2),
                                  (phis['phi_b'] / numpy.pi).__round__(2),
                                  (phis['phi_a2'] / numpy.pi).__round__(2)))
                plot_title = False
    # ray.shutdown()
    plt.xlabel(r'time, [$\pi]$')
    plt.legend()
    plt.show()

    print('Ciao')
