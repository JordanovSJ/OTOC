import qiskit
import numpy
import matplotlib.pyplot as plt
import ray
from functools import partial
import pandas

from functions import *


if __name__ == "__main__":

    max_entangling = False  # this controls if we want to use max. entangling U operation
    # noise = False
    average_shots = 50

    A_qubit = 0
    B_qubit = 1

    n_max_qubits = 8

    otoc_values_dict = {}

    ray.init(num_cpus=6)
    for n_qubits in range(2, n_max_qubits+1):

        print(n_qubits)
        ancilla_qubits = list(numpy.arange(n_qubits - 2) + 2)

        # get initial state (|0>+|1>)|0>
        initial_state_qasm = ('h q[{}];\n'.format(A_qubit))

        otoc_values = []

        times = numpy.arange(101)/100

        for time in times:
            print('time ', time)

            @ray.remote
            def get_otoc_value():
                from functions import qasm_header, otoc_circuit, get_statevector_from_qasm
                from entangling_operations import random_entangling_2

                header = qasm_header(2 + len(ancilla_qubits))
                otoc_qasm = otoc_circuit(time, A_qubit, B_qubit, ancilla_qubits, f_U=random_entangling_2)
                qasm = header + initial_state_qasm + otoc_qasm + initial_state_qasm
                statevector = get_statevector_from_qasm(qasm)
                return statevector[0]

            ray_ids = [get_otoc_value.remote() for i in range(average_shots)]
            otoc_value = sum(ray.get(ray_ids)) / average_shots
            otoc_values.append(otoc_value)

        otoc_values_dict['{} qubits'.format(n_qubits)] = otoc_values

        # plotting
        plt.subplot(n_max_qubits - 1, 1, n_qubits - 1)
        plt.plot(times, otoc_values, '*', color='b', label='no noise')
        if n_qubits == 2:
            if max_entangling:
                plt.title('OTOC values. Max entangling')
            else:
                plt.title('OTOC values. Partial entangling')
        plt.legend()
        plt.ylabel('OTOC \n{} qubits'.format(n_qubits))
        if n_qubits != n_max_qubits:
            plt.xticks(times, " ")  # cheat
        plt.ylim(-1.1, 1.1)

    ray.shutdown()
    plt.xlabel(r't, [$\pi$]')
    plt.show()

    otoc_values_dict['times'] = times
    pandas.DataFrame.from_dict(otoc_values_dict).to_csv('results/otoc_data.csv')
    print('Ciao')
